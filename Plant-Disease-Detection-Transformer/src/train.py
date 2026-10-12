"""Train a Vision Transformer plant-disease classifier on the PlantVillage subset.

Same data as the CNN project (same images, same stratified train/val/test split,
same 96x96 preprocessing) — only the architecture differs.

Usage:
    python src/train.py --epochs 30
    python src/train.py --arch vit_small --patch-size 8 --epochs 40
    python src/train.py --resume checkpoints/last.pt
"""

import argparse
import contextlib
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model, count_parameters
from preprocessing import (IMG_SIZE, build_loaders, compute_class_weights, cnn_project_root,
                           load_class_indices, load_splits)

MODEL_FILENAME = "vit_plant_disease.pt"


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Tee:
    """Mirror stdout into a log file (mirrors the CNN project's console logs)."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8", buffering=1) if path else None

    def write(self, text):
        sys.__stdout__.write(text)
        if self.file:
            self.file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        if self.file:
            self.file.flush()


class ModelEMA:
    """Exponential moving average of weights (standard for from-scratch ViT)."""

    def __init__(self, model, decay=0.9998):
        self.module = copy.deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def cpu_bf16_supported():
    checker = getattr(torch.cpu, "_is_avx512_bf16_supported", None)
    return bool(checker()) if callable(checker) else False


def amp_context(use_amp):
    """bfloat16 autocast for CPU training (~25% faster on AVX512-BF16 hardware)."""
    if not use_amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def build_optimizer(model, lr, weight_decay):
    """AdamW with no weight decay on biases, norms, cls token and positional embeddings."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("pos_embed") or name.endswith("cls_token"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))


def lr_lambda_factory(total_steps, warmup_steps, min_ratio=0.01):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return fn


def one_hot(y, num_classes):
    return torch.zeros(y.size(0), num_classes, dtype=torch.float32).scatter_(1, y.view(-1, 1), 1.0)


def rand_bbox(height, width, lam):
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = random.randint(0, height), random.randint(0, width)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    return y1, y2, x1, x2


def apply_mixup_cutmix(x, y, num_classes, mixup_alpha, cutmix_alpha, prob, smoothing):
    """Mixup / CutMix with soft targets; returns (x, soft_targets, lam)."""
    targets = one_hot(y, num_classes)
    if random.random() >= prob or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        soft = targets * (1 - smoothing) + smoothing / num_classes
        return x, soft, 1.0

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0))

    if use_cutmix:
        y1, y2, x1, x2 = rand_bbox(x.size(2), x.size(3), lam)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))
    else:
        x = x.mul_(lam).add_(x[perm], alpha=1.0 - lam)

    soft = targets.mul_(lam).add_(targets[perm], alpha=1.0 - lam)
    soft = soft * (1 - smoothing) + smoothing / num_classes
    return x, soft, lam


def soft_target_ce(logits, targets):
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, return_probs=False, use_amp=False):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    all_true, all_pred, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_context(use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        logits = logits.float()
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
        seen += y.size(0)
        all_true.append(y.cpu())
        all_pred.append(pred.cpu())
        if return_probs:
            all_probs.append(probs.cpu())

    out = {
        "loss": total_loss / max(1, seen),
        "accuracy": correct / max(1, seen),
    }
    if return_probs:
        out["true"] = torch.cat(all_true).numpy()
        out["pred"] = torch.cat(all_pred).numpy()
        out["probs"] = torch.cat(all_probs).numpy()
    return out


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, best, args, class_indices, meta):
    torch.save({
        "model": model.state_dict(),
        "ema": ema.module.state_dict() if ema else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best": best,
        "args": vars(args),
        "class_indices": class_indices,
        "meta": meta,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer plant-disease classifier")
    parser.add_argument("--split-dir", default=None, help="directory holding train/val/test CSVs (defaults to the CNN project's splits)")
    parser.add_argument("--out", default="models")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--arch", choices=["vit_tiny", "vit_small", "vit_base"], default="vit_tiny")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.2, help="Mixup alpha (0 disables)")
    parser.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha (0 disables)")
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--drop-path", type=float, default=0.05)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--attn-drop-rate", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--randaug-magnitude", type=int, default=9)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weighted", action="store_true", help="inverse-frequency loss weights")
    parser.add_argument("--balanced-sampler", action="store_true", help="weighted random sampling for the train split")
    parser.add_argument("--amp", choices=["auto", "bf16", "off"], default="auto",
                        help="CPU bfloat16 autocast (auto = on when the CPU supports AVX512-BF16)")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--time-budget-min", type=float, default=0.0, help="stop cleanly after this many minutes (0 = no limit)")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)
    log = Tee(os.path.join(args.reports, "train_console.log") if args.log_file is None else args.log_file)
    sys.stdout = log

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp == "bf16" or (args.amp == "auto" and device.type == "cpu" and cpu_bf16_supported())
    print("=" * 78)
    print("Plant Disease Detection - Vision Transformer training")
    print(f"device: {device} | torch threads: {torch.get_num_threads()} | seed: {args.seed}")
    print(f"bfloat16 autocast: {'on' if use_amp else 'off'}")
    sibling = cnn_project_root()
    print(f"data source: {'sibling CNN project (read-only): ' + sibling if sibling else 'local'}")
    print("=" * 78)

    frames, provenance = load_splits(args.split_dir)
    split_dir = provenance["split_dir"]
    class_indices = load_class_indices(split_dir)
    num_classes = provenance["num_classes"]

    train_loader, val_loader, test_loader = build_loaders(
        frames, img_size=args.img_size, batch_size=args.batch_size, workers=args.workers,
        seed=args.seed, randaug_magnitude=args.randaug_magnitude,
        class_weights=(compute_class_weights(frames, num_classes) if (args.class_weighted or args.balanced_sampler) else None),
        balanced_sampler=args.balanced_sampler,
    )

    model = build_model(
        arch=args.arch, img_size=args.img_size, patch_size=args.patch_size,
        num_classes=num_classes, drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path,
    ).to(device)
    tokens = model.num_patches + 1
    print(f"model: {args.arch} | patch {args.patch_size} | {tokens} tokens "
          f"({args.img_size // args.patch_size}x{args.img_size // args.patch_size}+1) | "
          f"depth {model.depth} | dim {model.embed_dim} | heads {model.blocks[0].attn.num_heads} | "
          f"params {count_parameters(model):,}")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * min(args.warmup_epochs, args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(total_steps, warmup_steps, args.min_lr_ratio)
    )
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    hard_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.class_weighted:
        weights = compute_class_weights(frames, num_classes).to(device)
        hard_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=weights)

    start_epoch, best = 1, {"val_accuracy": -1.0, "epoch": -1, "source": None}
    best_path = os.path.join(args.out, MODEL_FILENAME)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best", best)
        if ema and ckpt.get("ema"):
            ema.module.load_state_dict(ckpt["ema"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    history = {"epoch": [], "train_loss": [], "train_accuracy": [], "val_loss": [],
               "val_accuracy": [], "ema_val_loss": [], "ema_val_accuracy": [],
               "lr": [], "epoch_seconds": []}
    started = time.time()
    stop_reason = "completed"

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        run_loss, correct, seen = 0.0, 0, 0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, soft_targets, _ = apply_mixup_cutmix(
                x, y, num_classes, args.mixup, args.cutmix, args.mixup_prob, args.label_smoothing
            )

            with amp_context(use_amp):
                logits = model(x)
                loss = soft_target_ce(logits, soft_targets)
                if args.class_weighted:
                    loss = loss * soft_targets.sum(dim=1).mean()
            loss = loss.float()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema:
                ema.update(model, epoch * steps_per_epoch + step)

            run_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.size(0)

            if step % 25 == 0 or step == steps_per_epoch - 1:
                pct = 100.0 * (step + 1) / steps_per_epoch
                print(f"  epoch {epoch:3d} [{pct:5.1f}%] step {step + 1:4d}/{steps_per_epoch} "
                      f"loss {loss.item():.4f} lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        train_loss = run_loss / max(1, seen)
        train_acc = correct / max(1, seen)
        val = evaluate(model, val_loader, hard_criterion, device, num_classes, use_amp=use_amp)
        ema_val = (evaluate(ema.module, val_loader, hard_criterion, device, num_classes, use_amp=use_amp)
                   if ema else {"loss": float("nan"), "accuracy": -1.0})
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["ema_val_loss"].append(ema_val["loss"])
        history["ema_val_accuracy"].append(ema_val["accuracy"])
        history["lr"].append(current_lr)
        history["epoch_seconds"].append(elapsed)

        total_minutes = (time.time() - started) / 60.0
        print(f"EPOCH {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
              f"| ema val loss {ema_val['loss']:.4f} acc {ema_val['accuracy']:.4f} "
              f"| {elapsed / 60:.1f} min | total {total_minutes:.1f} min", flush=True)

        for source, metrics in (("raw", val), ("ema", ema_val)):
            if metrics["accuracy"] > best["val_accuracy"]:
                best = {"val_accuracy": metrics["accuracy"], "val_loss": metrics["loss"],
                        "epoch": epoch, "source": source}
                state = model.state_dict() if source == "raw" else ema.module.state_dict()
                torch.save(state, best_path)
                print(f"  new best ({source}) val acc {metrics['accuracy']:.4f} -> {best_path}")

        save_checkpoint(os.path.join(args.out, "last_checkpoint.pt"), model, ema, optimizer,
                        scheduler, epoch, best, args, class_indices, provenance)
        with open(os.path.join(args.reports, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if args.time_budget_min and total_minutes >= args.time_budget_min:
            stop_reason = f"time budget of {args.time_budget_min} min reached"
            print(f"STOPPING: {stop_reason}")
            break

    print(f"Training finished after {len(history['epoch'])} epochs ({stop_reason}).")
    print(f"Best validation accuracy {best['val_accuracy']:.4f} at epoch {best['epoch']} ({best['source']} weights)")

    final = build_model(arch=args.arch, img_size=args.img_size, patch_size=args.patch_size,
                        num_classes=num_classes, drop_rate=args.drop_rate,
                        attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path).to(device)
    final.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test = evaluate(final, test_loader, hard_criterion, device, num_classes, return_probs=True, use_amp=use_amp)
    print(f"TEST accuracy {test['accuracy']:.4f} | loss {test['loss']:.4f}")

    names = [class_indices[i] for i in range(num_classes)]
    report = classification_report(test["true"], test["pred"], target_names=names, digits=4, zero_division=0)
    print(report)
    cm = confusion_matrix(test["true"], test["pred"])

    np.save(os.path.join(args.reports, "confusion_matrix.npy"), cm)
    with open(os.path.join(args.reports, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 10))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Vision Transformer - Confusion Matrix (test acc {test['accuracy']:.4f})")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.reports, "confusion_matrix.png"), dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"Could not render confusion matrix plot: {exc}")

    with open(os.path.join(args.out, "class_indices.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in class_indices.items()}, f, indent=2)
    config = {
        "architecture": "VisionTransformer",
        "arch": args.arch,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "num_classes": num_classes,
        "num_patches": model.num_patches,
        "tokens": tokens,
        "embed_dim": model.embed_dim,
        "depth": model.depth,
        "params": count_parameters(model),
        "selected_weights": best["source"],
        "best_epoch": best["epoch"],
        "val_accuracy": best["val_accuracy"],
        "test_accuracy": test["accuracy"],
        "test_loss": test["loss"],
        "epochs_trained": len(history["epoch"]),
        "stop_reason": stop_reason,
        "data_source": provenance,
    }
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.reports, "data_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print(f"Artifacts saved -> {os.path.abspath(args.out)}")
    print("Done.")


if __name__ == "__main__":
    main()
