"""Data pipeline for the transformer plant-disease classifier.

The dataset, the stratified split CSVs and the diagnosis knowledge base are NOT
copied here. They are read (read-only) from the sibling CNN project so that the
CNN and the transformer are trained on provably identical data:

    ../Plant-Disease-Detection-CNN/data/PlantVillage   (20,636 images, 15 classes)
    ../Plant-Disease-Detection-CNN/data/split/*.csv    (stratified 80/10/10)
    ../Plant-Disease-Detection-CNN/data/diagnosis.json

Preprocessing deliberately mirrors the CNN pipeline (RGB, resize to 96x96,
scale to [0, 1], no mean/std normalisation) so the only difference between the
two experiments is the architecture.
"""

import hashlib
import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMG_SIZE = 96
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLING_ROOT = os.path.dirname(PROJECT_ROOT)
CNN_PROJECT_NAME = "Plant-Disease-Detection-CNN"

IMAGENET_STYLE = InterpolationMode.BICUBIC


def cnn_project_root():
    """Absolute path of the sibling CNN project that owns the dataset."""
    candidate = os.path.join(SIBLING_ROOT, CNN_PROJECT_NAME)
    return candidate if os.path.isdir(candidate) else None


def default_split_dir():
    """Locate the canonical split CSVs (sibling CNN project wins, local second)."""
    sibling = cnn_project_root()
    if sibling and os.path.isfile(os.path.join(sibling, "data", "split", "train.csv")):
        return os.path.join(sibling, "data", "split"), sibling
    local = os.path.join(PROJECT_ROOT, "data", "split")
    if os.path.isfile(os.path.join(local, "train.csv")):
        return local, PROJECT_ROOT
    raise FileNotFoundError(
        "Could not find the canonical split CSVs. Looked in "
        f"{os.path.join(sibling or '<no sibling CNN project>', 'data', 'split')} and "
        f"{local}. Pass --split-dir explicitly."
    )


def resolve_diagnosis_path():
    """The diagnosis knowledge base, read from the CNN project when not local."""
    local = os.path.join(PROJECT_ROOT, "data", "diagnosis.json")
    if os.path.isfile(local):
        return local
    sibling = cnn_project_root()
    if sibling:
        candidate = os.path.join(sibling, "data", "diagnosis.json")
        if os.path.isfile(candidate):
            return candidate
    return local


def load_class_indices(split_dir):
    with open(os.path.join(split_dir, "class_indices.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def resolve_path_root(split_dir, raw_paths):
    """Pick the directory that the split CSV paths are relative to.

    The canonical CSVs store ``data\\PlantVillage\\...`` relative to the CNN
    project root. Rather than assuming, score each candidate root by how many
    listed images actually resolve to files on disk.
    """
    candidates = [PROJECT_ROOT]
    sibling = cnn_project_root()
    if sibling:
        candidates.append(sibling)
    candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(split_dir))))

    seen, ordered = set(), []
    for c in candidates:
        key = os.path.normcase(os.path.abspath(c))
        if key not in seen:
            seen.add(key)
            ordered.append(os.path.abspath(c))

    sample = list(raw_paths)
    if len(sample) > 200:
        sample = sample[:: max(1, len(sample) // 200)]

    best_root, best_score = None, -1
    for root in ordered:
        score = 0
        for raw in sample:
            if os.path.isabs(raw):
                candidate = raw
            else:
                candidate = os.path.join(root, raw)
            if os.path.isfile(candidate):
                score += 1
        if score > best_score:
            best_root, best_score = root, score
    if best_score == 0:
        raise FileNotFoundError(
            "None of the candidate roots contain the images referenced by the split CSVs: "
            f"{ordered}"
        )
    return best_root


def load_splits(split_dir=None, verbose=True):
    """Load train/val/test with absolute image paths plus a data provenance record.

    The split CSVs store paths relative to the CNN project root (e.g.
    ``data\\PlantVillage\\...``), so relative entries are resolved against it.
    """
    if split_dir is None:
        split_dir, path_root = default_split_dir()
    else:
        split_dir = os.path.abspath(split_dir)
        path_root = None

    frames, all_raw = {}, []
    for name in ("train", "val", "test"):
        csv_path = os.path.join(split_dir, f"{name}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing split file: {csv_path}")
        df = pd.read_csv(csv_path)
        missing_cols = {"path", "label", "y"} - set(df.columns)
        if missing_cols:
            raise ValueError(f"{csv_path} is missing required column(s): {sorted(missing_cols)}")
        if path_root is None:
            frames[name] = df
            all_raw.extend(df["path"].astype(str).tolist())
            continue
        df = df.copy()
        df["path"] = [os.path.normpath(p if os.path.isabs(p) else os.path.join(path_root, p))
                      for p in df["path"].astype(str)]
        df["y"] = df["y"].astype(int)
        frames[name] = df

    if path_root is None:
        path_root = resolve_path_root(split_dir, all_raw)
        for df in frames.values():
            df["path"] = [os.path.normpath(p if os.path.isabs(p) else os.path.join(path_root, p))
                          for p in df["path"].astype(str)]
            df["y"] = df["y"].astype(int)

    provenance = verify_splits(frames, split_dir, path_root, verbose=verbose)
    return frames, provenance


def verify_splits(frames, split_dir, path_root, verbose=True):
    """Assert the splits are disjoint/exhaustive and fingerprint them."""
    seen = {}
    for name, df in frames.items():
        for path in df["path"]:
            if path in seen:
                raise ValueError(f"Image '{path}' appears in both '{seen[path]}' and '{name}' splits.")
            seen[path] = name

    missing = [p for p in seen if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images listed in the splits are missing, e.g. {missing[:3]}")

    class_indices = load_class_indices(split_dir)
    labels = {name: set(df["label"]) for name, df in frames.items()}
    union = set().union(*labels.values())
    if union != set(class_indices.values()):
        raise ValueError("class_indices.json does not match the labels present in the split CSVs.")

    provenance = {
        "split_dir": os.path.abspath(split_dir),
        "path_root": os.path.abspath(path_root),
        "num_classes": len(class_indices),
        "img_size": IMG_SIZE,
        "counts": {name: int(len(df)) for name, df in frames.items()},
        "class_distribution": {
            name: {k: int(v) for k, v in df["label"].value_counts().sort_index().items()}
            for name, df in frames.items()
        },
        "image_fingerprint": {},
        "all_images_exist": True,
        "splits_disjoint": True,
    }
    for name, df in frames.items():
        digest = hashlib.sha256(
            "\n".join(sorted(df["path"])).encode("utf-8")
        ).hexdigest()
        provenance["image_fingerprint"][name] = digest

    if verbose:
        counts = provenance["counts"]
        print(
            "Splits -> train {train} | val {val} | test {test} | classes {n}".format(
                n=provenance["num_classes"], **counts
            )
        )
    return provenance


class LeafDataset(Dataset):
    """PlantVillage leaf images stored as class sub-folders of JPEG files."""

    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = [int(y) for y in labels]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, self.labels[index]


def build_transforms(img_size=IMG_SIZE, train=False, randaug_magnitude=9):
    """Augmentation for training; plain resize for val/test and inference.

    The evaluation path matches the CNN's preprocessing exactly (squash resize
    to img_size, scale to [0, 1]) so test-time inputs are identical for both models.
    """
    if not train:
        return T.Compose([
            T.Resize((img_size, img_size), interpolation=IMAGENET_STYLE),
            T.ToTensor(),
        ])

    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.75, 1.333),
                            interpolation=IMAGENET_STYLE),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=randaug_magnitude),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.16), ratio=(0.3, 3.3), value=0),
    ])


def build_loaders(frames, img_size=IMG_SIZE, batch_size=64, workers=4, seed=42,
                  class_weights=None, balanced_sampler=False, randaug_magnitude=9):
    """Create train/val/test loaders. ``train`` loader is shuffled and augmented."""
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = LeafDataset(
        frames["train"]["path"], frames["train"]["y"],
        build_transforms(img_size, train=True, randaug_magnitude=randaug_magnitude),
    )
    eval_tf = build_transforms(img_size, train=False)
    val_ds = LeafDataset(frames["val"]["path"], frames["val"]["y"], eval_tf)
    test_ds = LeafDataset(frames["test"]["path"], frames["test"]["y"], eval_tf)

    sampler = None
    shuffle = True
    if balanced_sampler:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=class_weights, num_samples=len(train_ds), replacement=True, generator=g
        )
        shuffle = False

    common = dict(num_workers=workers, pin_memory=False, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, drop_last=True, generator=g, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def compute_class_weights(frames, num_classes):
    """Inverse-frequency weights, useful for the very imbalanced PlantVillage subset."""
    counts = torch.bincount(
        torch.tensor(frames["train"]["y"].tolist(), dtype=torch.long), minlength=num_classes
    ).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights / weights.mean()
