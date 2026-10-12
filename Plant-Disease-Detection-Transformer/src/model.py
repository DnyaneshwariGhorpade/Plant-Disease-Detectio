"""Vision Transformer (ViT) for plant-disease classification.

Pure transformer encoder over image patches — no convolutions in the feature
extractor. The patch embedding is a strided Conv2d (the standard ViT way of
slicing non-overlapping patches), everything after it is multi-head
self-attention + MLP blocks.

Default is ViT-Tiny sized for 96x96 inputs: a 8x8 patch stride yields a 12x12
= 144 token grid, which keeps fine-grained leaf-lesion detail that a 16x16
patch (36 tokens) would average away on such small images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth: per-sample drop of residual branches."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


class Attention(nn.Module):
    """Multi-head self-attention using PyTorch's fused scaled-dot-product kernel."""

    def __init__(self, dim, num_heads=3, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
                 attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=96, patch_size=8, in_chans=3, num_classes=15,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, qkv_bias=True,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_patches = (img_size // patch_size) ** 2
        self.num_features = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] if depth > 1 else [drop_path_rate]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        b = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = self.pos_drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        return self.head(self.forward_features(x))


ARCH_PRESETS = {
    "vit_tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12),
}


def build_model(arch="vit_tiny", img_size=96, patch_size=8, num_classes=15,
                drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.05, mlp_ratio=4.0):
    if arch not in ARCH_PRESETS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {sorted(ARCH_PRESETS)}.")
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        mlp_ratio=mlp_ratio,
        **ARCH_PRESETS[arch],
    )
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
