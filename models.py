"""
models.py — Network architectures for PatchCL-AE
==================================================

Implements the four core components (Tables 1, 2, & 3 of the paper):

  1. **Encoder (E)**       — 5 blocks with 3×3 convolutions, InstanceNorm, ReLU.
                             The first block (E1) uses a 7×7 convolution.
                             Extracts multi-scale feature maps for contrastive learning.

  2. **Decoder (De)**      — 4 blocks with bilinear upsampling + convolution to
                             reconstruct the image at the original 256×256 resolution.

  3. **Discriminator (D)** — 9-layer PatchGAN-style network with LeakyReLU to
                             provide an adversarial global image loss (L_Img).

  4. **Projection Head (P)** — 2-layer MLP that maps features from different
                             encoder layers into a 256-dimensional space for
                             patch-wise contrastive learning.

Key design notes:
  - InstanceNorm is used consistently (not BatchNorm) to prevent training
    instability, as recommended for image-to-image tasks.
  - The Encoder exposes intermediate feature maps (from each block) so the
    Projection Head and contrastive loss can operate at multiple scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Encoder (E) — Table 1
# ============================================================================

class EncoderBlock(nn.Module):
    """Single encoder block: Conv → InstanceNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """
    Encoder network with 5 blocks (E1–E5).

    E1: 7×7 conv, stride 2 → 64 channels   (H/2 × W/2)
    E2: 3×3 conv, stride 2 → 128 channels   (H/4 × W/4)
    E3: 3×3 conv, stride 2 → 256 channels   (H/8 × W/8)
    E4: 3×3 conv, stride 2 → 512 channels   (H/16 × W/16)
    E5: 3×3 conv, stride 2 → 512 channels   (H/32 × W/32)

    Returns a list of feature maps [e1, e2, e3, e4, e5] for multi-scale
    contrastive learning AND the final representation for the decoder.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # E1 uses a 7×7 kernel to capture larger receptive field early on
        self.e1 = EncoderBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.e2 = EncoderBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.e3 = EncoderBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.e4 = EncoderBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.e5 = EncoderBlock(512, 512, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale features [e1, e2, e3, e4, e5]."""
        e1 = self.e1(x)   # (B, 64,  128, 128)
        e2 = self.e2(e1)  # (B, 128, 64,  64)
        e3 = self.e3(e2)  # (B, 256, 32,  32)
        e4 = self.e4(e3)  # (B, 512, 16,  16)
        e5 = self.e5(e4)  # (B, 512, 8,   8)
        return [e1, e2, e3, e4, e5]


# ============================================================================
# Decoder (De) — Table 2
# ============================================================================

class DecoderBlock(nn.Module):
    """Single decoder block: Upsample → Conv → InstanceNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    """
    Decoder network with 4 upsampling blocks + a final output head.

    De1: 512 → 256  (8×8   → 16×16)
    De2: 256 → 128  (16×16 → 32×32)
    De3: 128 → 64   (32×32 → 64×64)
    De4: 64  → 64   (64×64 → 128×128)
    Out: Upsample → Conv → Tanh  (128×128 → 256×256 → 3ch, [-1,1] range mapped to [0,1])
    """

    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.de1 = DecoderBlock(512, 256)
        self.de2 = DecoderBlock(256, 128)
        self.de3 = DecoderBlock(128, 64)
        self.de4 = DecoderBlock(64, 64)

        # Final output layer: upsample to original resolution and map to RGB
        self.out_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),  # output in [-1, 1] to match normalised input range
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Takes the encoder feature list and decodes from the deepest (e5).
        Returns the reconstructed image X_hat of shape (B, 3, 256, 256).
        """
        z = features[-1]       # e5: (B, 512, 8, 8)
        z = self.de1(z)        # (B, 256, 16, 16)
        z = self.de2(z)        # (B, 128, 32, 32)
        z = self.de3(z)        # (B, 64,  64, 64)
        z = self.de4(z)        # (B, 64, 128, 128)
        x_hat = self.out_layer(z)  # (B, 3, 256, 256)
        return x_hat


# ============================================================================
# Discriminator (D) — Table 3 (PatchGAN style, 9 layers with LeakyReLU)
# ============================================================================

class Discriminator(nn.Module):
    """
    9-layer Discriminator with LeakyReLU for adversarial global image loss.

    Architecture (D1–D9):
      D1: Conv 4×4 s2 → 64   + LeakyReLU
      D2: Conv 4×4 s2 → 128  + InstanceNorm + LeakyReLU
      D3: Conv 4×4 s2 → 256  + InstanceNorm + LeakyReLU
      D4: Conv 4×4 s2 → 512  + InstanceNorm + LeakyReLU
      D5: Conv 4×4 s1 → 512  + InstanceNorm + LeakyReLU
      D6: Conv 4×4 s1 → 512  + InstanceNorm + LeakyReLU
      D7: Conv 4×4 s1 → 512  + InstanceNorm + LeakyReLU
      D8: Conv 4×4 s1 → 512  + InstanceNorm + LeakyReLU
      D9: Conv 4×4 s1 → 1    (patch-level real/fake prediction)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def disc_block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                                padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # D1–D8: convolutional feature extractor
        self.features = nn.Sequential(
            disc_block(in_channels, 64, stride=2, norm=False),  # D1
            disc_block(64, 128, stride=2),                       # D2
            disc_block(128, 256, stride=2),                      # D3
            disc_block(256, 512, stride=2),                      # D4
            disc_block(512, 512, stride=1),                      # D5
            disc_block(512, 512, stride=1),                      # D6
            disc_block(512, 512, stride=1),                      # D7
            disc_block(512, 512, stride=1),                      # D8
        )

        # D9: global average pool → scalar prediction per image
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.classifier(h)


# ============================================================================
# Projection Head (P) — 2-layer MLP → 256-d
# ============================================================================

class ProjectionHead(nn.Module):
    """
    2-layer MLP that projects feature vectors from any encoder layer into a
    common 256-dimensional embedding space for contrastive learning.

    Architecture:
        Linear(in_dim, 256) → ReLU → Linear(256, 256)

    The projection head is crucial because contrastive learning operates in
    this shared space: it allows patch embeddings from different encoder
    scales to be compared directly. By learning local semantic embeddings,
    the model can later detect anomalies as patches whose reconstructed
    embeddings deviate from the original — capturing semantic discrepancies
    rather than mere pixel-level differences.
    """

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiScaleProjectionHead(nn.Module):
    """
    Wraps one ProjectionHead per encoder layer, handling the varying
    channel dimensions (64, 128, 256, 512, 512 for E1–E5).
    """

    def __init__(self, encoder_channels: list[int] = None, proj_dim: int = 256):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 512]
        self.heads = nn.ModuleList([
            ProjectionHead(ch, proj_dim) for ch in encoder_channels
        ])

    def forward(self, features: list[torch.Tensor],
                sample_ids: list[torch.Tensor] | None = None
                ) -> list[torch.Tensor]:
        """
        Project features from each encoder layer.

        Args:
            features:   List of feature maps [e1, …, e5], each (B, C, H, W).
            sample_ids: Optional list of spatial indices to sample per layer.
                        Each element is a LongTensor of shape (N_s,) indexing
                        flattened spatial positions.

        Returns:
            List of projected vectors, one per layer.
            Each is (B, N_s, proj_dim) if sample_ids given, else (B, H*W, proj_dim).
        """
        projections = []
        for i, (feat, head) in enumerate(zip(features, self.heads)):
            B, C, H, W = feat.shape
            # Reshape to (B, C, H*W) then transpose to (B, H*W, C)
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

            if sample_ids is not None and i < len(sample_ids):
                ids = sample_ids[i]  # (N_s,)
                # Sample the same spatial locations across the batch
                feat_flat = feat_flat[:, ids, :]  # (B, N_s, C)

            proj = head(feat_flat)  # (B, N_s, proj_dim)
            projections.append(proj)

        return projections


# ============================================================================
# Helper: generate consistent spatial sample indices
# ============================================================================

def sample_patch_ids(feature_sizes: list[tuple[int, int]],
                     num_samples: int = 256,
                     device: torch.device = torch.device("cpu")) -> list[torch.Tensor]:
    """
    For each encoder layer, randomly sample *num_samples* spatial locations.
    The same locations are used for the original and reconstructed image to
    maintain spatial consistency (as required by the contrastive objective).

    Args:
        feature_sizes: List of (H, W) tuples for each encoder feature map.
        num_samples:   Number of patches to sample per layer (paper uses 256).
        device:        Target device for the index tensors.

    Returns:
        List of LongTensors, each of shape (num_samples,), containing
        random        python main.py --epochs 50 --batch-size 4        python main.py --epochs 50 --batch-size 4ly chosen flattened spatial indices.
    """
    ids = []
    for (H, W) in feature_sizes:
        total = H * W
        n = min(num_samples, total)
        perm = torch.randperm(total, device=device)[:n]
        ids.append(perm)
    return ids
