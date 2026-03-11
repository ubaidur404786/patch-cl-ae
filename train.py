"""
train.py — Training pipeline for PatchCL-AE (Algorithm 1)
==========================================================

Training overview (Algorithm 1 from the paper):
  1. Feed a noisy image X̃ = X + ε through the Encoder → multi-scale features.
  2. Decode the deepest features to produce X̂ (reconstructed / "normalised" image).
  3. Feed X̂ back through the Encoder to get reconstructed features.
  4. Sample 256 random spatial locations per layer (consistent between X and X̂)
     for contrastive feature calculation — ensuring spatial consistency.
  5. Project sampled features through the Projection Head.
  6. Compute L_Patch (patch-wise contrastive loss) between original and
     reconstructed projections.
  7. Feed X and X̂ to the Discriminator to compute L_Img (adversarial loss).
  8. Update Generator (E + De) with  L = L_Patch + λ * L_Img.
  9. Update Discriminator with its own adversarial loss.

Optimiser: Adam with lr=0.002, β₁=0.9, β₂=0.999, ε=1e-4.
"""

import os
import json
import csv

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import (
    Encoder, Decoder, Discriminator, MultiScaleProjectionHead,
    sample_patch_ids,
)
from losses import AdversarialLoss, PatchContrastiveLoss


def train_one_epoch(
    encoder: Encoder,
    decoder: Decoder,
    discriminator: Discriminator,
    proj_head: MultiScaleProjectionHead,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    train_loader,
    adv_criterion: AdversarialLoss,
    patch_criterion: PatchContrastiveLoss,
    device: torch.device,
    lambda_adv: float = 1.0,
    num_patch_samples: int = 256,
    scaler_G: GradScaler | None = None,
    scaler_D: GradScaler | None = None,
):
    """
    Run one training epoch according to Algorithm 1.
    Supports mixed-precision (AMP) when GradScalers are provided.

    Returns:
        dict with average loss values for the epoch.
    """
    encoder.train()
    decoder.train()
    discriminator.train()
    proj_head.train()

    use_amp = scaler_G is not None

    running = {"loss_G": 0.0, "loss_D": 0.0,
               "loss_patch": 0.0, "loss_adv_G": 0.0}
    count = 0

    for noisy, clean in tqdm(train_loader, desc="  Training", leave=False):
        noisy = noisy.to(device)
        clean = clean.to(device)
        B = clean.size(0)

        # ==================================================================
        # Steps 1-8: Generator forward + backward (under AMP)
        # ==================================================================
        with autocast(device_type="cuda", enabled=use_amp):
            feats_orig = encoder(clean)
            feats_noisy = encoder(noisy)
            x_hat = decoder(feats_noisy)
            feats_recon = encoder(x_hat)

            feat_sizes = [(f.shape[2], f.shape[3]) for f in feats_orig]
            patch_ids = sample_patch_ids(feat_sizes, num_samples=num_patch_samples,
                                         device=device)

            proj_orig = proj_head(feats_orig, sample_ids=patch_ids)
            proj_recon = proj_head(feats_recon, sample_ids=patch_ids)

            loss_patch = patch_criterion(proj_recon, proj_orig)
            pred_fake = discriminator(x_hat)
            loss_adv_G = adv_criterion(pred_fake, target_is_real=True)
            loss_G = loss_patch + lambda_adv * loss_adv_G

        opt_G.zero_grad()
        if use_amp:
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
        else:
            loss_G.backward()
            opt_G.step()

        # ==================================================================
        # Step 9: Discriminator update (under AMP)
        # ==================================================================
        with autocast(device_type="cuda", enabled=use_amp):
            pred_real = discriminator(clean)
            loss_D_real = adv_criterion(pred_real, target_is_real=True)
            pred_fake_detach = discriminator(x_hat.detach())
            loss_D_fake = adv_criterion(pred_fake_detach, target_is_real=False)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

        opt_D.zero_grad()
        if use_amp:
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
        else:
            loss_D.backward()
            opt_D.step()

        # Accumulate
        running["loss_G"] += loss_G.item() * B
        running["loss_D"] += loss_D.item() * B
        running["loss_patch"] += loss_patch.item() * B
        running["loss_adv_G"] += loss_adv_G.item() * B
        count += B

    return {k: v / max(count, 1) for k, v in running.items()}


def train(
    data_root: str = "./data",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 0.002,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-4,
    lambda_adv: float = 1.0,
    noise_std: float = 0.05,
    num_patch_samples: int = 256,
    save_dir: str = "./checkpoints",
    results_dir: str = "./results",
    device: str | None = None,
):
    """
    Full training loop for PatchCL-AE.

    Returns the trained models and the test data loader for downstream evaluation.
    """
    from dataset import get_dataloaders

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, test_loader = get_dataloaders(
        data_root=data_root, batch_size=batch_size,
        num_workers=2, noise_std=noise_std,
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    encoder = Encoder(in_channels=3).to(device)
    decoder = Decoder(out_channels=3).to(device)
    discriminator = Discriminator(in_channels=3).to(device)
    proj_head = MultiScaleProjectionHead().to(device)

    # ------------------------------------------------------------------
    # Optimisers — Adam with paper hyper-parameters
    # ------------------------------------------------------------------
    gen_params = (list(encoder.parameters())
                  + list(decoder.parameters())
                  + list(proj_head.parameters()))
    opt_G = torch.optim.Adam(gen_params, lr=lr, betas=(beta1, beta2), eps=eps)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr,
                             betas=(beta1, beta2), eps=eps)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    adv_criterion = AdversarialLoss()
    patch_criterion = PatchContrastiveLoss(temperature=0.07)

    # ------------------------------------------------------------------
    # Mixed-precision scalers (saves ~50% GPU memory)
    # ------------------------------------------------------------------
    use_amp = device.type == "cuda"
    scaler_G = GradScaler("cuda", enabled=use_amp)
    scaler_D = GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    history = []  # list of dicts, one per epoch

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        metrics = train_one_epoch(
            encoder, decoder, discriminator, proj_head,
            opt_G, opt_D, train_loader,
            adv_criterion, patch_criterion,
            device=device,
            lambda_adv=lambda_adv,
            num_patch_samples=num_patch_samples,
            scaler_G=scaler_G if use_amp else None,
            scaler_D=scaler_D if use_amp else None,
        )

        print(f"  loss_G={metrics['loss_G']:.4f}  loss_D={metrics['loss_D']:.4f}  "
              f"loss_patch={metrics['loss_patch']:.4f}  loss_adv_G={metrics['loss_adv_G']:.4f}")

        # Record history
        record = {"epoch": epoch, **metrics}
        history.append(record)

        # Save history after every epoch (JSON + CSV)
        _save_history(history, results_dir)

        # Save checkpoint every 10 epochs and at the end
        if epoch % 10 == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "discriminator": discriminator.state_dict(),
                "proj_head": proj_head.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }
            path = os.path.join(save_dir, f"patchcl_ae_epoch{epoch}.pt")
            torch.save(ckpt, path)
            print(f"  Checkpoint saved -> {path}")

    # Plot training curves
    _plot_training_curves(history, results_dir)

    return encoder, decoder, proj_head, test_loader, device


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _save_history(history: list[dict], results_dir: str):
    """Save training history as both JSON and CSV."""
    # JSON
    json_path = os.path.join(results_dir, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    # CSV
    csv_path = os.path.join(results_dir, "training_history.csv")
    if history:
        keys = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)


def _plot_training_curves(history: list[dict], results_dir: str):
    """Generate and save training loss curve plots."""
    epochs = [h["epoch"] for h in history]
    loss_G = [h["loss_G"] for h in history]
    loss_D = [h["loss_D"] for h in history]
    loss_patch = [h["loss_patch"] for h in history]
    loss_adv_G = [h["loss_adv_G"] for h in history]

    # --- Figure 1: All losses on one plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss_G, label="Loss_G (total generator)", linewidth=2)
    ax.plot(epochs, loss_D, label="Loss_D (discriminator)", linewidth=2)
    ax.plot(epochs, loss_patch, label="Loss_Patch (contrastive)", linewidth=2)
    ax.plot(epochs, loss_adv_G, label="Loss_Adv_G (adversarial)", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("PatchCL-AE Training Losses", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_losses_all.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: Separate subplots for clarity ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    titles = ["Generator Total Loss", "Discriminator Loss",
              "Patch Contrastive Loss", "Adversarial Generator Loss"]
    data = [loss_G, loss_D, loss_patch, loss_adv_G]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    for ax, title, vals, color in zip(axes.flat, titles, data, colors):
        ax.plot(epochs, vals, color=color, linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    fig.suptitle("PatchCL-AE Training Curves (per component)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_losses_separate.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: G vs D balance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss_G, label="Generator", color="#2196F3", linewidth=2)
    ax.plot(epochs, loss_D, label="Discriminator", color="#F44336", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Generator vs Discriminator Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_G_vs_D.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] Training plots saved -> {results_dir}")
