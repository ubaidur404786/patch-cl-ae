"""
evaluate.py — Evaluation, anomaly scoring, and visualization for PatchCL-AE
=============================================================================

Anomaly detection strategy (Section 3.3 of the paper):
  Instead of using naïve pixel-level subtraction (which is sensitive to noise
  and geometric shifts), PatchCL-AE computes anomaly scores in the deep
  semantic feature space learned by contrastive training.

  1. **Patch-wise Anomaly Score (S_patch):**  For each spatial location in each
     encoder layer, feed both the original X and reconstruction X̂ through the
     Encoder + Projection Head.  The discrepancy is measured via
     *Cosine Similarity* between the projected embeddings f and f̂.
     Low similarity → high anomaly likelihood.

  2. **Multi-scale Fusion:**  Resize the anomaly score map from each encoder
     layer to the original image resolution (256×256) and *sum* them to create
     the final Anomaly Map M.  Fusing multiple scales captures anomalies
     ranging from fine textures to coarse structural changes.

Produces the following saved outputs (all in save_dir):
  - patchcl_ae_results.png        — Per-sample: Original / Reconstruction / Heatmap
  - roc_curve.png                 — ROC curve with AUC annotation
  - score_distribution.png        — Histogram of anomaly scores (Normal vs Anomaly)
  - confusion_matrix.png          — Confusion matrix at optimal threshold
  - metrics_summary.csv           — All scalar metrics in a CSV table
  - evaluation_metrics.json       — Same metrics as machine-readable JSON
  - per_class_examples.png        — Grid of normal vs anomaly examples with scores
"""

import os
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix as sk_confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as MplNormalize

from models import Encoder, Decoder, MultiScaleProjectionHead


# ============================================================================
# Helper: denormalise images from [-1, 1] back to [0, 1] for visualisation
# ============================================================================

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Map images from [-1, 1] (Normalize(0.5, 0.5)) back to [0, 1]."""
    return tensor * 0.5 + 0.5


# ============================================================================
# 1. Compute patch-wise anomaly score maps
# ============================================================================

@torch.no_grad()
def compute_anomaly_maps(
    encoder: Encoder,
    decoder: Decoder,
    proj_head: MultiScaleProjectionHead,
    images: torch.Tensor,
    device: torch.device,
    image_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the multi-scale fused anomaly map M for a batch of images.

    Args:
        encoder, decoder, proj_head: Trained PatchCL-AE models.
        images:     Batch of clean images (B, 3, H, W) in [-1, 1].
        device:     Computation device.
        image_size: Target size for the fused anomaly map (default 256).

    Returns:
        anomaly_maps: (B, 1, image_size, image_size) — higher = more anomalous.
        x_hat:        (B, 3, image_size, image_size) — reconstructed images.
    """
    encoder.eval()
    decoder.eval()
    proj_head.eval()

    images = images.to(device)

    feats_orig = encoder(images)
    x_hat = decoder(feats_orig)
    feats_recon = encoder(x_hat)

    proj_orig = proj_head(feats_orig, sample_ids=None)
    proj_recon = proj_head(feats_recon, sample_ids=None)

    B = images.size(0)
    fused_map = torch.zeros(B, 1, image_size, image_size, device=device)

    for l in range(len(proj_orig)):
        p_o = F.normalize(proj_orig[l], dim=-1)
        p_r = F.normalize(proj_recon[l], dim=-1)
        cos_sim = (p_o * p_r).sum(dim=-1)
        anomaly = 1.0 - cos_sim
        H_l, W_l = feats_orig[l].shape[2], feats_orig[l].shape[3]
        anomaly_map = anomaly.view(B, 1, H_l, W_l)
        anomaly_map_up = F.interpolate(anomaly_map, size=(image_size, image_size),
                                       mode="bilinear", align_corners=False)
        fused_map = fused_map + anomaly_map_up

    return fused_map, x_hat


# ============================================================================
# 2. Image-level anomaly score (for ROC / AUC)
# ============================================================================

def image_level_score(anomaly_maps: torch.Tensor) -> np.ndarray:
    """
    Reduce per-pixel anomaly maps to a single scalar per image.
    Uses the mean of the top-k (k=100) anomaly values as a robust summary.
    """
    B = anomaly_maps.size(0)
    flat = anomaly_maps.view(B, -1)
    k = min(100, flat.size(1))
    topk_vals, _ = flat.topk(k, dim=1)
    scores = topk_vals.mean(dim=1)
    return scores.cpu().numpy()


# ============================================================================
# 3. Full evaluation loop
# ============================================================================

@torch.no_grad()
def evaluate(
    encoder: Encoder,
    decoder: Decoder,
    proj_head: MultiScaleProjectionHead,
    test_loader,
    device: torch.device,
    save_dir: str = "./results",
    num_vis: int = 8,
):
    """
    Run full evaluation on the test set and produce comprehensive outputs.
    """
    encoder.eval()
    decoder.eval()
    proj_head.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_scores = []
    all_labels = []
    vis_images, vis_recons, vis_maps, vis_labels = [], [], [], []

    for batch in test_loader:
        images, labels = batch
        anomaly_maps, x_hat = compute_anomaly_maps(
            encoder, decoder, proj_head, images, device
        )
        scores = image_level_score(anomaly_maps)
        all_scores.append(scores)
        all_labels.append(labels.numpy())

        if len(vis_images) < num_vis:
            need = num_vis - len(vis_images)
            vis_images.append(images[:need].cpu())
            vis_recons.append(x_hat[:need].cpu())
            vis_maps.append(anomaly_maps[:need].cpu())
            vis_labels.extend(labels[:need].tolist())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    auc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    # Optimal threshold via Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    preds = (all_scores >= best_thresh).astype(int)

    tp = int(np.sum((preds == 1) & (all_labels == 1)))
    tn = int(np.sum((preds == 0) & (all_labels == 0)))
    fp = int(np.sum((preds == 1) & (all_labels == 0)))
    fn = int(np.sum((preds == 0) & (all_labels == 1)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision_val = tp / max(tp + fp, 1)
    f1 = 2 * precision_val * sensitivity / max(precision_val + sensitivity, 1e-8)

    metrics = {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_val),
        "f1": float(f1),
        "threshold": float(best_thresh),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total_normal": int(np.sum(all_labels == 0)),
        "total_anomaly": int(np.sum(all_labels == 1)),
    }

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  AUC          = {auc:.4f}")
    print(f"  Threshold    = {best_thresh:.4f} (Youden's J)")
    print(f"  Accuracy     = {accuracy:.4f}")
    print(f"  Sensitivity  = {sensitivity:.4f}  (Recall / TPR)")
    print(f"  Specificity  = {specificity:.4f}")
    print(f"  Precision    = {precision_val:.4f}")
    print(f"  F1 Score     = {f1:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*50}")

    # ------------------------------------------------------------------
    # Save metrics as JSON + CSV
    # ------------------------------------------------------------------
    with open(os.path.join(save_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(save_dir, "metrics_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            w.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])

    # ------------------------------------------------------------------
    # Collect per-class examples (4 normal + 4 anomaly with scores)
    # ------------------------------------------------------------------
    vis_images_t = torch.cat(vis_images, dim=0)[:num_vis]
    vis_recons_t = torch.cat(vis_recons, dim=0)[:num_vis]
    vis_maps_t = torch.cat(vis_maps, dim=0)[:num_vis]
    vis_labels_list = vis_labels[:num_vis]

    # ------------------------------------------------------------------
    # FIGURE 1: Per-sample results (Original / Reconstruction / Heatmap)
    # ------------------------------------------------------------------
    _plot_sample_results(vis_images_t, vis_recons_t, vis_maps_t,
                         vis_labels_list, all_scores[:num_vis], save_dir)

    # ------------------------------------------------------------------
    # FIGURE 2: ROC Curve
    # ------------------------------------------------------------------
    _plot_roc_curve(fpr, tpr, auc, best_thresh, best_idx, save_dir)

    # ------------------------------------------------------------------
    # FIGURE 3: Score Distribution (Normal vs Anomaly)
    # ------------------------------------------------------------------
    _plot_score_distribution(all_scores, all_labels, best_thresh, save_dir)

    # ------------------------------------------------------------------
    # FIGURE 4: Confusion Matrix
    # ------------------------------------------------------------------
    _plot_confusion_matrix(all_labels, preds, save_dir)

    # ------------------------------------------------------------------
    # FIGURE 5: Metrics bar chart
    # ------------------------------------------------------------------
    _plot_metrics_bar(metrics, save_dir)

    # ------------------------------------------------------------------
    # FIGURE 6: Per-class example grid (Normal vs Anomaly side-by-side)
    # ------------------------------------------------------------------
    _plot_per_class_examples(all_scores, all_labels, vis_images_t,
                             vis_recons_t, vis_maps_t, vis_labels_list,
                             save_dir)

    print(f"\n[INFO] All evaluation figures and data saved to: {save_dir}/")

    return metrics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def _plot_sample_results(images, recons, maps, labels, scores, save_dir):
    """
    Figure 1: Per-sample 3-column figure.
      Col 1 — Original MRI (with class label + anomaly score)
      Col 2 — Reconstruction by auto-encoder
      Col 3 — Anomaly heatmap overlay (Jet colormap, red = high anomaly)
    """
    n = images.size(0)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("PatchCL-AE — Per-Sample Anomaly Detection Results",
                 fontsize=16, fontweight="bold", y=1.01)

    for i in range(n):
        img_np = denormalize(images[i]).permute(1, 2, 0).numpy().clip(0, 1)
        rec_np = denormalize(recons[i]).permute(1, 2, 0).numpy().clip(0, 1)
        map_np = maps[i, 0].numpy()

        label_str = "ANOMALY (Tumor)" if labels[i] == 1 else "NORMAL (No Tumor)"
        color = "#D32F2F" if labels[i] == 1 else "#388E3C"
        score_str = f"Score: {scores[i]:.4f}" if i < len(scores) else ""

        # Col 1: Original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"{label_str}\n{score_str}",
                             fontsize=11, color=color, fontweight="bold")
        axes[i, 0].axis("off")

        # Col 2: Reconstruction
        axes[i, 1].imshow(rec_np)
        axes[i, 1].set_title("Auto-Encoder Reconstruction", fontsize=11)
        axes[i, 1].axis("off")

        # Col 3: Heatmap overlay
        axes[i, 2].imshow(img_np)
        im = axes[i, 2].imshow(map_np, cmap="jet", alpha=0.5,
                                norm=MplNormalize(vmin=map_np.min(),
                                                  vmax=map_np.max()))
        axes[i, 2].set_title("Anomaly Heatmap", fontsize=11)
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(save_dir, "patchcl_ae_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Sample results figure saved -> {path}")


def _plot_roc_curve(fpr, tpr, auc, best_thresh, best_idx, save_dir):
    """
    Figure 2: ROC Curve with AUC, optimal operating point, and annotations.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color="#1976D2", lw=2.5,
            label=f"PatchCL-AE  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")

    # Mark optimal threshold point
    ax.scatter(fpr[best_idx], tpr[best_idx], s=120, c="#D32F2F", zorder=5,
               edgecolors="white", linewidths=2)
    ax.annotate(f"Optimal Threshold = {best_thresh:.4f}\n"
                f"TPR = {tpr[best_idx]:.3f}, FPR = {fpr[best_idx]:.3f}",
                xy=(fpr[best_idx], tpr[best_idx]),
                xytext=(fpr[best_idx] + 0.1, tpr[best_idx] - 0.15),
                fontsize=10, color="#D32F2F",
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5))

    # Fill AUC area
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1976D2")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — PatchCL-AE Anomaly Detection", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] ROC curve saved -> {path}")


def _plot_score_distribution(scores, labels, threshold, save_dir):
    """
    Figure 3: Histogram of anomaly scores for Normal vs Anomaly,
    with threshold line and overlap visualisation.
    """
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(scores.min(), scores.max(), 50)
    ax.hist(normal_scores, bins=bins, alpha=0.6, color="#4CAF50",
            label=f"Normal (n={len(normal_scores)})", edgecolor="white")
    ax.hist(anomaly_scores, bins=bins, alpha=0.6, color="#F44336",
            label=f"Anomaly (n={len(anomaly_scores)})", edgecolor="white")

    # Threshold line
    ax.axvline(threshold, color="#FF9800", lw=2.5, linestyle="--",
               label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Anomaly Score", fontsize=13)
    ax.set_ylabel("Number of Images", fontsize=13)
    ax.set_title("Anomaly Score Distribution — Normal vs Tumor",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add text box with separation statistics
    mean_n, std_n = normal_scores.mean(), normal_scores.std()
    mean_a, std_a = anomaly_scores.mean(), anomaly_scores.std()
    textstr = (f"Normal:  μ={mean_n:.4f}, σ={std_n:.4f}\n"
               f"Anomaly: μ={mean_a:.4f}, σ={std_a:.4f}")
    props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props)

    fig.tight_layout()
    path = os.path.join(save_dir, "score_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Score distribution saved -> {path}")


def _plot_confusion_matrix(labels, preds, save_dir):
    """
    Figure 4: Confusion matrix with counts and percentages.
    """
    cm = sk_confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Annotate cells with count + percentage
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_pct[i, j]
            text_color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    color=text_color, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"], fontsize=12)
    ax.set_yticklabels(["Normal", "Anomaly"], fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix — PatchCL-AE", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved -> {path}")


def _plot_metrics_bar(metrics, save_dir):
    """
    Figure 5: Bar chart summarising key classification metrics.
    """
    metric_names = ["AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    metric_vals = [metrics["auc"], metrics["accuracy"], metrics["sensitivity"],
                   metrics["specificity"], metrics["precision"], metrics["f1"]]
    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C2185B", "#00796B"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(metric_names, metric_vals, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.6)

    # Value labels on top of bars
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("PatchCL-AE — Evaluation Metrics Summary",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(save_dir, "metrics_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Metrics bar chart saved -> {path}")


def _plot_per_class_examples(all_scores, all_labels, images, recons, maps,
                             vis_labels, save_dir):
    """
    Figure 6: Side-by-side comparison grid.
    Top row: Normal examples (low anomaly score — model reconstructs well).
    Bottom row: Anomaly examples (high anomaly score — reconstruction fails).
    Each cell shows Original + Heatmap overlay + score.
    """
    # Separate indices from visualised samples
    normal_idx = [i for i, l in enumerate(vis_labels) if l == 0]
    anomaly_idx = [i for i, l in enumerate(vis_labels) if l == 1]

    n_show = min(4, max(len(normal_idx), len(anomaly_idx)))
    if n_show == 0:
        return

    normal_idx = normal_idx[:n_show]
    anomaly_idx = anomaly_idx[:n_show]
    max_cols = max(len(normal_idx), len(anomaly_idx))

    if max_cols == 0:
        return

    fig, axes = plt.subplots(2, max_cols * 2, figsize=(5 * max_cols, 10))
    if max_cols == 1:
        axes = axes.reshape(2, 2)

    fig.suptitle("PatchCL-AE — Normal vs Anomaly Comparison\n"
                 "Left: Original MRI  |  Right: Anomaly Heatmap",
                 fontsize=14, fontweight="bold", y=1.02)

    def _draw_pair(row, col_offset, idx, label_text, score_val):
        img_np = denormalize(images[idx]).permute(1, 2, 0).numpy().clip(0, 1)
        map_np = maps[idx, 0].numpy()

        # Original
        axes[row, col_offset].imshow(img_np)
        axes[row, col_offset].set_title(f"{label_text}\nScore: {score_val:.4f}",
                                        fontsize=10, fontweight="bold")
        axes[row, col_offset].axis("off")

        # Heatmap
        axes[row, col_offset + 1].imshow(img_np)
        axes[row, col_offset + 1].imshow(map_np, cmap="jet", alpha=0.5,
                                          norm=MplNormalize(vmin=map_np.min(),
                                                            vmax=map_np.max()))
        axes[row, col_offset + 1].set_title("Heatmap", fontsize=10)
        axes[row, col_offset + 1].axis("off")

    # Row 0: Normal examples
    for c, idx in enumerate(normal_idx):
        s = all_scores[idx] if idx < len(all_scores) else 0.0
        _draw_pair(0, c * 2, idx, "NORMAL", s)
    # Hide unused
    for c in range(len(normal_idx), max_cols):
        axes[0, c * 2].axis("off")
        axes[0, c * 2 + 1].axis("off")

    # Row 1: Anomaly examples
    for c, idx in enumerate(anomaly_idx):
        s = all_scores[idx] if idx < len(all_scores) else 0.0
        _draw_pair(1, c * 2, idx, "ANOMALY", s)
    for c in range(len(anomaly_idx), max_cols):
        axes[1, c * 2].axis("off")
        axes[1, c * 2 + 1].axis("off")

    # Row labels
    fig.text(0.01, 0.75, "Normal\n(No Tumor)", fontsize=12, fontweight="bold",
             color="#388E3C", va="center", rotation=90)
    fig.text(0.01, 0.25, "Anomaly\n(Tumor)", fontsize=12, fontweight="bold",
             color="#D32F2F", va="center", rotation=90)

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    path = os.path.join(save_dir, "per_class_examples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Per-class examples saved -> {path}")
