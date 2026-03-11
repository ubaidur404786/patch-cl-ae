"""
main.py — Entry point for PatchCL-AE: Medical Anomaly Detection
=================================================================

PatchCL-AE (Patch-wise Contrastive Learning Auto-Encoder) detects anomalies in
medical images by training a denoising auto-encoder solely on normal images,
then measuring patch-level semantic discrepancies between an input and its
reconstruction.

The key innovation is the use of Contrastive Learning at the patch level:
instead of relying on pixel-wise reconstruction error (which is noisy and
sensitive to small geometric shifts), PatchCL-AE learns rich local semantic
embeddings.  At inference time, a reconstructed patch that deviates from its
original counterpart in this embedding space signals an anomaly — the
auto-encoder, never having seen pathological structures, cannot faithfully
reproduce them.

Usage:
    python main.py                          # train + evaluate with defaults
    python main.py --epochs 100             # more training
    python main.py --evaluate-only --ckpt checkpoints/patchcl_ae_epoch50.pt
"""

import argparse
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="PatchCL-AE: Patch-wise Contrastive Learning AE for "
                    "Medical Anomaly Detection (Brain Tumor MRI)")

    # Data
    p.add_argument("--data-root", type=str, default="./data",
                   help="Root directory for dataset storage.")
    p.add_argument("--image-size", type=int, default=256,
                   help="Resize images to this resolution (default: 256).")
    p.add_argument("--noise-std", type=float, default=0.05,
                   help="Std-dev of Gaussian noise for denoising AE (default: 0.05).")

    # Training
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs (default: 50).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size (default: 4).")
    p.add_argument("--lr", type=float, default=0.002,
                   help="Learning rate (default: 0.002).")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="Adam β₁ (default: 0.9).")
    p.add_argument("--beta2", type=float, default=0.999,
                   help="Adam β₂ (default: 0.999).")
    p.add_argument("--eps", type=float, default=1e-4,
                   help="Adam ε (default: 1e-4).")
    p.add_argument("--lambda-adv", type=float, default=1.0,
                   help="Weight for adversarial loss (default: 1.0).")
    p.add_argument("--num-patch-samples", type=int, default=256,
                   help="Number of random spatial locations sampled per layer "
                        "for contrastive learning (default: 256).")

    # Checkpointing
    p.add_argument("--save-dir", type=str, default="./checkpoints",
                   help="Directory for model checkpoints.")
    p.add_argument("--results-dir", type=str, default="./results",
                   help="Directory for evaluation results & figures.")

    # Evaluation-only mode
    p.add_argument("--evaluate-only", action="store_true",
                   help="Skip training; load checkpoint and evaluate.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to a checkpoint file for evaluation-only mode.")

    # Device
    p.add_argument("--device", type=str, default=None,
                   help="Device to use (default: auto-detect cuda/cpu).")

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print(f"{'='*60}")
    print(f"  PatchCL-AE: Patch-wise Contrastive Learning Auto-Encoder")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    if args.evaluate_only:
        # ---------------------------------------------------------------
        # Evaluation-only mode
        # ---------------------------------------------------------------
        if args.ckpt is None:
            print("[ERROR] --ckpt is required in --evaluate-only mode.")
            sys.exit(1)

        from models import Encoder, Decoder, MultiScaleProjectionHead
        from dataset import get_dataloaders
        from evaluate import evaluate

        _, test_loader = get_dataloaders(
            data_root=args.data_root, batch_size=args.batch_size,
            noise_std=0.0, image_size=args.image_size,
        )

        encoder = Encoder(in_channels=3).to(device)
        decoder = Decoder(out_channels=3).to(device)
        proj_head = MultiScaleProjectionHead().to(device)

        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        proj_head.load_state_dict(ckpt["proj_head"])
        print(f"[INFO] Loaded checkpoint from {args.ckpt} (epoch {ckpt['epoch']})")

        results = evaluate(encoder, decoder, proj_head, test_loader,
                           device=device, save_dir=args.results_dir)
        print(f"\nDone. AUC = {results['auc']:.4f}")

    else:
        # ---------------------------------------------------------------
        # Training + Evaluation
        # ---------------------------------------------------------------
        from train import train
        from evaluate import evaluate

        encoder, decoder, proj_head, test_loader, device = train(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            lambda_adv=args.lambda_adv,
            noise_std=args.noise_std,
            num_patch_samples=args.num_patch_samples,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
            device=args.device,
        )

        print(f"\n{'='*60}")
        print(f"  Evaluation")
        print(f"{'='*60}")

        results = evaluate(encoder, decoder, proj_head, test_loader,
                       device=device, save_dir=args.results_dir)
        print(f"\nDone. AUC = {results['auc']:.4f}")


if __name__ == "__main__":
    main()
