"""
PatchCL-AE: Patch-wise Contrastive Learning Auto-Encoder for Medical Anomaly Detection
========================================================================================

This package implements the PatchCL-AE framework as described in:
"Anomaly detection for medical images using patch-wise contrastive learning-based auto-encoder"

The key insight of PatchCL-AE is that Contrastive Learning enables the model to understand
local semantics at the patch level, allowing it to ignore pixel-level shifts and noise while
pinpointing true semantic discrepancies (e.g., lesions in medical images).

Modules:
    - dataset:  Data downloading, loading, and preprocessing (Kermany Retinal OCT)
    - models:   Encoder, Decoder, Discriminator, and Projection Head architectures
    - losses:   Adversarial (global image) loss and Patch-wise Contrastive loss
    - train:    Full training pipeline (Algorithm 1 from the paper)
    - evaluate: Anomaly scoring, multi-scale fusion, and visualization
"""
