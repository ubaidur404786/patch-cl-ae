"""
losses.py — Loss functions for PatchCL-AE
==========================================

Two core losses drive training:

1. **Global Adversarial Image Loss (L_Img)**
   An adversarial loss (LSGAN formulation) ensures that the reconstructed
   image X_hat matches the global visual style of the raw input X.  The
   Discriminator tries to tell real from reconstructed, while the
   Encoder+Decoder (generator) fools it.

2. **Patch-wise Contrastive Loss (L_Patch)**  — Eq. 3 from the paper
   This is the key innovation.  Patches are treated like "fingerprints":
   for a given spatial location, the reconstructed patch embedding (Query)
   is pulled closer to the original patch embedding at the *same* location
   (Positive) and pushed away from embeddings at *all other* locations in
   the same image (Negatives).

   Contrastive learning enables the model to understand local semantics.
   Unlike pixel-level reconstruction losses, it is invariant to small
   pixel shifts and noise, while being highly sensitive to true semantic
   discrepancies (e.g., lesions).  This makes the patch-wise anomaly
   score far more reliable for detecting pathological regions.

   Mathematical formula (Eq. 3):

     L_Patch = Σ_l Σ_i  -log [ exp(sim(z, z⁺) / τ) /
                                (exp(sim(z, z⁺) / τ) + Σ_{z⁻} exp(sim(z, z⁻) / τ)) ]

   where sim(·,·) is cosine similarity and τ is a temperature hyper-parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Adversarial Image Loss (L_Img) — LSGAN formulation
# ============================================================================

class AdversarialLoss(nn.Module):
    """
    Least-Squares GAN loss (LSGAN) for training stability — Section 3.2.

    For the discriminator:
        L_D = 0.5 * E[(D(real) - 1)²] + 0.5 * E[(D(fake))²]

    For the generator (encoder + decoder):
        L_G = 0.5 * E[(D(fake) - 1)²]

    The discriminator now outputs a scalar per image (after global average
    pooling), so this reduces to a standard MSE between the scalar
    prediction and the target label (1 for real, 0 for fake).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return F.mse_loss(pred, target)


# ============================================================================
# 2. Patch-wise Contrastive Loss (L_Patch) — Eq. 3
# ============================================================================

class PatchContrastiveLoss(nn.Module):
    """
    Patch-wise contrastive loss (InfoNCE / NT-Xent style) operating in the
    projected embedding space.

    For each spatial location *i* in each encoder layer *l*:
      - Query  (z):   projected embedding of the *reconstructed* patch at (l, i)
      - Positive (z⁺): projected embedding of the *original* patch at (l, i)
      - Negatives (z⁻): projected embeddings of the original patches at all
                         *other* sampled locations in the same layer

    The loss encourages the encoder to produce locally consistent representations:
    a reconstructed patch should be semantically identical to its original
    counterpart.  During inference, locations where this agreement breaks down
    signal anomalous (pathological) content — because the auto-encoder, trained
    only on normal data, cannot faithfully reconstruct abnormal structures.

    Args:
        temperature (float): Softmax temperature τ (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                proj_query: list[torch.Tensor],
                proj_positive: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute L_Patch summed over all encoder layers.

        Args:
            proj_query:    List of projected reconstructed features per layer,
                           each of shape (B, S_l, D) with D = 256.
            proj_positive: List of projected original features per layer,
                           each of shape (B, S_l, D).

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=proj_query[0].device)

        for l in range(len(proj_query)):
            q = proj_query[l]    # (B, S, D) — reconstructed patch embeddings
            p = proj_positive[l] # (B, S, D) — original patch embeddings

            # L2-normalise along the feature dimension for cosine similarity
            q = F.normalize(q, dim=-1)
            p = F.normalize(p, dim=-1)

            B, S, D = q.shape

            # Cosine similarity between query and ALL positive positions
            # sim_matrix[b, i, j] = cos(q_i, p_j)  — shape (B, S, S)
            sim_matrix = torch.bmm(q, p.transpose(1, 2)) / self.temperature

            # The positive for query i is the positive at position i (diagonal)
            # We use cross-entropy where the target class for each query is its
            # own index (i.e., the diagonal element).
            # Reshape to (B*S, S) and targets to (B*S,)
            sim_matrix = sim_matrix.reshape(B * S, S)
            targets = torch.arange(S, device=q.device).repeat(B)

            loss_l = F.cross_entropy(sim_matrix, targets)
            total_loss = total_loss + loss_l

        return total_loss
