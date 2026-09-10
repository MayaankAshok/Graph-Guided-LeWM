"""
Quasimetric Head for Latent World Models (LeWM).
Parametrizes an asymmetric dynamical reachability function d_Q(z_i, z_j)
satisfying:
1. Positivity: d_Q(z_i, z_j) >= 0 and d_Q(z_i, z_i) = 0
2. Directed Triangle Inequality: d_Q(z_i, z_k) <= d_Q(z_i, z_j) + d_Q(z_j, z_k)
3. Directed Asymmetry: d_Q(z_i, z_j) != d_Q(z_j, z_i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentQuasimetric(nn.Module):
    def __init__(self, latent_dim: int = 192, hidden_dim: int = 256, proj_dim: int = 64):
        super().__init__()
        # Asymmetric potential function (captures directional progress/time arrow)
        self.phi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        # Metric projection (captures local state differences)
        self.psi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """
        Computes directed distance d_Q(z_src -> z_dst).
        z_src: (..., latent_dim)
        z_dst: (..., latent_dim)
        Returns: (...,) non-negative distance
        """
        # Directional potential difference (strictly asymmetric)
        phi_src = self.phi(z_src)
        phi_dst = self.phi(z_dst)
        potential_diff = F.relu(phi_dst - phi_src).squeeze(-1)

        # Local pseudo-metric (symmetric component)
        psi_src = self.psi(z_src)
        psi_dst = self.psi(z_dst)
        spatial_dist = torch.norm(psi_src - psi_dst, p=2, dim=-1)

        return potential_diff + spatial_dist


def compute_quasimetric_loss(
    model: LatentQuasimetric,
    z_t: torch.Tensor,
    z_tpk: torch.Tensor,
    steps_k: torch.Tensor,
    z_rand: torch.Tensor,
    margin: float = 50.0,
) -> dict:
    """
    Self-supervised training loss for the quasimetric head.
    z_t: start states (B, D)
    z_tpk: successor states reached k steps later (B, D)
    steps_k: ground truth transition step count k (B,)
    z_rand: random negative / uncoupled states (B, D)
    """
    # 1. Positive forward pairs: d_Q(z_t -> z_t+k) should match step count k
    d_fwd = model(z_t, z_tpk)
    loss_fwd = F.smooth_l1_loss(d_fwd, steps_k)

    # 2. Reverse penalty: d_Q(z_t+k -> z_t) for irreversible processes
    d_bwd = model(z_tpk, z_t)
    loss_bwd = F.relu(steps_k + 10.0 - d_bwd).mean()

    # 3. Unreachable / random contrastive margin
    d_rand = model(z_t, z_rand)
    loss_rand = F.relu(margin - d_rand).mean()

    total_loss = loss_fwd + 0.5 * loss_bwd + 0.2 * loss_rand
    return {
        "loss": total_loss,
        "loss_fwd": loss_fwd.item(),
        "loss_bwd": loss_bwd.item(),
        "loss_rand": loss_rand.item(),
        "d_fwd_mean": d_fwd.mean().item(),
        "d_bwd_mean": d_bwd.mean().item(),
    }
