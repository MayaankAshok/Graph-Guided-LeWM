"""Graph-Assisted Stitching (GAS; Baek et al., ICML 2025, arXiv:2506.07744) on top of the
frozen LeWM latent.

The paper's pipeline, implemented here component by component (section numbers are the
paper's; the official JAX code at github.com/qortmdgh4141/GAS was read for the details the
paper leaves out -- each such detail is flagged "official code" below):

  1. Temporal Distance Representation (Sec. 3.3, Eq. 3-6): psi: S -> R^32 with
     V(s,g) = -||psi(s) - psi(g)||, trained by expectile TD (HILP-style). Here S is the
     frozen 192-d LeWM CLS latent `z` (bak/docs/gas-lewm-proposal.md: "image -> frozen LeWM
     ViT -> z -> MLP -> psi(s)"), so the TDR is an MLP over cached vectors.
  2. Temporal Efficiency filter (Sec. 4.2, Eq. 7-9) + TD-aware clustering + edges
     (Sec. 4.1, App. A Alg. 2).
  3. Graph construction: TD-aware clustering and graph edges used by CEM objectives.

Nothing here is Hydra- or environment-specific; `gas_mpc_prepare.py` builds the TDR and
graph assets consumed by the CEM planner.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-10  # official code's normalisation epsilon


# ------------------------------------------------------------------
# networks
# ------------------------------------------------------------------

class LNMLP(nn.Module):
    """(hidden,)*n_layers with LayerNorm + GELU per layer (paper Tab. 7), linear head."""

    def __init__(self, in_dim, out_dim, hidden=512, n_layers=3, layer_norm=True):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden) if layer_norm else nn.Identity(), nn.GELU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        return self.head(self.body(x))


class TDR(nn.Module):
    """Two-member psi ensemble (HILP/GAS 'tdr_value' network): V_k(s,g) = -||psi_k(s)-psi_k(g)||.
    Planning and the low-level agent read psi_0 (`phi`); the ensemble only serves the
    min-over-targets in the TDR loss, as in the official code."""

    def __init__(self, in_dim, tdr_dim=32, hidden=512, n_layers=3):
        super().__init__()
        self.nets = nn.ModuleList([LNMLP(in_dim, tdr_dim, hidden, n_layers) for _ in range(2)])
        self.tdr_dim = tdr_dim

    def phi(self, z, k=0):
        return self.nets[k](z)

    def values(self, zs, zg):
        out = []
        for net in self.nets:
            diff = net(zs) - net(zg)
            out.append(-torch.sqrt((diff ** 2).sum(-1).clamp_min(1e-6)))
        return out  # [v1, v2], each (B,)


def ema_update(target, source, tau):
    with torch.no_grad():
        for pt, ps in zip(target.parameters(), source.parameters()):
            pt.mul_(1 - tau).add_(ps, alpha=tau)


def expectile_loss(adv, diff, tau):
    weight = torch.where(adv < 0, 1 - tau, tau)
    return (weight * diff ** 2).mean()


def unit(x, dim=-1):
    if torch.is_tensor(x):
        return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + EPS)
    return x / (np.linalg.norm(x, axis=dim, keepdims=True) + EPS)


# ------------------------------------------------------------------
# episode layout helpers
# ------------------------------------------------------------------

def episode_rows(ep_idx, step_idx):
    """List of absolute row-index arrays, one per episode, in step order."""
    order = np.lexsort((step_idx, ep_idx))
    ep_o = ep_idx[order]
    boundaries = np.nonzero(np.diff(ep_o))[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    return [order[s:e] for s, e in zip(starts, ends)]


def episode_end_row(episodes, n):
    """For each row, the LAST row of its own episode (0..n-1 indexing kept absolute)."""
    end = np.full(n, -1, dtype=np.int64)
    for rows in episodes:
        end[rows] = rows[-1]
    return end


# ------------------------------------------------------------------
# Stage 1: TDR training batches
# ------------------------------------------------------------------

class TDRSampler:
    """HILP/GAS goal relabelling (paper App. D): with p_trajgoal the goal is a future state
    of the same trajectory at a Geom(1-gamma) offset (clipped to the episode end), with
    p_randgoal a uniform dataset state; s == g is never sampled (p_curgoal = 0), so the
    reward is always -1 and the mask always 1 -- V(s', g) hits 0 by parameterisation when
    s' == g. Restricted to training episodes."""

    def __init__(self, train_episodes, n_rows, gamma, p_trajgoal=0.625, seed=0):
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.p_trajgoal = p_trajgoal
        self.ep_end = episode_end_row(train_episodes, n_rows)
        rows = np.concatenate(train_episodes)
        self.train_rows = rows
        self.rows_with_next = rows[self.ep_end[rows] != rows]  # a real next step exists
        # contiguity check: rows within an episode must be consecutive integers for i+1 to
        # be the next step (true for load_landmarks' per-episode concatenation)
        for e in train_episodes:
            assert np.all(np.diff(e) == 1), "episode rows must be contiguous for next = row + 1"

    def sample(self, batch_size):
        s = self.rng.choice(self.rows_with_next, size=batch_size)
        nx = s + 1
        offset = self.rng.geometric(1 - self.gamma, size=batch_size)  # >= 1
        g_traj = np.minimum(s + offset, self.ep_end[s])
        g_rand = self.rng.choice(self.train_rows, size=batch_size)
        use_traj = self.rng.random(batch_size) < self.p_trajgoal
        g = np.where(use_traj, g_traj, g_rand)
        g = np.where(g == s, nx, g)  # never s == g
        return s, nx, g


def tdr_loss(tdr, tdr_target, zs, zn, zg, gamma, tau):
    """Eq. 5-6 with the official code's two-member target min (HILP value_loss)."""
    with torch.no_grad():
        nv1, nv2 = tdr_target.values(zn, zg)
        next_v = torch.minimum(nv1, nv2)
        q = -1.0 + gamma * next_v                # reward -1, mask 1 (see TDRSampler)
        tv1, tv2 = tdr_target.values(zs, zg)
        adv = q - 0.5 * (tv1 + tv2)
    v1, v2 = tdr.values(zs, zg)
    return expectile_loss(adv, q - v1, tau) + expectile_loss(adv, q - v2, tau), float(v1.mean().detach())


# ------------------------------------------------------------------
# Stage 2: TE filter, TD-aware clustering, edges (App. A, Alg. 2)
# ------------------------------------------------------------------

def waypoint_by_distance(H, episodes, h_td):
    """Eq. 7: F(s_t, H_TD) = first later state of the same trajectory whose TDR distance from
    s_t is >= H_TD; the episode's last state if none (official code's default). Returns an
    absolute-row array (defaults to the row itself for rows outside `episodes`)."""
    way = np.arange(len(H), dtype=np.int64)
    for rows in episodes:
        Hr = H[rows]
        L = len(rows)
        D = np.linalg.norm(Hr[:, None, :] - Hr[None, :, :], axis=-1)
        later = np.triu(np.ones((L, L), dtype=bool), k=1)
        hit = (D >= h_td) & later
        has = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        way[rows] = np.where(has, rows[first], rows[-1])
    return way


def temporal_efficiency(H, episodes, way, h_td):
    """Eq. 8-9: cos(psi(s_opt) - psi(s_cur), psi(s_reached) - psi(s_cur)) with s_opt = F(s, H_TD)
    and s_reached = s_{t+H_TD} (episode end if the trajectory is shorter -- official code).
    Returns a float array over all rows (NaN outside `episodes`)."""
    te = np.full(len(H), np.nan, dtype=np.float32)
    for rows in episodes:
        L = len(rows)
        t = np.arange(L)
        reached = rows[np.minimum(t + h_td, L - 1)]
        opt = way[rows]
        v_opt = H[opt] - H[rows]
        v_reached = H[reached] - H[rows]
        te[rows] = (unit(v_opt) * unit(v_reached)).sum(-1)
    return te


def build_node_graph(centers, h_td):
    """Edges between every pair of nodes with ||v_i - v_j|| <= H_TD, weight = the distance
    (official code; the paper's Alg. 2 only states the threshold). Symmetric csr."""
    n = len(centers)
    rows, cols, vals = [], [], []
    chunk = max(1, int(2e8 // max(1, n * centers.shape[1] * 8)))
    for a in range(0, n, chunk):
        D = np.linalg.norm(centers[a:a + chunk, None, :] - centers[None, :, :], axis=-1)
        ii, jj = np.nonzero(D <= h_td)
        keep = (ii + a) != jj
        rows.append(ii[keep] + a); cols.append(jj[keep]); vals.append(D[ii[keep], jj[keep]])
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    return csr_matrix((vals.astype(np.float64), (rows, cols)), shape=(n, n))


def _n_components(graph):
    from scipy.sparse.csgraph import connected_components
    return connected_components(graph, directed=False)[0]
