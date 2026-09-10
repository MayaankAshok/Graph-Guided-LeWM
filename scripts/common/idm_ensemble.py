"""Small inverse-dynamics-model (IDM) ensemble: (z1, z2) -> action, trained on real
adjacent (z_t, z_{t+1}, a_t) triples -- environment-agnostic, mirrors the MLP/training
conventions already in common.training.

Stage E1 of docs/experiment-plan-predictor-stitching.md only asks whether such an ensemble
can recover a KNOWN action from a pair of latents at all, on held-out same-episode
transitions. Ensemble disagreement as a confidence signal for genuinely out-of-distribution
cross-episode pairs is stage E2, not exercised here.
"""

import numpy as np
import torch

from common.training import MLP

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train_idm_ensemble(z, action, i_idx, j_idx, action_dim, n_members=6, hidden=256,
                        n_steps=3000, batch_size=256, lr=1e-3, seed=0):
    """Train `n_members` independent IDM heads on (z[i_idx], z[j_idx]) -> action[i_idx],
    each on its own bootstrap resample of the given pairs (real adjacent same-episode
    transitions) and its own weight initialization, so ensemble disagreement reflects both
    data and init variance. Returns a list of trained, eval-mode MLPs."""
    n = len(i_idx)
    zi = torch.from_numpy(z[i_idx]).float().to(DEV)
    zj = torch.from_numpy(z[j_idx]).float().to(DEV)
    a = torch.from_numpy(action[i_idx]).float().to(DEV)
    x = torch.cat([zi, zj], dim=-1)

    models = []
    for m in range(n_members):
        rng = np.random.default_rng(seed + m)
        boot = rng.integers(0, n, size=n)
        x_m, a_m = x[boot], a[boot]

        torch.manual_seed(seed + m)
        model = MLP(in_dim=x.shape[1], hidden=hidden, out_dim=action_dim).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_steps):
            idx = torch.randint(0, n, (batch_size,), device=DEV)
            pred = model(x_m[idx])
            loss = ((pred - a_m[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        models.append(model.eval())
    return models


def ensemble_predict(models, z, i_idx, j_idx):
    """Returns stacked predictions, shape (n_members, n_pairs, action_dim)."""
    zi = torch.from_numpy(z[i_idx]).float().to(DEV)
    zj = torch.from_numpy(z[j_idx]).float().to(DEV)
    x = torch.cat([zi, zj], dim=-1)
    with torch.no_grad():
        preds = torch.stack([m(x) for m in models], dim=0)
    return preds.cpu().numpy()


def _classifier_features(z, i_idx, j_idx):
    zi = torch.from_numpy(z[i_idx]).float().to(DEV)
    zj = torch.from_numpy(z[j_idx]).float().to(DEV)
    return torch.cat([zi, zj, zj - zi], dim=-1)


def train_transition_classifier(z, pos_i, pos_j, neg_i, neg_j, n_members=5, hidden=256,
                                 n_steps=3000, batch_size=256, lr=1e-3, seed=0):
    """Binary classifier ensemble: is (z_i, z_j) a real one-step transition, or not? Unlike
    the action-regression IDM ensemble above (which always outputs SOME action regardless of
    whether the pair is connectable, and whose inter-member variance was found empirically
    NOT to predict actual prediction error on Push-T candidates -- correlation ~0), this asks
    the more direct question a discriminative model is built for. Train with
    distance-matched negatives (build these in the caller, e.g. via nearest-distance lookup
    against a pool of real cross-episode pairs) so the classifier can't just learn "small
    distance = real" -- it has to find a real distinguishing feature. Features are
    [z_i, z_j, z_j - z_i] (both endpoints plus raw displacement)."""
    x_pos = _classifier_features(z, pos_i, pos_j)
    x_neg = _classifier_features(z, neg_i, neg_j)
    x = torch.cat([x_pos, x_neg], dim=0)
    y = torch.cat([torch.ones(len(pos_i), 1), torch.zeros(len(neg_i), 1)], dim=0).to(DEV)
    n = x.shape[0]

    models = []
    for m in range(n_members):
        rng = np.random.default_rng(seed + m)
        boot = rng.integers(0, n, size=n)
        x_m, y_m = x[boot], y[boot]

        torch.manual_seed(seed + m)
        model = MLP(in_dim=x.shape[1], hidden=hidden, out_dim=1).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(n_steps):
            idx = torch.randint(0, n, (batch_size,), device=DEV)
            logit = model(x_m[idx])
            loss = loss_fn(logit, y_m[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        models.append(model.eval())
    return models


def classifier_predict_proba(models, z, i_idx, j_idx):
    """Mean predicted probability of "real transition" across the ensemble, shape (n_pairs,)."""
    x = _classifier_features(z, i_idx, j_idx)
    with torch.no_grad():
        probs = torch.stack([torch.sigmoid(m(x)) for m in models], dim=0)
    return probs.mean(dim=0).squeeze(-1).cpu().numpy()
