"""Unified noisy-rollout collector for the "mixed"/"mixed_large" data tiers -- config-driven
consolidation of tworoom_rollout_collector.py/pusht_rollout_collector.py. The actual
env-stepping mechanics (ExpertPolicy vs WeakPolicy, vectorized or not) live in
common/envs.py's EnvMechanics.collect_noisy_rollout; this module handles what's genuinely
shared: pixel encoding via the frozen LeWM checkpoint, and caching.

Not a Hydra entry point itself -- imported by datatiers.py (which builds the mixed/
mixed_large tiers) rather than run standalone, same role tworoom_rollout_collector.py/
pusht_rollout_collector.py already had.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.envs import ENV_MECHANICS
from common.graph_lib import DEV
from common.lewm_loader import load_lewm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def encode_pixels(pixels, ckpt_dir, batch_size=128):
    """Encode a (N,224,224,3) uint8 array with the frozen pretrained LeWM encoder."""
    model = load_lewm(ckpt_dir=ckpt_dir, device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    chunks = []
    for i in range(0, len(pixels), batch_size):
        chunk = pixels[i:i + batch_size]
        t = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        t = (t - mean) / std
        with torch.no_grad():
            out = model.encode({"pixels": t.unsqueeze(1)})
        chunks.append(out["emb"][:, 0].cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def collect_rollouts(env_name, n_episodes, seed, cache_dir, cache_name=None, **policy_kwargs):
    """Roll out env_name's noisy/weak collection policy for n_episodes. Returns
    dict(pixels, state, action, ep_idx, step_idx)."""
    if cache_name is not None:
        cache_path = Path(cache_dir) / f"{cache_name}.npz"
        if cache_path.exists():
            from common.log_util import log
            log(f"[rollout] loading cached '{cache_name}'")
            d = np.load(cache_path)
            return {k: d[k] for k in d.files}

    mech = ENV_MECHANICS[env_name]
    result = mech.collect_noisy_rollout(n_episodes, seed, **policy_kwargs)

    if cache_name is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(cache_dir) / f"{cache_name}.npz", **result)
        from common.log_util import log
        log(f"[rollout] cached to {cache_name}.npz")
    return result
