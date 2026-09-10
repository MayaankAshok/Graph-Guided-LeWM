"""Load a pretrained LeWM checkpoint (quentinll/lewm-<env> on HuggingFace). Environment-
agnostic -- already took ckpt_dir as a parameter before this move, used unchanged for both
Two-Room and Push-T. Relocated from tworoom_lewm_loader.py (kept as a re-export shim there)
since the "tworoom" prefix was always a misnomer once Push-T started reusing it.

The installed stable_pretraining (0.1.8) / transformers (5.15.1) ViT
implementation renamed the internal block attributes relative to whatever
version produced these HF-hosted checkpoints:

    old (checkpoint): encoder.encoder.layer.{i}.attention.attention.{query,key,value}
                       encoder.encoder.layer.{i}.attention.output.dense
                       encoder.encoder.layer.{i}.intermediate.dense
                       encoder.encoder.layer.{i}.output.dense
    new (installed):   encoder.layers.{i}.attention.{q_proj,k_proj,v_proj,o_proj}
                       encoder.layers.{i}.mlp.{fc1,fc2}

Verified (2026-08-28) that this is a pure renaming: remapping every
`encoder.encoder.layer.*` key leaves zero unmapped/missing/extra keys and
zero shape mismatches against the freshly instantiated model. Embeddings,
cls_token, position embeddings and the final LayerNorm already match under
both versions (only the per-block attention/mlp names moved).
"""

import json
import os
import re
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# LEWM_CKPT_DIR overrides the checkpoint location -- unset by default, so local runs are
# unchanged. Callers normally pass ckpt_dir explicitly (per environment); this default only
# matters for callers that don't.
CKPT_DIR = Path(os.environ.get("LEWM_CKPT_DIR",
                                str(REPO_ROOT / "data" / "checkpoints" / "models--quentinll--lewm-tworooms")))


def _remap_key(k: str) -> str | None:
    m = re.match(r"^encoder\.encoder\.layer\.(\d+)\.(.*)$", k)
    if not m:
        return k
    i, rest = m.group(1), m.group(2)
    if rest.startswith("attention.attention.query"):
        rest2 = rest.replace("attention.attention.query", "attention.q_proj")
    elif rest.startswith("attention.attention.key"):
        rest2 = rest.replace("attention.attention.key", "attention.k_proj")
    elif rest.startswith("attention.attention.value"):
        rest2 = rest.replace("attention.attention.value", "attention.v_proj")
    elif rest.startswith("attention.output.dense"):
        rest2 = rest.replace("attention.output.dense", "attention.o_proj")
    elif rest.startswith("intermediate.dense"):
        rest2 = rest.replace("intermediate.dense", "mlp.fc1")
    elif rest.startswith("output.dense"):
        rest2 = rest.replace("output.dense", "mlp.fc2")
    elif rest.startswith("layernorm_before") or rest.startswith("layernorm_after"):
        rest2 = rest
    else:
        return None
    return f"encoder.layers.{i}.{rest2}"


def load_tworoom_lewm(ckpt_dir: Path = CKPT_DIR, device: str = "cpu", strict: bool = True):
    """Return the pretrained LeWM model (JEPA: encoder/predictor/action_encoder/...), eval
    mode. Name kept for backward compatibility with ~20 existing call sites; `load_lewm`
    below is a neutral alias for new code."""
    sd = torch.load(ckpt_dir / "weights.pt", map_location="cpu")
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    model = instantiate(cfg)

    remapped = {}
    unmapped = []
    for k, v in sd.items():
        if k.startswith("encoder.encoder.layer."):
            nk = _remap_key(k)
            if nk is None:
                unmapped.append(k)
            else:
                remapped[nk] = v
        else:
            remapped[k] = v

    if unmapped:
        raise RuntimeError(f"Could not remap checkpoint keys: {unmapped}")

    missing, unexpected = model.load_state_dict(remapped, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"load_state_dict mismatch: missing={missing} unexpected={unexpected}")

    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


load_lewm = load_tworoom_lewm

ACTION_ENCODER_DIM = 10  # both quentinll/lewm-tworooms and lewm-pusht declare action_encoder
# input_dim=10 in their config.json regardless of the real env's action_dim (2 for both) --
# see [[lewm-action-encoder-padding]]. Zero-padding works (front or back position both give
# one-step forward-prediction MSE ~0.13 on real Two-Room transitions vs. ~0.82-0.97 for
# no-action/mean baselines); tiling the action to fill 10 dims is actively worse than doing
# nothing (MSE 1.79) -- confirmed wrong, not just unverified.


def load_local_jepa_checkpoint(ckpt_dir: Path, weights_file: str, device: str = "cpu"):
    """Load a checkpoint produced by THIS repo's own train.py (e.g.
    data/checkpoints/lewm_pusht_1step_predictor) -- distinct from load_tworoom_lewm's HF
    quentinll/lewm-<env> checkpoints. config.json's `_target_`s point at top-level jepa.py/
    module.py (jepa.JEPA, module.ARPredictor, module.Embedder, module.MLP), not
    stable_worldmodel.wm.lewm.*, so those modules must be importable -- add REPO_ROOT to
    sys.path since callers typically run from scripts/. No ViT key remap needed: this state
    dict was saved by the currently-installed library (utils.SaveCkptCallback ->
    stable_worldmodel.wm.utils.save_pretrained, a plain `torch.save(model.state_dict(), ...)`),
    never an older one."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    model = instantiate(cfg)
    sd = torch.load(ckpt_dir / weights_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def pad_action(action):
    """Zero-pad a real (N, action_dim) action array into the (N, ACTION_ENCODER_DIM) shape
    the checkpoint's action_encoder expects. Front-padding is arbitrary among positions that
    were empirically verified to work equally well -- see [[lewm-action-encoder-padding]]."""
    import numpy as np
    n, d = action.shape
    if d == ACTION_ENCODER_DIM:
        return action
    padded = np.zeros((n, ACTION_ENCODER_DIM), dtype=action.dtype)
    padded[:, :d] = action
    return padded
