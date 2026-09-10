"""Precompute frozen encoder+projector embeddings for every frame in an h5 dataset, so
training a predictor on top of a frozen encoder never has to repeat that ViT forward pass.

Motivation: for lewm_pusht_1step.yaml (frozen encoder, tiny predictor), the ViT-tiny forward
pass dominates every training step (measured ~2s/step at batch 256 on a GTX 1080 Ti -- the
predictor itself is a few hundred thousand params and would otherwise be near-free). Since
the encoder is frozen, its output per frame never changes -- computing it once and caching
the result turns a several-hour-per-epoch bottleneck into a ~1.7GB embeddings file (2.3M
frames x 192 floats) that trains at effectively predictor-only speed. It also means never
needing to re-stage the full ~46GB raw-pixel dataset onto /ssd_scratch after a node change --
only this small file needs to move.

Output schema mirrors the input h5's (ep_len/ep_offset/action + a new "emb" column replacing
"pixels"), so it loads through stable_worldmodel's existing HDF5Dataset/load_dataset unchanged
-- just point config/train/data/pusht_emb.yaml's `name` at it with keys_to_load=[emb, action].

Usage:
  python precompute_embeddings.py \
    --h5-path /ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5 \
    --ckpt-dir /home2/mayaank.ashok/lewm_research/data/checkpoints/models--quentinll--lewm-pusht \
    --out-path /share1/mayaank.ashok/lewm_data/pusht_expert_train_emb.h5
"""

import argparse
from pathlib import Path

import h5py
import hydra
import torch
from torch import nn

from train import load_pretrained_encoder
from utils import get_img_preprocessor


class EncoderProjector(nn.Module):
    """Just the two frozen submodules precompute needs -- load_pretrained_encoder() only
    looks for `encoder`/`projector` attributes, so this is enough without instantiating an
    (unused) predictor/action_encoder too."""

    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector


def build_encoder_projector(img_size: int, embed_dim: int, ckpt_dir: Path, device: str):
    encoder = hydra.utils.instantiate({
        "_target_": "stable_pretraining.backbone.utils.vit_hf",
        "size": "tiny",
        "patch_size": 14,
        "image_size": img_size,
        "pretrained": False,
        "use_mask_token": False,
    })
    projector = hydra.utils.instantiate({
        "_target_": "module.MLP",
        "input_dim": embed_dim,
        "output_dim": embed_dim,
        "hidden_dim": 2048,
        "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
    })
    model = EncoderProjector(encoder, projector)
    load_pretrained_encoder(model, ckpt_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5-path", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model = build_encoder_projector(args.img_size, args.embed_dim, Path(args.ckpt_dir), args.device)
    # Same preprocessing stable_worldmodel's HDF5Dataset applies before handing "pixels" to
    # JEPA.encode() -- see stable_worldmodel/data/formats/hdf5.py's _load_slice (permutes
    # HWC->CHW before the transform) and utils.get_img_preprocessor (ToImage + resize).
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5_path, "r") as fin:
        n = fin["pixels"].shape[0]
        ep_len = fin["ep_len"][:]
        ep_offset = fin["ep_offset"][:]
        action = fin["action"][:]

        with h5py.File(out_path, "w") as fout:
            fout.create_dataset("emb", shape=(n, args.embed_dim), dtype="float32")
            fout.create_dataset("action", data=action)
            fout.create_dataset("ep_len", data=ep_len)
            fout.create_dataset("ep_offset", data=ep_offset)

            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                batch = fin["pixels"][start:end]  # (B, H, W, C) uint8
                pixels = torch.from_numpy(batch).permute(0, 3, 1, 2)
                pixels = transform({"pixels": pixels})["pixels"]
                pixels = pixels.to(args.device, non_blocking=True)

                with torch.no_grad():
                    out = model.encoder(pixels, interpolate_pos_encoding=True)
                    cls = out.last_hidden_state[:, 0]
                    emb = model.projector(cls)

                fout["emb"][start:end] = emb.float().cpu().numpy()
                if (start // args.batch_size) % 20 == 0:
                    print(f"{end}/{n}", flush=True)

    print(f"done: wrote {n} embeddings to {out_path}")


if __name__ == "__main__":
    main()
