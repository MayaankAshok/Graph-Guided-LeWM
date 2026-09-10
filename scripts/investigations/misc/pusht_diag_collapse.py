"""Is the real pretrained quentinll/lewm-pusht checkpoint collapsed, like the local
from-scratch epoch-1 one (data/checkpoints/lewm/weights_epoch_1.pt) was? Same checks as
diag_collapse.py, but loads via tworoom_lewm_loader's key-remapping loader (needed for any
externally-produced HF checkpoint -- swm.wm.utils.load_pretrained lacks the remap and only
happened to work for the from-scratch checkpoint because that one was trained with the
currently-installed stable_pretraining's key names already)."""
import os
import sys
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root
from tworoom_lewm_loader import load_tworoom_lewm
from utils import get_column_normalizer, get_img_preprocessor

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = ROOT / "data" / "checkpoints" / "models--quentinll--lewm-pusht"

torch.manual_seed(0)
model = load_tworoom_lewm(ckpt_dir=CKPT_DIR, device=DEV)
model.interpolate_pos_encoding = True

ds = swm.data.load_dataset(
    "pusht_expert_train.h5", transform=None,
    cache_dir=os.path.join(ROOT, "data"), num_steps=4, frameskip=5,
    keys_to_load=["pixels", "action", "proprio", "state"],
    keys_to_cache=["action", "proprio", "state"],
)
tfs = [get_img_preprocessor("pixels", "pixels", 224)]
for c in ["action", "proprio", "state"]:
    tfs.append(get_column_normalizer(ds, c, c))
ds.transform = spt.data.transforms.Compose(*tfs)
loader = torch.utils.data.DataLoader(
    ds, batch_size=16, shuffle=True, num_workers=0,
    generator=torch.Generator().manual_seed(1),
)

batch = next(iter(loader))
px = batch["pixels"].to(DEV).float()
B, T = px.shape[:2]
flat = px.reshape(B * T, *px.shape[2:])
print(f"input pixels: shape={tuple(flat.shape)}  std={flat.std().item():.4f}  "
      f"per-sample-mean spread={flat.mean(dim=(1, 2, 3)).std().item():.4f}")

with torch.no_grad():
    out = model.encoder(flat, interpolate_pos_encoding=True)
    cls = out.last_hidden_state[:, 0]
    proj = model.projector(cls)


def report(name, x):
    per_dim_std = x.std(dim=0)
    xc = (x - x.mean(0)).double()
    ev = torch.linalg.eigvalsh(xc.T @ xc / max(len(x) - 1, 1)).clamp_min(0)
    pr = (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30)).item()
    print(f"\n{name}:")
    print(f"  mean ||x||^2 over samples   = {x.pow(2).sum(-1).mean().item():.3f}")
    print(f"  std of ||x||^2 ACROSS samples = {x.pow(2).sum(-1).std().item():.5f}   <-- ~0 means collapse")
    print(f"  mean per-dim std across samples = {per_dim_std.mean().item():.6f}")
    print(f"  participation ratio (d_eff) = {pr:.2f} of {x.shape[-1]}")


report("ViT cls token (pre-projector)", cls)
report("projector output = SIGReg target", proj)

with torch.no_grad():
    e0 = model.encoder(flat[0:1], interpolate_pos_encoding=True).last_hidden_state[:, 0]
    e1 = model.encoder(flat[1:2], interpolate_pos_encoding=True).last_hidden_state[:, 0]
cos = torch.nn.functional.cosine_similarity(e0, e1).item()
print(f"\ncosine(cls_img0, cls_img1) = {cos:.6f}   (1.0 == identical output for different inputs)")
print(f"relative L2 gap            = {((e0 - e1).norm() / e0.norm()).item():.6f}")

# mean off-diagonal cosine over all pairs in the batch, on the actual "z" B0/graph pipeline
# uses (projector output) -- directly comparable to main.tex's tworoom "off-diag cosine ~0.03"
zc = proj / proj.norm(dim=-1, keepdim=True)
sim = zc @ zc.T
off = sim[~torch.eye(len(proj), dtype=bool, device=DEV)]
print(f"\nmean off-diagonal cosine over all {len(off)} pairs (projector/z) = {off.mean().item():.6f}")
