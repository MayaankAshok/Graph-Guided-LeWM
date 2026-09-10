"""Pilot: do the two calibrated signals exist in a trained LeWM latent?

  u(s,a,s') = residual surprise      -> needs realized s'
  v(s,a)    = predicted-latent typicality (||pred||^2 vs chi^2_d) -> needs only (s,a)

v is the new object: it is the one that can be evaluated at arbitrary actions,
which is what would let us delete twin-Q / IQL's V-net.

Test: real actions vs. shuffled actions (the canonical OOD-action query).
"""
import os, sys, json
import numpy as np
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"C:\Mayaank\IIITH\CSTAR\LEWM")
from utils import get_img_preprocessor, get_column_normalizer

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda"
HIST, NPRED, FRAMESKIP, IMG = 3, 1, 5, 224
N_BATCH, BS = int(os.environ.get("NB", 40)), 8

ROOT = r"C:\Mayaank\IIITH\CSTAR\LEWM"

print("== loading model ==", flush=True)
model = swm.wm.utils.load_pretrained(os.path.join(ROOT, "data", "checkpoints", "lewm")).to(DEV).eval()
model.requires_grad_(False)
model.interpolate_pos_encoding = True

print("== loading dataset ==", flush=True)
ds = swm.data.load_dataset(
    "pusht_expert_train.h5", transform=None, cache_dir=os.path.join(ROOT, "data"),
    num_steps=NPRED + HIST, frameskip=FRAMESKIP,
    keys_to_load=["pixels", "action", "proprio", "state"],
    keys_to_cache=["action", "proprio", "state"],
)
tfs = [get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)]
for col in ["action", "proprio", "state"]:
    tfs.append(get_column_normalizer(ds, col, col))
ds.transform = spt.data.transforms.Compose(*tfs)

loader = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, drop_last=True,
                                     num_workers=0, generator=torch.Generator().manual_seed(0))

def resid_and_pred(emb, action):
    """Return (residual sq-norm, predicted-latent sq-norm) for the final step."""
    act_emb = model.action_encoder(action)
    pred = model.predict(emb[:, :HIST], act_emb[:, :HIST])[:, -1]   # (B, D)
    tgt = emb[:, HIST]                                              # realized next latent
    return (pred - tgt).pow(2).sum(-1), pred.pow(2).sum(-1)

Z2, U_real, U_shuf, U_scal, V_real, V_shuf, V_scal = [], [], [], [], [], [], []

print("== running ==", flush=True)
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= N_BATCH: break
        px = batch["pixels"].to(DEV).float()
        act = torch.nan_to_num(batch["action"].to(DEV).float(), 0.0)
        if i == 0:
            print("  pixels", tuple(px.shape), "action", tuple(act.shape), flush=True)

        emb = model.encode({"pixels": px})["emb"]        # (B, T, D)
        Z2.append(emb.reshape(-1, emb.shape[-1]).pow(2).sum(-1).cpu())

        # counterfactual action sets
        perm = torch.randperm(act.shape[0], device=DEV)
        act_shuf = act[perm]                              # plausible magnitude, wrong for this state
        act_scal = act * 3.0                              # outside the action support

        for a, U, V in ((act, U_real, V_real), (act_shuf, U_shuf, V_shuf), (act_scal, U_scal, V_scal)):
            u, v = resid_and_pred(emb, a)
            U.append(u.cpu()); V.append(v.cpu())
        if (i + 1) % 10 == 0: print(f"  batch {i+1}/{N_BATCH}", flush=True)

cat = lambda x: torch.cat(x).numpy().astype(np.float64)
Z2, U_real, U_shuf, U_scal = cat(Z2), cat(U_real), cat(U_shuf), cat(U_scal)
V_real, V_shuf, V_scal = cat(V_real), cat(V_shuf), cat(V_scal)
D = 192

def q(name, x):
    print(f"  {name:22s} mean={x.mean():10.2f}  med={np.median(x):10.2f}  "
          f"p05={np.percentile(x,5):9.2f}  p95={np.percentile(x,95):10.2f}")

print(f"\n===== A. Is the latent marginal chi^2_{D}? (SIGReg premise) =====")
print(f"  n = {len(Z2)} encoded frames;  chi^2_{D} has mean {D}, sd {np.sqrt(2*D):.1f}")
q("||z||^2", Z2)
print(f"  KS vs chi2_{D}: stat={sps.kstest(Z2, 'chi2', args=(D,)).statistic:.4f}")
# per-dim standardized check
print(f"  implied per-dim scale: {np.sqrt(Z2.mean()/D):.3f}  (1.0 == unit isotropic)")

print(f"\n===== B. u(s,a,s'): residual surprise =====")
q("||delta||^2 real", U_real); q("||delta||^2 shuffled", U_shuf); q("||delta||^2 x3", U_scal)
au_shuf = roc_auc_score(np.r_[np.zeros(len(U_real)), np.ones(len(U_shuf))], np.r_[U_real, U_shuf])
au_scal = roc_auc_score(np.r_[np.zeros(len(U_real)), np.ones(len(U_scal))], np.r_[U_real, U_scal])
print(f"  AUROC real-vs-shuffled = {au_shuf:.3f}")
print(f"  AUROC real-vs-scaled   = {au_scal:.3f}")

print(f"\n===== C. v(s,a): predicted-latent typicality (NO s' needed) =====")
q("||pred||^2 real", V_real); q("||pred||^2 shuffled", V_shuf); q("||pred||^2 x3", V_scal)
# two-sided atypicality against the chi^2_D null
atyp = lambda x: 1.0 - 2.0 * np.minimum(sps.chi2.cdf(x, D), 1 - sps.chi2.cdf(x, D))
av_shuf = roc_auc_score(np.r_[np.zeros(len(V_real)), np.ones(len(V_shuf))], np.r_[atyp(V_real), atyp(V_shuf)])
av_scal = roc_auc_score(np.r_[np.zeros(len(V_real)), np.ones(len(V_scal))], np.r_[atyp(V_real), atyp(V_scal)])
print(f"  AUROC(atypicality) real-vs-shuffled = {av_shuf:.3f}")
print(f"  AUROC(atypicality) real-vs-scaled   = {av_scal:.3f}")

json.dump(dict(n_frames=int(len(Z2)), n_trans=int(len(U_real)), d=D,
               z2_mean=float(Z2.mean()), z2_med=float(np.median(Z2)),
               u_real_med=float(np.median(U_real)), u_shuf_med=float(np.median(U_shuf)),
               u_scal_med=float(np.median(U_scal)),
               v_real_med=float(np.median(V_real)), v_shuf_med=float(np.median(V_shuf)),
               v_scal_med=float(np.median(V_scal)),
               auroc_u_shuf=float(au_shuf), auroc_u_scal=float(au_scal),
               auroc_v_shuf=float(av_shuf), auroc_v_scal=float(av_scal)),
          open(os.path.join(os.path.dirname(__file__), "pilot_results.json"), "w"), indent=2)
print("\nwrote pilot_results.json")
