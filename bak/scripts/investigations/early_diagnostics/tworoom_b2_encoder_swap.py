"""B2: is the B0/B1 graph advantage about LeWM specifically, or about any latent?

Same 300 real episodes, same pixels, same pipeline (calibrate rho_hat from adjacent
pairs -> chi^2 threshold at q=1e-3 -> transition + identification edges, id_weight=1
-> Spearman vs. the same wall-respecting ground-truth oracle used in B0/B1), with only
the encoder swapped:

  1. LeWM (pretrained quentinll/lewm-tworooms)      -- already run in B0, reused here
  2. random-init ViT-tiny (identical architecture, untrained weights)
  3. raw pixels (16x16 average-pooled RGB, flattened -- no network at all)
  4. PCA on the pooled pixels (192 components, fit on the FULL dataset)

DINO-WM / PLDM / LeJEPA (the other baselines named in the proposal) ship only via a
Google Drive folder linked from the repo's README; confirmed dead (direct fetch
404s), and no alternative source exists -- checked quentinll's HuggingFace profile
(only the 4 LeWM model/dataset repos) and each baseline's own independent repo
(gaoyuezhou/dino_wm, vladisai/PLDM, rbalestr-lab/lejepa), none of which host a
Two-Room-specific checkpoint. Not run here -- flagged explicitly rather than
silently skipped, see the report.

Arm 4 (PCA) exists because raw pixels (arm 3) failed for a specific, diagnosed
reason: the fixed wall/door background dominates the 768-dim pooled-pixel vector so
completely that near/far pairs are barely distinguishable in Euclidean terms at any
threshold (median cross-episode sqdist 2.24). PCA's entire job is finding the
directions that actually vary (agent position) and discarding the near-zero-variance
ones (the fixed background) -- a near-free (no training loop) way to test whether
that alone fixes it. Its 192-component basis is fit via IncrementalPCA over the
ENTIRE dataset (920,809 frames, all 10,000 episodes, not just the 300-episode
landmark subset) -- fitting the basis on much more data than the 300 episodes it's
then evaluated on gives the most stable possible estimate of the dominant variance
directions, cheap to do since pooling is a pure CPU operation with no network
forward pass.

Kill criterion (from docs/graph-proposal/main.tex): if a RANDOM ViT also scores near
LeWM's 0.95, the SIGReg-specific framing is dead -- the graph mechanism might still
work, but the paper's story about *why* changes completely.
"""

import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- required for the h5 file's pixel compression filter.
# Earlier scripts (B0/B1) never needed this explicitly because load_tworoom_lewm()
# transitively imports stable_worldmodel, which imports hdf5plugin as a side effect,
# registering the filter before any raw pixel read happens. Here, when the LeWM stage
# is a pure cache hit (no model load), that side effect never fires, and the first
# fresh h5py pixel read fails with a cryptic "OSError: can't open directory" -- caught
# by testing this exact read in isolation, not by inspection.
import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))  # own dir -- finds tworoom_b1_graph_diagnostics sibling
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # scripts/ -- finds tworoom_b0_graph_gate etc.
from tworoom_b0_graph_gate import (
    DEV, H5_PATH, IMAGENET_MEAN, IMAGENET_STD,
    build_true_distance_oracle, build_weighted_graph, find_graph_edges, log,
)
from tworoom_b1_graph_diagnostics import lean_evaluate

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS = ROOT / "outputs" / "b0_tworoom" / "landmarks_ep300.npz"
OUT_DIR = ROOT / "outputs" / "b2_tworoom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Q_DEFAULT = 1e-3
N_EPISODES = 300
SEED = 0


def calibrate_rho(z, ep_idx, step_idx, d):
    order = np.lexsort((step_idx, ep_idx))
    ep_o, step_o = ep_idx[order], step_idx[order]
    adj = (ep_o[1:] == ep_o[:-1]) & (step_o[1:] == step_o[:-1] + 1)
    z_o = z[order]
    z_t, z_t1 = z_o[:-1][adj], z_o[1:][adj]
    rho_hat = float(np.mean(np.sum(z_t * z_t1, axis=1)) / d)
    return rho_hat


def encode_dataset(encode_fn, cache_name, out_dim, chosen_ep):
    """Run encode_fn over the EXACT episode set used in the cached LeWM landmarks.

    encode_fn: batch of pixels (B,224,224,3) uint8 numpy -> (B, out_dim) numpy
    `chosen_ep` is read directly off the cached LeWM landmarks (np.unique(ep_idx)),
    not re-derived by replaying an RNG -- replaying seeds is fragile (the cached
    file could have been produced by an earlier code path with different RNG
    consumption before it) and this is directly, cheaply verifiable instead.
    """
    cache = OUT_DIR / f"landmarks_{cache_name}.npz"
    if cache.exists():
        log(f"[encode:{cache_name}] loading cached")
        d = np.load(cache)
        return d["z"], d["ep_idx"], d["step_idx"], d["proprio"]

    f = h5py.File(H5_PATH, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    ep_offset = f["ep_offset"][:]
    ep_len = f["ep_len"][:]

    all_z, all_ep, all_step, all_pos = [], [], [], []
    t0 = time.time()
    BS = 128
    for ei, e in enumerate(chosen_ep):
        s, L = int(ep_offset[e]), int(ep_len[e])
        pix = f["pixels"][s : s + L]
        pos = f["proprio"][s : s + L]
        steps = np.arange(L)

        z_chunks = []
        for b0 in range(0, L, BS):
            chunk = pix[b0 : b0 + BS]
            z_chunks.append(encode_fn(chunk))
        z_ep = np.concatenate(z_chunks, axis=0)
        assert z_ep.shape == (L, out_dim), f"{z_ep.shape} != {(L, out_dim)}"

        all_z.append(z_ep)
        all_ep.append(np.full(L, e, dtype=np.int64))
        all_step.append(steps)
        all_pos.append(pos)

        if ei % 100 == 0:
            log(f"[encode:{cache_name}] episode {ei}/{N_EPISODES} "
                f"cum_frames={sum(len(x) for x in all_ep)} elapsed={time.time()-t0:.1f}s")

    f.close()
    z = np.concatenate(all_z, axis=0).astype(np.float32)
    ep_idx = np.concatenate(all_ep, axis=0)
    step_idx = np.concatenate(all_step, axis=0)
    proprio = np.concatenate(all_pos, axis=0).astype(np.float32)
    log(f"[encode:{cache_name}] done: {z.shape[0]} frames in {time.time()-t0:.1f}s")
    np.savez(cache, z=z, ep_idx=ep_idx, step_idx=step_idx, proprio=proprio)
    return z, ep_idx, step_idx, proprio


def count_identification_edges(z, ep_idx, eps2, batch=500):
    """Count cross-episode pairs under threshold WITHOUT materializing the index
    arrays -- find_graph_edges' actual edge-count-explosion cases (raw pixels: 381M
    edges; PCA turns out to hit the same ~382M) have caused a hard native segfault on
    this machine when concatenating index arrays of that size (no Python exception to
    catch -- the whole process dies). This gives the same count via a running sum of
    boolean-mask sums per chunk, peak memory bounded to one (batch, n) tensor."""
    n = z.shape[0]
    z_t = torch.from_numpy(z).to(DEV)
    ep_t = torch.from_numpy(ep_idx).to(DEV)
    total = 0
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        sqd = torch.cdist(z_t[i0:i1], z_t, p=2) ** 2
        diff_ep = ep_t[i0:i1].unsqueeze(1) != ep_t.unsqueeze(0)
        below = (sqd < eps2) & diff_ep
        total += int(below.sum().item())
        del sqd, below
    return total // 2  # each unordered pair counted twice (i,j) and (j,i)


def cross_episode_sqdist_diagnostic(z, ep_idx, n_pairs=20000, seed=0):
    """Median/mean squared distance on a random sample of cross-episode pairs -- the
    same diagnostic already used to explain raw pixels' failure (median 2.24: the
    fixed background dominates so completely that near/far pairs are barely
    distinguishable at any threshold)."""
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = ep_idx[i] != ep_idx[j]
    i, j = i[keep], j[keep]
    sqd = np.sum((z[i] - z[j]) ** 2, axis=1)
    return dict(median=float(np.median(sqd)), mean=float(sqd.mean()), std=float(sqd.std()))


# above this count, materializing find_graph_edges' index arrays has hit a hard
# native segfault on this machine (no catchable Python exception) -- checked cheaply
# first via count_identification_edges instead of attempting it blind.
EDGE_COUNT_DANGER_THRESHOLD = 10_000_000


def run_pipeline(name, z, ep_idx, step_idx, proprio, true_dist_oracle):
    n, d = z.shape
    rho_hat = calibrate_rho(z, ep_idx, step_idx, d)
    eps2 = 2 * (1 - rho_hat) * sps.chi2.ppf(Q_DEFAULT, d)
    log(f"[{name}] d={d} rho_hat={rho_hat:.4f} eps2={eps2:.2f}")

    log(f"[{name}] checking identification-edge count is safe to materialize...")
    n_edges_est = count_identification_edges(z, ep_idx, eps2)
    if n_edges_est > EDGE_COUNT_DANGER_THRESHOLD:
        sqdist = cross_episode_sqdist_diagnostic(z, ep_idx)
        log(f"[{name}] INTRACTABLE: {n_edges_est} identification edges (over the "
            f"{EDGE_COUNT_DANGER_THRESHOLD} safety threshold) -- skipping full graph "
            f"construction rather than risk a repeat of raw_pixels' hard segfault. "
            f"cross-episode sqdist median={sqdist['median']:.4f} mean={sqdist['mean']:.4f}")
        return dict(d=d, rho_hat=rho_hat, eps2=eps2, n_id_edges_est=n_edges_est,
                    intractable=True, cross_episode_sqdist=sqdist)

    trans_i, trans_j, id_i, id_j = find_graph_edges(z, ep_idx, step_idx, eps2)
    avg_deg = 2 * len(id_i) / n
    graph = build_weighted_graph(n, trans_i, trans_j, id_i, id_j, id_weight=1.0)
    ev = lean_evaluate(z, proprio, graph, true_dist_oracle)

    log(f"[{name}] n_id_edges={len(id_i)} avg_degree={avg_deg:.1f}")
    log(f"[{name}] overall: eucl={ev['overall']['euclidean']:.4f} graph={ev['overall']['graph']:.4f}")
    log(f"[{name}] same-room: eucl={ev['same_room']['euclidean']:.4f} graph={ev['same_room']['graph']:.4f}")
    log(f"[{name}] cross-room: eucl={ev['cross_room']['euclidean']:.4f} graph={ev['cross_room']['graph']:.4f}")

    return dict(d=d, rho_hat=rho_hat, eps2=eps2, n_id_edges=int(len(id_i)),
                avg_degree=avg_deg, evaluation=ev)


# ============================================================
# Encoders
# ============================================================

def make_lewm_encoder():
    from tworoom_lewm_loader import load_tworoom_lewm
    model = load_tworoom_lewm(device=DEV)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    def encode_fn(chunk):
        t = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        t = (t - mean) / std
        with torch.no_grad():
            out = model.encode({"pixels": t.unsqueeze(1)})
        return out["emb"][:, 0].cpu().numpy()

    return encode_fn, 192


def make_random_vit_encoder():
    import stable_pretraining as spt
    torch.manual_seed(1234)
    vit = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False
    ).to(DEV).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)

    def encode_fn(chunk):
        t = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        t = (t - mean) / std
        with torch.no_grad():
            out = vit(t, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]
        return cls.cpu().numpy()

    return encode_fn, 192


def make_raw_pixel_encoder(pool=14):
    """16x16 average-pooled RGB, flattened -- 16*16*3 = 768 dims. No network."""
    def encode_fn(chunk):
        t = torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0  # (B,3,224,224)
        pooled = Fnn.avg_pool2d(t, kernel_size=pool, stride=pool)  # (B,3,16,16)
        flat = pooled.reshape(pooled.size(0), -1)
        return flat.numpy()

    out_dim = (224 // pool) * (224 // pool) * 3
    return encode_fn, out_dim


def fit_pca_on_full_dataset(n_components=192, pool=14, batch_size=256):
    """Fit a PCA basis on 16x16-pooled RGB pixels (768 dims) over the FULL dataset --
    all 920,809 frames across all 10,000 episodes -- not just the 300-episode landmark
    subset used for the actual graph/eval comparison. Uses sklearn's IncrementalPCA so
    memory stays bounded to one batch at a time regardless of dataset size (a full
    (920809, 768) float32 array would be ~2.8GB -- feasible but needless when
    IncrementalPCA gets an equivalent basis via streaming batches).

    Cached to disk (mean + components as plain arrays, not a pickled sklearn object,
    so the cache format doesn't depend on the installed sklearn version)."""
    cache = OUT_DIR / f"pca_basis_k{n_components}.npz"
    if cache.exists():
        log(f"[pca] loading cached basis from {cache}")
        d = np.load(cache)
        return d["mean"], d["components"]

    from sklearn.decomposition import IncrementalPCA

    f = h5py.File(H5_PATH, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
    n_frames = f["pixels"].shape[0]
    log(f"[pca] fitting IncrementalPCA(n_components={n_components}) over all {n_frames} frames "
        f"in the dataset (batch_size={batch_size})...")

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    t0 = time.time()
    for b0 in range(0, n_frames, batch_size):
        b1 = min(n_frames, b0 + batch_size)
        if b1 - b0 < n_components:
            break  # IncrementalPCA requires each batch >= n_components; drop a short tail batch
        chunk = f["pixels"][b0:b1]  # (B,224,224,3) uint8, contiguous read
        t = torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0
        pooled = Fnn.avg_pool2d(t, kernel_size=pool, stride=pool)
        flat = pooled.reshape(pooled.size(0), -1).numpy()
        ipca.partial_fit(flat)
        if (b0 // batch_size) % 50 == 0:
            log(f"[pca] fit progress {b1}/{n_frames} elapsed={time.time()-t0:.1f}s")
    f.close()

    log(f"[pca] fit done in {time.time()-t0:.1f}s -- "
        f"explained variance ratio (top 192 components) sum={ipca.explained_variance_ratio_.sum():.4f}")
    mean = ipca.mean_.astype(np.float32)
    components = ipca.components_.astype(np.float32)
    np.savez(cache, mean=mean, components=components,
             explained_variance_ratio=ipca.explained_variance_ratio_)
    return mean, components


def make_pca_encoder(mean, components, pool=14):
    """PCA projection of 16x16-pooled RGB pixels onto a basis fit over the full
    dataset (fit_pca_on_full_dataset). Output dim = components.shape[0]."""
    mean_t = torch.from_numpy(mean)
    comp_t = torch.from_numpy(components)  # (k, 768)

    def encode_fn(chunk):
        t = torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0
        pooled = Fnn.avg_pool2d(t, kernel_size=pool, stride=pool)
        flat = pooled.reshape(pooled.size(0), -1)
        projected = (flat - mean_t) @ comp_t.T
        return projected.numpy()

    return encode_fn, components.shape[0]


def run_arm(name, results, fn):
    """Run one encoder arm in isolation -- a crash in one (e.g. random_vit's collapsed
    representation exploding into ~100M candidate identification edges, which has hit
    transient OOM on this machine under memory pressure before) must not prevent the
    other arms, especially new ones, from running and being reported. Note this only
    catches ordinary Python exceptions (MemoryError included) -- raw_pixels' 381M-edge
    explosion has hit a hard native segfault on this machine, which kills the whole
    process regardless of any try/except here; that's why raw_pixels and random_vit
    (the two arms already known to be memory-hazardous) are ordered LAST, after PCA."""
    try:
        results[name] = fn()
    except Exception as e:
        log(f"[{name}] FAILED: {type(e).__name__}: {e}")
        results[name] = {"error": f"{type(e).__name__}: {e}"}
    import json
    (OUT_DIR / "b2_results.json").write_text(json.dumps(results, indent=2, default=str))


def main():
    log(f"device={DEV}")
    true_dist_oracle = build_true_distance_oracle()

    results = {}

    # 1. LeWM -- reuse the canonical B0 cache directly rather than re-encoding
    log("\n" + "=" * 60 + "\nEncoder 1/4: LeWM (pretrained)\n" + "=" * 60)
    d = np.load(LANDMARKS)
    z, ep_idx, step_idx, proprio = d["z"], d["ep_idx"], d["step_idx"], d["proprio"]
    chosen_ep = np.unique(ep_idx)
    chosen_ep.sort()
    log(f"reusing exact episode set from cached LeWM landmarks: {len(chosen_ep)} episodes, "
        f"{len(ep_idx)} frames")
    run_arm("lewm_pretrained", results,
             lambda: run_pipeline("lewm", z, ep_idx, step_idx, proprio, true_dist_oracle))

    # 2. PCA on the pooled pixels, basis fit over the full 920,809-frame dataset --
    # ordered right after LeWM (before random_vit/raw_pixels below) since those two are
    # already known to be memory-hazardous on this machine, up to and including a hard
    # native segfault for raw_pixels that no try/except can survive; results are saved
    # incrementally (see run_arm) so a later crash can't erase this arm's outcome.
    log("\n" + "=" * 60 + "\nEncoder 2/4: PCA (192 components, fit on full dataset)\n" + "=" * 60)
    mean, components = fit_pca_on_full_dataset(n_components=192)
    encode_fn4, out_dim4 = make_pca_encoder(mean, components)
    z4, ep4, st4, pr4 = encode_dataset(encode_fn4, "pca192", out_dim4, chosen_ep)
    assert np.array_equal(ep4, ep_idx) and np.array_equal(st4, step_idx), "frame set mismatch vs LeWM run"
    assert np.allclose(pr4, proprio), "proprio mismatch vs LeWM run"
    run_arm("pca", results,
             lambda: run_pipeline("pca", z4, ep4, st4, pr4, true_dist_oracle))

    # 3. random-init ViT, identical architecture
    log("\n" + "=" * 60 + "\nEncoder 3/4: random-init ViT-tiny (untrained)\n" + "=" * 60)
    encode_fn, out_dim = make_random_vit_encoder()
    z2, ep2, st2, pr2 = encode_dataset(encode_fn, "random_vit", out_dim, chosen_ep)
    assert np.array_equal(ep2, ep_idx) and np.array_equal(st2, step_idx), "frame set mismatch vs LeWM run"
    assert np.allclose(pr2, proprio), "proprio mismatch vs LeWM run"
    run_arm("random_vit", results,
             lambda: run_pipeline("random_vit", z2, ep2, st2, pr2, true_dist_oracle))

    # 4. raw pixels, no network
    log("\n" + "=" * 60 + "\nEncoder 4/4: raw pixels (16x16 avg-pooled RGB)\n" + "=" * 60)
    encode_fn3, out_dim3 = make_raw_pixel_encoder()
    z3, ep3, st3, pr3 = encode_dataset(encode_fn3, "raw_pixels", out_dim3, chosen_ep)
    assert np.array_equal(ep3, ep_idx) and np.array_equal(st3, step_idx), "frame set mismatch vs LeWM run"
    assert np.allclose(pr3, proprio), "proprio mismatch vs LeWM run"
    run_arm("raw_pixels", results,
             lambda: run_pipeline("raw_pixels", z3, ep3, st3, pr3, true_dist_oracle))

    log(f"\nfinal results already written incrementally to {OUT_DIR / 'b2_results.json'}")

    log("\n" + "=" * 60 + "\nSUMMARY (300 episodes)\n" + "=" * 60)
    log(f"{'encoder':<18}{'d':>5}{'overall(g)':>12}{'same(g)':>10}{'cross(g)':>10}{'overall(eucl)':>15}")
    for name, r in results.items():
        if "error" in r:
            log(f"{name:<18}FAILED: {r['error']}")
            continue
        if r.get("intractable"):
            log(f"{name:<18}{r['d']:>5}  INTRACTABLE (~{r['n_id_edges_est']} identification edges, "
                f"cross-episode sqdist median={r['cross_episode_sqdist']['median']:.4f})")
            continue
        ev = r["evaluation"]
        log(f"{name:<18}{r['d']:>5}{ev['overall']['graph']:>12.4f}{ev['same_room']['graph']:>10.4f}"
            f"{ev['cross_room']['graph']:>10.4f}{ev['overall']['euclidean']:>15.4f}")

    # ============================================================
    # Small-scale, apples-to-apples comparison: raw_pixels and PCA are intractable at
    # 300 episodes because avg. degree scales ~linearly with landmark count there (near-
    # constant near-duplicate density), so total edges scale ~quadratically -- unlike
    # LeWM, where degree stays roughly constant with scale (B1's own finding). Checked
    # directly: at 10 episodes raw_pixels/PCA drop to ~440K-450K edges, comfortably safe.
    # Re-uses the SAME already-encoded arrays (z, z2, z3, z4) for all four encoders --
    # no re-encoding needed, just a boolean mask on episode id.
    # ============================================================
    N_SMALL = 10
    small_ep = chosen_ep[:N_SMALL]
    mask = np.isin(ep_idx, small_ep)
    log(f"\n{'=' * 60}\nSmall-scale ({N_SMALL} episodes, {int(mask.sum())} frames), "
        f"same episodes across all four encoders\n{'=' * 60}")

    small_results = {}
    small_arms = [
        ("lewm_pretrained", z), ("pca", z4), ("random_vit", z2), ("raw_pixels", z3),
    ]
    for name, zz in small_arms:
        run_arm_into = lambda zz=zz, name=name: run_pipeline(
            f"{name}_n{N_SMALL}", zz[mask], ep_idx[mask], step_idx[mask], proprio[mask], true_dist_oracle)
        try:
            small_results[name] = run_arm_into()
        except Exception as e:
            log(f"[{name}_n{N_SMALL}] FAILED: {type(e).__name__}: {e}")
            small_results[name] = {"error": f"{type(e).__name__}: {e}"}
        import json
        (OUT_DIR / f"b2_small_scale_n{N_SMALL}_results.json").write_text(
            json.dumps(small_results, indent=2, default=str))

    log(f"\n{'=' * 60}\nSUMMARY ({N_SMALL} episodes, same subset, all four encoders)\n{'=' * 60}")
    log(f"{'encoder':<18}{'d':>5}{'overall(g)':>12}{'same(g)':>10}{'cross(g)':>10}{'overall(eucl)':>15}")
    for name, r in small_results.items():
        if "error" in r:
            log(f"{name:<18}FAILED: {r['error']}")
            continue
        if r.get("intractable"):
            log(f"{name:<18}{r['d']:>5}  still INTRACTABLE (~{r['n_id_edges_est']} edges)")
            continue
        ev = r["evaluation"]
        log(f"{name:<18}{r['d']:>5}{ev['overall']['graph']:>12.4f}{ev['same_room']['graph']:>10.4f}"
            f"{ev['cross_room']['graph']:>10.4f}{ev['overall']['euclidean']:>15.4f}")

    log("\nNOTE: DINO-WM / PLDM / LeJEPA baselines not run -- only shipped via a Google "
        "Drive folder linked from the repo README that 404s on direct fetch, confirmed dead "
        "with no alternative source. See script docstring.")


if __name__ == "__main__":
    main()
