"""Does the graph hitting-time label need Dijkstra, or is a one-stitch construction enough?

One-stitch ("tuple") distance: follow the start's own episode forward to an identification
edge (u -> v) into the goal's episode, then follow that episode forward to the goal:
    T_1(s, g) = min over id edges (u, v), ep(u) = ep(s), u >= s, ep(v) = ep(g), v <= g
                of (u - s) + 1 + (g - v),
plus the direct same-episode gap when ep(s) = ep(g), s <= g. Dijkstra's T_G additionally
allows any number of id hops and any interleaving. This script builds the same graph as
viability_graph_labels.py, takes the label bank's pairs, and reports (i) how many id hops the
Dijkstra shortest paths actually use, (ii) how often T_1 == T_G, (iii) how much coverage is
lost under one-stitch only, (iv) the wall-clock of both.

Run:
    python scripts/investigations/viability_onehop_vs_dijkstra.py \
        --cache outputs/pusht/critic_training/cache_1000_s0.npz --labels outputs/pusht/critic_training/labels_1000_s0.npz
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.graph_lib import build_identification_edges, build_weighted_graph, find_transition_edges
from common.log_util import log
from common.viability import SKIP


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="outputs/pusht/critic_training/cache_1000_s0.npz")
    p.add_argument("--labels", default="outputs/pusht/critic_training/labels_1000_s0.npz")
    p.add_argument("--n-goals", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    c = np.load(args.cache)
    z = c["z"].astype(np.float32)
    ep_offset, ep_len = c["ep_offset"].astype(np.int64), c["ep_len"].astype(np.int64)
    ep_idx = np.repeat(np.arange(len(ep_len)), ep_len)
    step_idx = np.concatenate([np.arange(l) for l in ep_len])
    L = np.load(args.labels)
    meta = json.loads(str(L["meta"]))
    ginfo, b_max = meta["graph"], meta["b_max"]
    limit = SKIP * b_max + 0.5

    ti, tj = find_transition_edges(ep_idx, step_idx)
    id_i, id_j, _, eps2 = build_identification_edges(z, ep_idx, step_idx, k=ginfo["id_k"], eps2_override=ginfo["eps2"])
    graph = build_weighted_graph(len(z), ti, tj, id_i, id_j, 1.0)
    graph_t = graph.T.tocsr()
    # id edges grouped by (episode of u, episode of v), both orientations
    by_pair = defaultdict(list)
    for a, b in zip(id_i, id_j):
        by_pair[(int(ep_idx[a]), int(ep_idx[b]))].append((int(a), int(b)))
        by_pair[(int(ep_idx[b]), int(ep_idx[a]))].append((int(b), int(a)))
    by_pair = {k: np.array(v) for k, v in by_pair.items()}
    # is (u -> v) an id edge? (as opposed to a temporal edge u -> u+1 in the same episode)
    is_temporal = lambda u, v: ep_idx[u] == ep_idx[v] and v == u + 1

    rng = np.random.default_rng(args.seed)
    goals_all = np.unique(L["goal"])
    goals = rng.choice(goals_all, size=min(args.n_goals, len(goals_all)), replace=False)
    hops_hist = defaultdict(int)
    n_pairs = agree = t1_disc_tg_fin = both_fin = both_disc = tg_disc_t1_fin = 0
    gaps = []
    t_dij = t_one = 0.0
    for g in goals:
        m = L["goal"] == g
        starts, tg = L["start"][m], L["tg"][m]
        t0 = time.time()
        D, P = dijkstra(graph_t, directed=True, indices=[int(g)], limit=limit, return_predecessors=True)
        D, P = D[0], P[0]
        t_dij += time.time() - t0
        # hop count along the reversed-graph shortest path from g back to each start
        for s, tgs in zip(starts, tg):
            n_pairs += 1
            if np.isfinite(D[s]):
                hops, node = 0, int(s)
                while node != g and P[node] >= 0:
                    nxt = int(P[node])          # predecessor in the reversed graph = next node towards g
                    if not is_temporal(node, nxt):
                        hops += 1
                    node = nxt
                hops_hist[min(hops, 5)] += 1
        # one-stitch distance for the same pairs
        t0 = time.time()
        eg = int(ep_idx[g])
        for s, tgs in zip(starts, tg):
            es = int(ep_idx[s])
            best = np.inf
            if es == eg and s <= g:
                best = float(g - s)
            edges = by_pair.get((es, eg))
            if edges is not None:
                u, v = edges[:, 0], edges[:, 1]
                ok = (u >= s) & (v <= g)
                if ok.any():
                    best = min(best, float(((u[ok] - s) + 1 + (g - v[ok])).min()))
            t1 = best if best <= limit else np.inf
            tg_fin = tgs >= 0
            if tg_fin and np.isfinite(t1):
                both_fin += 1
                agree += int(abs(t1 - tgs) < 0.5)
                gaps.append(t1 - tgs)
            elif tg_fin and not np.isfinite(t1):
                t1_disc_tg_fin += 1
            elif (not tg_fin) and np.isfinite(t1):
                tg_disc_t1_fin += 1
            else:
                both_disc += 1
        t_one += time.time() - t0

    tot_fin = sum(hops_hist.values())
    log(f"graph: {ginfo}")
    log(f"{len(goals)} goals, {n_pairs} pairs | dijkstra {t_dij:.1f}s, one-stitch {t_one:.1f}s")
    log("id hops on Dijkstra shortest paths (finite pairs): " +
        " ".join(f"{k}{'+' if k == 5 else ''} hops: {v / tot_fin:.2%}" for k, v in sorted(hops_hist.items())))
    log(f"pairs finite under both: {both_fin / n_pairs:.2%}  (T_1 == T_G on {agree / max(both_fin, 1):.1%} of them; "
        f"median T_1 - T_G = {np.median(gaps) if gaps else float('nan'):.0f} steps, mean {np.mean(gaps) if gaps else float('nan'):.1f})")
    log(f"finite under Dijkstra but DISCONNECTED under one-stitch: {t1_disc_tg_fin / n_pairs:.2%}")
    log(f"disconnected under both: {both_disc / n_pairs:.2%}; finite one-stitch but beyond-bound Dijkstra: {tg_disc_t1_fin / n_pairs:.2%}")
    for kind, name in ((0, "same-episode fwd"), (1, "same-episode bwd"), (2, "cross-episode")):
        m = np.isin(L["goal"], goals) & (L["kind"] == kind)
        log(f"  bank composition among these goals: {name}: {m.sum()} pairs, {np.mean(L['tg'][m] >= 0):.2%} finite under Dijkstra")


if __name__ == "__main__":
    main()
