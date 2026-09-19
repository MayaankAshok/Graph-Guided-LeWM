"""Training-supported directions, held-out routes, and graph goal attachments.

Offline diagnostic, not a headline planner evaluation. Reuses frozen seed-specific
TDR embeddings and the existing radius graph; never adds geometric edges.
Example: python scripts/gas_mpc_support_audit.py --device cuda
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

# Each route worker owns a single CPU thread; CUDA projection runs separately.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.quantile(x, .1)), "p90": float(np.quantile(x, .9))}


def project(psi, centers, out, device, batch):
    """Exact nearest final cluster center, rather than old clustering membership."""
    signature = hashlib.sha256(centers.tobytes()).hexdigest()
    meta = {"centers_sha256": signature, "shape": list(psi.shape)}
    if (out / "projection.json").exists():
        if json.loads((out / "projection.json").read_text()) != meta:
            raise ValueError("Projection cache belongs to different assets; choose another --out")
        return np.load(out / "nearest_node.npy", mmap_mode="r"), np.load(out / "nearest_distance.npy", mmap_mode="r")
    import torch
    c = torch.as_tensor(centers, device=device)
    c2 = (c * c).sum(1)
    node = np.lib.format.open_memmap(out / "nearest_node.npy", mode="w+", dtype="int32", shape=(len(psi),))
    distance = np.lib.format.open_memmap(out / "nearest_distance.npy", mode="w+", dtype="float32", shape=(len(psi),))
    for lo in range(0, len(psi), batch):
        x = torch.as_tensor(np.array(psi[lo:lo + batch]), device=device)
        d2 = (x * x).sum(1)[:, None] + c2[None] - 2 * x @ c.T
        val, idx = d2.min(1)
        node[lo:lo + len(x)] = idx.cpu().numpy()
        distance[lo:lo + len(x)] = val.clamp_min(0).sqrt().cpu().numpy()
        if lo % (batch * 100) == 0:
            print(f"projection {lo}/{len(psi)}", flush=True)
    node.flush()
    distance.flush()
    (out / "projection.json").write_text(json.dumps(meta))
    return node, distance


def edge_indices(keys, query):
    i = np.searchsorted(keys, query)
    valid = i < len(keys)
    valid[valid] &= keys[i[valid]] == query[valid]
    return i, valid


def directed_support(graph, nodes, ep_id, held, horizons):
    """An arc is supported iff an observed TRAIN forward pair uses this radius edge.

    Horizon k accumulates all frame lags 1..k; both directions can survive.
    Counts describe occupancy transitions, not proof of medoid controllability.
    """
    n = graph.shape[0]
    source = np.repeat(np.arange(n), np.diff(graph.indptr))
    keys = source.astype(np.int64) * n + graph.indices
    assert np.all(keys[1:] > keys[:-1])
    counts = np.zeros(len(keys), np.int64)
    train = ~np.isin(ep_id, held)
    graphs, summaries = {}, {}
    episode_events = []
    for lag in range(1, max(horizons) + 1):
        valid = train[:-lag] & (ep_id[:-lag] == ep_id[lag:]) & (nodes[:-lag] != nodes[lag:])
        rows = np.flatnonzero(valid)
        ix, supported = edge_indices(keys, nodes[rows].astype(np.int64) * n + nodes[rows + lag])
        ix, rows = ix[supported], rows[supported]
        counts += np.bincount(ix, minlength=len(keys))
        episode_events.append(np.unique(ix.astype(np.int64) * (ep_id.max() + 1) + ep_id[rows]))
        if lag in horizons:
            mask = counts > 0
            g = csr_matrix((graph.data[mask], (source[mask], graph.indices[mask])), shape=graph.shape)
            graphs[str(lag)] = g
            reverse, ok = edge_indices(keys, graph.indices.astype(np.int64) * n + source)
            assert ok.all()
            mutual = mask & mask[reverse]
            weak, weak_labels = connected_components(g, directed=True, connection="weak")
            strong, strong_labels = connected_components(g, directed=True, connection="strong")
            summaries[str(lag)] = {
                "arcs": int(mask.sum()), "arc_reduction_fraction": float(1 - mask.mean()),
                "undirected_pairs_with_any_direction": int((mask | mask[reverse]).sum() // 2),
                "bidirectional_pairs": int(mutual.sum() // 2),
                "one_way_pairs": int(mask.sum() - mutual.sum()),
                "weak_components": int(weak), "strong_components": int(strong),
                "largest_weak_component_nodes": int(np.bincount(weak_labels).max()),
                "largest_strong_component_nodes": int(np.bincount(strong_labels).max()),
                "observed_pair_counts_per_retained_arc": stats(counts[mask]),
            }
            print(f"training-supported directions at <= {lag} steps: {summaries[str(lag)]}", flush=True)
    unique = np.unique(np.concatenate(episode_events))
    ep_counts = np.bincount(unique // (ep_id.max() + 1), minlength=len(keys))
    mask = ep_counts >= 2
    robust_name = f"{max(horizons)}_min2episodes"
    graphs[robust_name] = csr_matrix((graph.data[mask], (source[mask], graph.indices[mask])), shape=graph.shape)
    summaries[robust_name] = {"arcs": int(mask.sum()), "arc_reduction_fraction": float(1 - mask.mean()),
                                          "distinct_training_episodes_per_retained_arc": stats(ep_counts[mask])}
    return graphs, summaries, keys, source, counts, ep_counts


def goal_tree(graph, centers, goal, radius):
    """Temporary directed arcs center -> goal; reverse search finds predecessors.

    Attachment arcs remain GEOMETRIC and are audited separately by the oracle.
    """
    distances = np.linalg.norm(centers - goal, axis=1)
    attach_radius = max(radius, 1.2 * float(distances.min()))
    attachments = np.flatnonzero(distances <= attach_radius)
    n = len(centers)
    coo = graph.tocoo()
    extended = csr_matrix((np.r_[coo.data, distances[attachments]],
                           (np.r_[coo.row, attachments], np.r_[coo.col, np.full(len(attachments), n)])), shape=(n + 1, n + 1))
    dist, successor = dijkstra(extended.T.tocsr(), directed=True, indices=n, return_predecessors=True)
    return dist[:n], successor[:n], attachments, distances, attach_radius


def route_for_start(dist, successor, centers, point, radius):
    distance = np.linalg.norm(centers - point, axis=1)
    nearby = np.flatnonzero(distance <= radius)
    if not len(nearby):
        return None, True, None
    costs = distance[nearby] + dist[nearby]
    if not np.isfinite(costs).any():
        return None, False, None
    node = int(nearby[np.argmin(costs)])
    cost = float(costs.min())
    route = []
    while node < len(centers):
        route.append(node)
        node = int(successor[node])
        if node < 0:
            raise AssertionError("Finite route has no successor")
        if len(route) > len(centers):
            raise AssertionError("Cycle in Dijkstra predecessors")
    return route, False, cost


def polyline_distance(points, line):
    if not len(points):
        return np.empty(0)
    vectors = np.diff(line, axis=0)
    norm = (vectors * vectors).sum(1)
    keep = norm > 1e-12
    if not keep.any():
        return np.linalg.norm(points - line[0], axis=1)
    vectors, starts, norm = vectors[keep], line[:-1][keep], norm[keep]
    delta = points[:, None] - starts[None]
    fraction = np.clip((delta * vectors[None]).sum(-1) / norm[None], 0, 1)
    return np.linalg.norm(delta - fraction[..., None] * vectors[None], axis=-1).min(1)


def self_test():
    g = csr_matrix(([2., 3.], ([0, 1], [1, 2])), shape=(3, 3))
    centers = np.array([[0., 0.], [2., 0.], [5., 0.]])
    dist, successor, _, _, _ = goal_tree(g, centers, centers[2], .1)
    route, fallback, cost = route_for_start(dist, successor, centers, centers[0], .1)
    assert route == [0, 1, 2] and cost == 5 and not fallback
    reverse, *_ = goal_tree(g, centers, centers[0], .1)
    assert np.isinf(reverse[2]), "Direction was accidentally reversed"
    assert np.allclose(polyline_distance(np.array([[1., 1.]]), centers[:2]), [1.])
    keys = np.array([1, 5, 8])
    ix, valid = edge_indices(keys, np.array([1, 4, 8, 9]))
    assert valid.tolist() == [True, False, True, False]
    # Episode boundary and held-out transitions must never support an arc.
    tiny = csr_matrix(([1., 1.], ([0, 1], [1, 0])), shape=(2, 2))
    gs, _, _, _, _, _ = directed_support(tiny, np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), [1], [1])
    assert gs['1'][0, 1] == 1 and gs['1'][1, 0] == 0



_AUDIT_CONTEXT = None


def analyze_window(window):
    psi, nodes, nearest, centers, graphs, asset, state, radius = _AUDIT_CONTEXT
    queries = []
    start, goal, gap = window['start_row'], window['goal_row'], window['gap']
    all_rows = np.arange(start, goal)  # exact goal excluded
    trace_rows = np.r_[np.arange(start, goal, 5), goal]
    trace = np.array(psi[trace_rows])
    goal_nodes = np.flatnonzero(np.linalg.norm(centers - psi[goal], axis=1) <= max(radius, 1.2 * float(nearest[goal])))
    item = {**window, 'nearest_goal_node': int(nodes[goal]), 'goal_attachment_count': len(goal_nodes),
            'nearest_goal_node_seen_before_goal': bool((nodes[all_rows] == nodes[goal]).any()),
            'any_goal_attachment_seen_before_goal': bool(np.isin(nodes[all_rows], goal_nodes).any()),
            'nearest_goal_node_seen_at_least_5steps_before_goal': bool((nodes[start:max(start, goal - 4)] == nodes[goal]).any()),
            'graphs': {}}
    # First entrance of the nearest goal cluster, counted as rows before goal.
    visits = np.flatnonzero(nodes[all_rows] == nodes[goal])
    item['nearest_goal_node_first_visit_steps_before_goal'] = int(gap - visits[0]) if len(visits) else None
    for name, g in graphs.items():
        dist, successor, attachments, goal_distance, expanded = goal_tree(g, centers, psi[goal], radius)
        route, no_attach, cost = route_for_start(dist, successor, centers, psi[start], radius)
        strict = dijkstra(g, directed=True, indices=int(nodes[start]), limit=np.inf)[int(nodes[goal])]
        metrics = {'finite_route': route is not None, 'start_attachment_absent': no_attach,
                   'goal_attachment_expanded': expanded > radius + 1e-6,
                   'strict_nearest_nodes_finite': bool(np.isfinite(strict))}
        if route is not None:
            final = route[-1]
            line = np.vstack([psi[start], centers[route], psi[goal]])
            interior = polyline_distance(trace[1:-1], line)
            metrics.update({'route_tdr_length': cost, 'route_node_count': len(route), 'final_node': final,
                'interior_distance_to_path': stats(interior),
                'interior_fraction_within_half_radius': float((interior <= radius / 2).mean()) if len(interior) else None,
                'final_node_seen_before_goal': bool((nodes[all_rows] == final).any()),
                'final_node_seen_at_least_5steps_before_goal': bool((nodes[start:max(start, goal - 4)] == final).any()),
                'minimum_actual_trajectory_tdr_to_final_center': float(np.linalg.norm(psi[start:goal + 1] - centers[final], axis=1).min()),
                'final_center_to_goal_tdr': float(goal_distance[final]),
                'final_medoid_to_goal_tdr': float(np.linalg.norm(psi[asset['node_medoid_row'][final]] - psi[goal]))})
            if window['split'] == 'heldout' and name in ['undirected', '12']:
                queries.append({**window, 'graph': name, 'final_node': final,
                    'node_medoid_row': int(asset['node_medoid_row'][final]),
                    'node_state': asset['node_state'][final].tolist(), 'goal_state': state[goal].tolist(),
                    'node_center_to_goal_tdr': float(goal_distance[final]),
                    'node_medoid_to_goal_tdr': metrics['final_medoid_to_goal_tdr']})
        item['graphs'][name] = metrics
    return item, queries


def main(args):
    self_test()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.bank / "cache_full.npz") as data:
        offsets, lengths = data['ep_offset'], data['ep_len']
        state = data['state']
    psi = np.load(args.bank / f"psi_full_s{args.asset_seed}.npy", mmap_mode='r')
    with args.graph.open('rb') as f:
        asset = pickle.load(f)
    assert asset['seed'] == args.asset_seed, 'TDR and graph asset seeds differ'
    original = asset['graph'].tocsr()
    original.sort_indices()
    centers = asset['centers'].astype(np.float32)
    radius = float(asset['h_td'])
    order = np.random.default_rng(args.split_seed).permutation(len(lengths))
    held = np.sort(order[:int(.02 * len(lengths))])
    train = order[len(held):]
    ep_id = np.repeat(np.arange(len(lengths)), lengths)
    assert np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])])
    assert not np.isin(ep_id[asset['kept_rows']], held).any()
    assert int(lengths[held].sum()) == len(psi) - asset['stats']['n_train_states']
    rng = np.random.default_rng(args.query_seed)
    # Match the full episode-length distribution without looking at geometry.
    available = list(train)
    control = []
    for e in held:
        pool = np.asarray(available)
        delta = np.abs(lengths[pool] - lengths[e])
        chosen = int(rng.choice(pool[delta == delta.min()]))
        control.append(chosen)
        available.remove(chosen)
    if args.projection_only:
        project(psi, centers, args.out, args.device, args.batch)
        return
    if not (args.out / 'projection.json').exists():
        # Release the CUDA runtime's host-memory reservation before sparse routes.
        # This matters on the 16GB Windows research machine.
        import subprocess
        import sys
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--projection-only'], check=True)
    nodes, nearest = project(psi, centers, args.out, args.device, args.batch)
    graphs, graph_stats, keys, sources, counts, ep_counts = directed_support(original, nodes, ep_id, held, [1, 5, 12])
    graphs = {'undirected': original, **graphs}
    from scipy.sparse import save_npz
    for name, g in graphs.items():
        save_npz(args.out / f"graph_{name}.npz", g)
    np.savez(args.out / 'direction_support.npz', source=sources, target=original.indices,
             weight=original.data, count_le12=counts, episodes_le12=ep_counts)
    result = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'heldout_episodes': held.tolist(), 'train_control_episodes': control,
              'heldout_frames': int(lengths[held].sum()), 'original_nodes': len(centers),
              'original_undirected_edges': original.nnz // 2, 'original_arcs': original.nnz,
              'graph_directions': graph_stats, 'coverage': {}, 'windows_summary': {}, 'arbitrary_summary': {},
              'notes': ['Held out from this TDR and graph; not necessarily encoder pretraining.',
                        'Forward direction support uses only training episodes and exact nearest final centers.',
                        'Geometric query attachments remain unverified until simulator oracle audit.',
                        'No all-node start fallback; infinity and attachment absence are reported separately.',
                        'Strict endpoint distances attach only to each endpoint nearest center.',
                        'Node traversal is measured before the exact goal frame to avoid a trivial hit.',
                        'Unsuccessful finite oracle search does not certify physical unreachability.']}
    windows = []
    for split, episodes in [('heldout', held), ('train_control', control)]:
        rows = np.concatenate([np.arange(offsets[e], offsets[e] + lengths[e]) for e in episodes])
        transition = []
        for e in episodes:
            r = np.arange(offsets[e], offsets[e] + lengths[e] - 5)
            changed = nodes[r] != nodes[r + 5]
            _, valid = edge_indices(keys, nodes[r[changed]].astype(np.int64) * len(centers) + nodes[r[changed] + 5])
            transition.extend(valid.tolist())
        result['coverage'][split] = {'episodes': len(episodes), 'frames': len(rows),
            'episode_lengths': stats(lengths[episodes]), 'nearest_center_tdr': stats(nearest[rows]),
            'within_half_radius': float((nearest[rows] <= radius / 2).mean()),
            'within_radius': float((nearest[rows] <= radius).mean()),
            'changed_node_5step_transitions_on_original_edges': float(np.mean(transition))}
        for gap in args.goal_lengths:
            for e in episodes:
                max_start = int(lengths[e]) - gap
                if max_start <= 0:
                    continue
                starts = rng.choice(max_start, size=min(args.windows_per_episode, max_start), replace=False)
                for t in starts:
                    start, goal = int(offsets[e] + t), int(offsets[e] + t + gap)
                    windows.append({'split': split, 'episode': int(e), 'gap': gap, 'start_row': start, 'goal_row': goal})
    records, oracle_queries = [], []
    global _AUDIT_CONTEXT
    _AUDIT_CONTEXT = (psi, nodes, nearest, centers, graphs, asset, state, radius)
    from multiprocessing import get_context
    from contextlib import nullcontext
    context = get_context('fork').Pool(args.workers) if args.workers > 1 else nullcontext(None)
    with context as pool:
        iterator = pool.imap(analyze_window, windows, chunksize=4) if pool else map(analyze_window, windows)
        for j, (item, queries) in enumerate(iterator):
            oracle_queries.extend(queries)
            records.append(item)
            if j % 100 == 0:
                print(f"routes {j}/{len(windows)}", flush=True)
                (args.out / 'progress.json').write_text(json.dumps({'completed': j, 'total': len(windows)}))
    for split in ['heldout', 'train_control']:
        for gap in args.goal_lengths:
            selected = [r for r in records if r['split'] == split and r['gap'] == gap]
            if not selected:
                continue
            entry = {'windows': len(selected), 'episodes': len(set(r['episode'] for r in selected)),
                'nearest_goal_node_seen_before_goal': float(np.mean([r['nearest_goal_node_seen_before_goal'] for r in selected])),
                'nearest_goal_node_seen_at_least_5steps_before_goal': float(np.mean([r['nearest_goal_node_seen_at_least_5steps_before_goal'] for r in selected])),
                'any_goal_attachment_seen_before_goal': float(np.mean([r['any_goal_attachment_seen_before_goal'] for r in selected])),
                'nearest_goal_node_first_visit_steps_before_goal': stats([r['nearest_goal_node_first_visit_steps_before_goal'] for r in selected if r['nearest_goal_node_first_visit_steps_before_goal'] is not None]),
                'graphs': {}}
            for name in graphs:
                m = [r['graphs'][name] for r in selected]
                finite = [x for x in m if x['finite_route']]
                entry['graphs'][name] = {'finite_route_fraction': float(np.mean([x['finite_route'] for x in m])),
                    'strict_nearest_nodes_finite_fraction': float(np.mean([x['strict_nearest_nodes_finite'] for x in m])),
                    'start_attachment_absent_fraction': float(np.mean([x['start_attachment_absent'] for x in m])),
                    **{k: stats([x[k] for x in finite if x[k] is not None]) for k in ['route_node_count', 'route_tdr_length', 'interior_fraction_within_half_radius', 'final_node_seen_before_goal', 'final_node_seen_at_least_5steps_before_goal', 'final_center_to_goal_tdr', 'final_medoid_to_goal_tdr']}}
            result['windows_summary'][f'{split}|{gap}'] = entry
    # Arbitrary pairs: cross held-out episodes and reverse pairs. There is no recorded
    # goal-conditioned trajectory for these, so do not label its traversal a failure.
    arbitrary = []
    for kind in ['cross_episode', 'backward_same_episode']:
        for _ in range(args.arbitrary_pairs):
            e = int(rng.choice(held))
            if kind == 'cross_episode':
                other = int(rng.choice(held[held != e]))
                start = int(offsets[e] + rng.integers(lengths[e]))
                goal = int(offsets[other] + rng.integers(lengths[other]))
            else:
                a, b = np.sort(rng.choice(int(lengths[e]), 2, replace=False))
                start, goal = int(offsets[e] + b), int(offsets[e] + a)
            record = {'kind': kind, 'start_row': start, 'goal_row': goal, 'graphs': {}}
            for name, g in graphs.items():
                dist, successor, _, goal_distance, _ = goal_tree(g, centers, psi[goal], radius)
                route, missing, cost = route_for_start(dist, successor, centers, psi[start], radius)
                strict = dijkstra(g, directed=True, indices=int(nodes[start]))[int(nodes[goal])]
                record['graphs'][name] = {'finite_route': route is not None, 'strict_nearest_nodes_finite': bool(np.isfinite(strict)),
                    'final_node': route[-1] if route else None, 'route_tdr_length': cost}
                if route and name in ['undirected', '12']:
                    final = route[-1]
                    oracle_queries.append({'split': 'arbitrary', 'kind': kind, 'start_row': start, 'goal_row': goal,
                        'graph': name, 'final_node': final, 'node_medoid_row': int(asset['node_medoid_row'][final]),
                        'node_state': asset['node_state'][final].tolist(), 'goal_state': state[goal].tolist(),
                        'node_center_to_goal_tdr': float(goal_distance[final]),
                        'node_medoid_to_goal_tdr': float(np.linalg.norm(psi[asset['node_medoid_row'][final]] - psi[goal]))})
            arbitrary.append(record)
        chosen = [r for r in arbitrary if r['kind'] == kind]
        result['arbitrary_summary'][kind] = {name: {'pairs': len(chosen), 'finite_route_fraction': float(np.mean([r['graphs'][name]['finite_route'] for r in chosen])),
            'strict_nearest_nodes_finite_fraction': float(np.mean([r['graphs'][name]['strict_nearest_nodes_finite'] for r in chosen]))} for name in graphs}
    result['elapsed_seconds'] = time.time() - started
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    (args.out / 'windows.json').write_text(json.dumps(records))
    (args.out / 'arbitrary_pairs.json').write_text(json.dumps(arbitrary))
    (args.out / 'oracle_queries.json').write_text(json.dumps(oracle_queries))
    print(f"done: {len(records)} windows, {len(arbitrary)} arbitrary pairs, {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, default=ROOT / 'outputs/pusht')
    p.add_argument('--graph', type=Path, default=ROOT / 'outputs/pusht/graph_full_s0_htd8_te0.9.pkl')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/pusht/diagnostics/heldout_support_20260918')
    p.add_argument('--asset-seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--query-seed', type=int, default=20260918)
    p.add_argument('--goal-lengths', type=int, nargs='+', default=[12, 25, 50, 100])
    p.add_argument('--windows-per-episode', type=int, default=2)
    p.add_argument('--arbitrary-pairs', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--projection-only', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--workers', type=int, default=8 if os.name == 'posix' else 1)
    p.add_argument('--batch', type=int, default=128)
    main(p.parse_args())
