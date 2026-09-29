"""Execute graph-to-goal physics queries and blind rendered encoder inversion.

python scripts/diagnostics/gas_mpc_oracle_audit.py --mode reachability --workers 4
python scripts/diagnostics/gas_mpc_oracle_audit.py --mode inversion
Imports of the model and simulator are deferred until the selected stage.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.planning_oracles import (make_oracle_env, reset_visible_state, pose_error,
                                      oracle_reachability, oracle_latent_to_state, make_encoder)

ROOT = Path(__file__).resolve().parents[2]


def read_npz_rows(path, column, rows):
    """Read selected rows without allocating the full 1.8GB embedding column."""
    rows = np.asarray(rows, dtype=np.int64)
    with zipfile.ZipFile(path) as archive, archive.open(column + '.npy') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        assert not fortran and len(shape) == 2
        stride = shape[1] * dtype.itemsize
        output = np.empty((len(rows), shape[1]), dtype=dtype)
        unique = np.unique(rows)
        current = 0
        for row in unique:
            skip = (int(row) - current) * stride
            while skip:
                chunk = f.read(min(skip, 4 * 1024 * 1024))
                if not chunk:
                    raise EOFError('Truncated cache')
                skip -= len(chunk)
            raw = f.read(stride)
            output[rows == row] = np.frombuffer(raw, dtype=dtype)
            current = int(row) + 1
    return output


def reach_job(job):
    query, config = job
    env = make_oracle_env()
    started = time.time()
    initial = reset_visible_state(env, query['node_state'], query['goal_state'], config['seed'])
    position, angle, _ = pose_error(initial, query['goal_state'])
    answer = {**query, 'initial_position_error_px': position, 'initial_angle_error_rad': angle, 'searches': {}}
    for horizon in config['horizons']:
        result = oracle_reachability(query['node_state'], query['goal_state'], horizon,
            seed=config['seed'], population=config['population'], iterations=config['iterations'],
            restarts=config['restarts'], env=env)
        answer['searches'][str(horizon)] = result
    # A witness found in a shorter search also proves reachability at larger caps.
    # Retain every independent search, including failures, for inspection.
    answer['independent_searches'] = dict(answer['searches'])
    for horizon in config['horizons']:
        witnesses = [r for h, r in answer['independent_searches'].items()
                     if int(h) <= horizon and r['status'] == 'reachable']
        if witnesses:
            best = min(witnesses, key=lambda r: r['first_hit'])
            answer['searches'][str(horizon)] = {**best, 'horizon_env_steps': horizon,
                                               'includes_witnesses_from_shorter_caps': True}
    env.close()
    answer['elapsed_seconds'] = time.time() - started
    return answer


def reachability(args):
    queries = json.loads((args.audit / 'oracle_queries.json').read_text())
    # Compare the path-selected final attachment with the closest goal cluster.
    # This control isolates the attachment choice from basic graph coverage.
    import pickle
    with args.graph.open('rb') as f:
        asset = pickle.load(f)
    nodes = np.load(args.audit / 'nearest_node.npy', mmap_mode='r')
    psi = np.load(args.bank / 'psi_full_s0.npy', mmap_mode='r')
    nearest_queries = []
    for query in queries:
        if query['graph'] != 'undirected':
            continue
        node = int(nodes[query['goal_row']])
        row = int(asset['node_medoid_row'][node])
        nearest_queries.append({**query, 'graph': 'nearest_goal_node', 'final_node': node,
            'node_medoid_row': row, 'node_state': asset['node_state'][node].tolist(),
            'node_center_to_goal_tdr': float(np.linalg.norm(asset['centers'][node] - psi[query['goal_row']])),
            'node_medoid_to_goal_tdr': float(np.linalg.norm(psi[row] - psi[query['goal_row']]))})
    queries.extend(nearest_queries)
    rng = np.random.default_rng(args.seed)
    groups = {}
    for query in queries:
        key = (query['split'], query['graph'], str(query.get('gap', query.get('kind'))))
        groups.setdefault(key, []).append(query)
    selected = []
    cells = sorted(set((key[0], key[2]) for key in groups))
    for split, label in cells:
        maps = {graph: {(q['start_row'], q['goal_row']): q for q in groups[(split, graph, label)]}
                for graph in ['undirected', '12', 'nearest_goal_node']}
        # Hold the goal/start identity fixed across all three attachment choices.
        common = sorted(set.intersection(*(set(m) for m in maps.values())))
        for index in rng.choice(len(common), min(args.per_group, len(common)), replace=False):
            identity = common[index]
            for graph, mapping in maps.items():
                selected.append({**mapping[identity], 'group': '|'.join((split, graph, label))})
    config = {'seed': args.seed, 'horizons': args.horizons, 'population': args.population,
              'iterations': args.iterations, 'restarts': args.restarts, 'per_group': args.per_group,
              'sampling': 'Matched start/goal identities across three choices; finite routes in both graphs required.'}
    partial = args.out / 'reachability_partial.json'
    completed = []
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved['config'] != config or saved['selected_queries'] != selected:
            raise ValueError('Changed oracle configuration; choose another --out')
        completed = saved['results']
    done_keys = {(q['group'], q['start_row'], q['goal_row'], q['final_node']) for q in completed}
    pending = [q for q in selected if (q['group'], q['start_row'], q['goal_row'], q['final_node']) not in done_keys]
    def save():
        partial.write_text(json.dumps({'config': config, 'selected_queries': selected, 'results': completed}))
    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(q, config) for q in pending]
        for result in pool.map(reach_job, jobs):
            completed.append(result)
            save()
            print(f"oracle {len(completed)}/{len(selected)}: {result['group']} "
                  f"{[(h, r['status'], r['first_hit']) for h, r in result['searches'].items()]}", flush=True)
    summary = {'config': config, 'queries': len(completed), 'groups': {},
        'scope': 'Stratified finite simulator-search pilot of route-selected final MEDOID to held-out goal; not all goal attachments.',
        'h_td_tdr_units': 8., 'h_td_calibrated_env_steps': 12,
        'state_caveat': 'Exact visible pose and agent velocity restored; missing block velocities set to zero.',
        'negative_result': 'unknown: no witness found within this search budget, not certified unreachable.'}
    for key in sorted(groups):
        name = '|'.join(key)
        chosen = [r for r in completed if r['group'] == name]
        summary['groups'][name] = {'queries': len(chosen), 'initially_successful': sum(r['searches'][str(args.horizons[0])]['first_hit'] == 0 for r in chosen), 'horizons': {}}
        for horizon in args.horizons:
            found = [r['searches'][str(horizon)] for r in chosen]
            successes = [r for r in found if r['status'] == 'reachable']
            summary['groups'][name]['horizons'][str(horizon)] = {
                'reachable_witnesses': len(successes), 'unknown': len(found) - len(successes),
                'witness_fraction': len(successes) / len(found) if found else None,
                'shortest_found_hit_steps': [r['first_hit'] for r in successes]}
    (args.out / 'reachability_summary.json').write_text(json.dumps(summary, indent=2))


def inversion(args):
    import torch
    import pickle
    from common.lewm_loader import load_lewm
    with np.load(args.bank / 'cache_full.npz') as cache:
        offsets, lengths, states = cache['ep_offset'], cache['ep_len'], cache['state']
    with args.graph.open('rb') as f:
        graph = pickle.load(f)
    summary_file = args.audit / 'summary.json'
    held = (np.array(json.loads(summary_file.read_text())['heldout_episodes']) if summary_file.exists()
            else np.sort(np.random.default_rng(0).permutation(len(lengths))[:int(.02 * len(lengths))]))
    train = np.setdiff1d(np.arange(len(lengths)), held)
    rng = np.random.default_rng(args.seed)
    health_rows = [int(offsets[e] + rng.integers(lengths[e])) for e in rng.choice(train, 4, replace=False)]
    rows = [int(offsets[e] + rng.integers(lengths[e])) for e in rng.choice(held, args.inversion_targets, replace=False)]
    target_z = read_npz_rows(args.bank / 'cache_full.npz', 'z', health_rows + rows)
    model = load_lewm(args.ckpt, device=args.device)
    encode = make_encoder(model, args.device, batch=16)
    health = {}
    for with_target in [True, False]:
        env = make_oracle_env(rendering=True, with_target=with_target)
        images = []
        for row in health_rows:
            reset_visible_state(env, states[row], seed=args.seed)
            images.append(env.render())
        rendered = encode(images)
        health[str(with_target)] = {'mean_squared_latent_residual': float(np.mean(np.sum((rendered - target_z[:4]) ** 2, axis=1))),
                                   'squared_latent_residuals': np.sum((rendered - target_z[:4]) ** 2, axis=1).tolist()}
        env.close()
    with_target = health['True']['mean_squared_latent_residual'] <= health['False']['mean_squared_latent_residual']
    env = make_oracle_env(rendering=True, with_target=with_target)
    # Check that velocities leave the encoder input unchanged.
    state = states[health_rows[0]].copy()
    reset_visible_state(env, state, seed=args.seed)
    image = env.render()
    state[5:7] += 100
    reset_visible_state(env, state, seed=args.seed)
    velocity_pixels_identical = bool(np.array_equal(image, env.render()))
    node_z = graph['node_z']
    offdiag = target_z[:4] @ target_z[:4].T
    norm = np.linalg.norm(target_z[:4], axis=1)
    cosine = offdiag / (norm[:, None] * norm[None])
    result = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'render_health_training_rows': health_rows, 'render_health': health, 'with_target_selected': with_target,
              'velocity_pixels_identical': velocity_pixels_identical,
              'off_diagonal_cosine_small_sample': float(cosine[~np.eye(4, dtype=bool)].mean()),
              'scope': 'Blind held-out cached-z inversion; initial states retrieved from training graph medoids. Frozen encoder only.',
              'targets': []}
    for i, row in enumerate(rows):
        z = target_z[i + 4]
        nearest = np.argsort(np.sum((node_z - z) ** 2, axis=1))[:4]
        initial = graph['node_state'][nearest]
        # Baseline evaluates the SAME renderer on retrieval states, for a fair cost comparison.
        images = []
        for s in initial:
            reset_visible_state(env, s, seed=args.seed)
            images.append(env.render())
        baseline_costs = np.sum((encode(images) - z) ** 2, axis=1)
        best = int(baseline_costs.argmin())
        answer = oracle_latent_to_state(z, encode, initial, seed=args.seed + i, population=args.population,
            iterations=args.inversion_iterations, env=env)
        reset_visible_state(env, states[row], seed=args.seed)
        actual_residual = float(np.sum((encode([env.render()])[0] - z) ** 2))
        position, angle, _ = pose_error(answer['state'], states[row])
        baseline_position, baseline_angle, _ = pose_error(initial[best], states[row])
        target = {'row': row, 'episode': int(np.searchsorted(offsets, row, side='right') - 1),
            'privileged_state_for_scoring_only': states[row].tolist(), 'retrieved_node_ids': nearest.tolist(),
            'retrieval_render_squared_residual': float(baseline_costs.min()),
            'true_state_render_squared_residual': actual_residual,
            'position_error_px': position, 'angle_error_rad': angle,
            'retrieval_position_error_px': baseline_position, 'retrieval_angle_error_rad': baseline_angle,
            'success_predicate_pose_reconstruction': position < 20 and angle < np.pi / 9, **answer}
        result['targets'].append(target)
        (args.out / 'inversion_results.json').write_text(json.dumps(result, indent=2))
        print(f"inversion {i + 1}/{len(rows)}: L2 {baseline_costs.min():.3f} -> {answer['latent_squared_residual']:.3f}, "
              f"pose {position:.1f}px {angle:.3f}rad", flush=True)
    env.close()
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['reachability', 'inversion'], required=True)
    p.add_argument('--bank', type=Path, default=ROOT / 'outputs/pusht')
    p.add_argument('--audit', type=Path, default=ROOT / 'outputs/pusht/diagnostics/heldout_support_20260918')
    p.add_argument('--graph', type=Path, default=ROOT / 'outputs/pusht/graph_full_s0_htd8_te0.9.pkl')
    p.add_argument('--ckpt', type=Path, default=ROOT / 'data/checkpoints/models--quentinll--lewm-pusht')
    p.add_argument('--out', type=Path)
    p.add_argument('--seed', type=int, default=20260919)
    p.add_argument('--per-group', type=int, default=8)
    p.add_argument('--horizons', nargs='+', type=int, default=[12, 25])
    p.add_argument('--population', type=int, default=48)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--restarts', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--inversion-targets', type=int, default=8)
    p.add_argument('--inversion-iterations', type=int, default=10)
    args = p.parse_args()
    if args.out is None:
        args.out = args.audit / 'oracles'
        if args.mode == 'reachability':
            args.out = args.out / 'paired'
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'reachability':
        reachability(args)
    else:
        inversion(args)
