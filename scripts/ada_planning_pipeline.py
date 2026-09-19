"""Resumable training-only planning pipeline; dry-run unless --run is supplied.

Run on an Ada compute node with staged HDF5 datasets and pretrained LeWM checkpoints:
  /home2/mayaank.ashok/.venv/bin/python scripts/ada_planning_pipeline.py
  /home2/mayaank.ashok/.venv/bin/python scripts/ada_planning_pipeline.py --run --gpus 4

Environments run sequentially. Preparation runs alone; critic/eval queues run two
tasks per GPU. Missing/incomplete critics restart from scratch (the trainer has
no optimizer-resume path). Superseded outputs are moved to bak before replacement.
Evaluates L2 and OUR on seeds 0..4, first 50 of each task200u pool by default.
--smoke runs the entire path with real data in a fresh pipeline_smoke scratch
directory: two-step TDRs/critics, two-task pools, one task per eval, one CEM
iteration, and five real environment steps. It cannot satisfy main-run checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SEEDS = range(5)
PROTOCOLS = ('same25', 'same50', 'same100', 'cross')
TASKS_PER_GPU = 2
SPLIT_KEYS = ('training_only', 'n_source_episodes', 'heldout_frac',
              'episode_id', 'heldout_episode_ids')


def critic_ok(ck, split, seed, min_steps=60000):
    import numpy as np
    a = ck['args']
    train, val = np.asarray(ck['train_episodes']), np.asarray(ck['val_episodes'])
    used = np.r_[train, val]
    return (ck['step'] >= min_steps and a['seed'] == seed and
            a.get('training_only') and a.get('exclude_tdr_holdout') and
            a.get('label_source') == 'tdr' and len(train) > 0 and len(val) > 0 and
            np.array_equal(ck['evaluation_episodes'], split['heldout_episode_ids']) and
            np.isin(used, split['episode_id']).all() and
            not np.intersect1d(train, val).size and
            not np.intersect1d(used, split['heldout_episode_ids']).size)


def eval_ok(r, method, proto, seed, n, tag, budget, smoke=False):
    hits = r['first_hit_step']
    successes = sum(h >= 0 for h in hits)
    return (r['method'] == tag and r['protocol'] == proto and r['seed'] == seed and
            r['n'] == n and r['tasks'] == (2 if smoke else 200) and
            r['pool'] == ('task2u' if smoke else 'task200u') and bool(r.get('smoke_test',False)) == smoke and
            r['budget'] == budget and r['mpc']['method'] == method and
            r['mpc']['graph_seed'] == seed and len(hits) == n and
            all(isinstance(h, (int, float)) and (h == -1 or 0 <= h <= budget) for h in hits) and
            r['n_success'] == successes and abs(r['success_rate'] - 100*successes/n) < 1e-6)


def available(check):
    try:
        return bool(check())
    except (OSError, KeyError, ValueError, AssertionError, EOFError, pickle.UnpicklingError):
        return False


def archive(path, out):
    """Move the named file/link itself; never follow a scratch-cache symlink."""
    if not path.exists() and not path.is_symlink():
        return
    assert path.absolute().is_relative_to(out.absolute())
    dest = out/'bak'/('pipeline_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))/path.relative_to(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    path.rename(dest)
    print(f'  archived {path} -> {dest}', flush=True)


def smoke_fixture(env,out):
    """Small real-data fixture drawn exclusively from main-training episodes."""
    import hdf5plugin  # noqa: F401
    import h5py
    import numpy as np
    from common.envs import ENV_MECHANICS
    mech=ENV_MECHANICS[env]
    source=mech.h5_path(ROOT)
    if (out/'fixture.h5').exists():
        marker=json.loads((out/'SMOKE_ONLY.json').read_text())
        assert marker['smoke_test'] and not marker['main_eligible']
        with np.load(ROOT/'outputs'/env/'cache_train.npz') as d:
            assert np.isin(marker['source_training_episodes'],d['episode_id']).all()
            assert not np.intersect1d(marker['source_training_episodes'],d['heldout_episode_ids']).size
        os.environ[mech.h5_path_env_var]=str(out/'fixture.h5')
        print(f'Reusing isolated SMOKE fixture {out}',flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    with np.load(ROOT/'outputs'/env/'cache_train.npz') as d:
        ids=d['episode_id']; held=d['heldout_episode_ids']; mean=d['act_mean']; std=d['act_std']
    with h5py.File(source,'r') as f:
        offsets=f['ep_offset'][:]; lengths=f['ep_len'][:]
        eligible=ids[lengths[ids]>=110]
        chosen=np.sort(np.random.default_rng(20260919).choice(eligible,20,replace=False))
        assert not np.intersect1d(chosen,held).size
        lens=np.minimum(lengths[chosen],128); starts=np.r_[0,np.cumsum(lens)[:-1]]
        total=int(sum(lens)); full_rows=len(f['pixels'])
        with h5py.File(out/'fixture.h5','w') as g:
            for k,v in f.attrs.items(): g.attrs[k]=v
            g.create_dataset('ep_len',data=lens); g.create_dataset('ep_offset',data=starts)
            for key,ds in f.items():
                if key in ['ep_len','ep_offset']: continue
                if ds.ndim and len(ds)==full_rows:
                    new=g.create_dataset(key,shape=(total,)+ds.shape[1:],dtype=ds.dtype,compression='lzf')
                    for i,(ep,L,start) in enumerate(zip(chosen,lens,starts)):
                        lo=int(offsets[ep]); L=int(L); start=int(start)
                        data=ds[lo:lo+L]
                        if key in ['ep_idx','episode_idx']: data=np.full_like(data,i)
                        if key=='step_idx': data=np.arange(L,dtype=ds.dtype).reshape(data.shape)
                        new[start:start+L]=data
                    for k,v in ds.attrs.items(): new.attrs[k]=v
                else:
                    data=ds[chosen] if ds.ndim and len(ds)==len(lengths) else ds[()]
                    g.create_dataset(key,data=data)
    np.savez(out/'cache_full.npz',act_mean=mean,act_std=std,episode_id=np.arange(20))
    (out/'SMOKE_ONLY.json').write_text(json.dumps(dict(smoke_test=True,main_eligible=False,
        source=str(source),source_training_episodes=chosen.tolist(),main_holdout_overlap=0)))
    os.environ[mech.h5_path_env_var]=str(out/'fixture.h5')
    print(f'SMOKE fixture {env}: {total} frames; zero main-holdout episodes; isolated at {out}',flush=True)


def environment(args):
    if args.smoke:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_offset)
        os.environ['MUJOCO_EGL_DEVICE_ID']=str(args.gpu_offset)
    import numpy as np
    import torch
    os.chdir(ROOT)
    env = args.worker_env
    out = Path(args.smoke_root)/env if args.smoke else ROOT/'outputs'/env
    pool_name = 'task2u' if args.smoke else 'task200u'
    task_count = 2 if args.smoke else 200
    heldout_frac = 0.1 if args.smoke else 0.02
    tdr_steps = 2 if args.smoke else 50000
    critic_steps = 2 if args.smoke else 60000
    os.environ.update(GAS_MPC_ENV=env, GAS_MPC_OUT=str(out), GAS_MPC_POOL='task200u',
                      GAS_MPC_TRAIN_CACHE_DIR=str(out) if args.smoke else str(Path(args.train_cache_root)/env),
                      GAS_MPC_SMOKE='1' if args.smoke else '0',
                      MUJOCO_GL='egl', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    os.environ['GAS_MPC_POOL'] = pool_name
    os.environ.pop('GAS_MPC_TDR_TAG', None)
    defaults = {'PUSHT_H5_PATH': '/ssd_scratch/mayaank.ashok/lewm_data/datasets/pusht_expert_train.h5',
                'REACHER_H5_PATH': '/ssd_scratch/mayaank.ashok/lewm_data/datasets/reacher.h5',
                'CUBE_H5_PATH': '/ssd_scratch/mayaank.ashok/lewm_data/datasets/ogbench/cube_single_expert.h5'}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    from common.envs import ENV_MECHANICS
    from common.heldout_tasks import validate_training_cache
    if args.smoke:
        assert 'pipeline_smoke' in out.parts and out != ROOT/'outputs'/env
        smoke_fixture(env,out)
    from gas_mpc_eval import method_tag, TASK_SEED, PROTOCOLS as protocols
    from omegaconf import OmegaConf
    mech = ENV_MECHANICS[env]
    h5 = mech.h5_path(ROOT)
    weights = mech.ckpt_dir(ROOT)/'weights.pt'
    print(f'\n=== {env}: {args.n} tasks/protocol; CEM and asset seeds 0..4 ===', flush=True)
    for path in [h5, weights, weights.with_name('config.json'), out/'cache_full.npz']:
        if not path.exists():
            raise FileNotFoundError(f'Prerequisite missing: {path}; stage it before running this pipeline')
    # cache_full is evaluation/normalization metadata only; never read its frame arrays.
    with np.load(out/'cache_full.npz') as d:
        assert 'act_mean' in d and 'act_std' in d
    if args.run and torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f'Requested {args.gpus} visible GPUs; found {torch.cuda.device_count()}')
    split = None
    def cache_ok():
        nonlocal split
        with np.load(out/'cache_train.npz') as d:
            split = {k:d[k] for k in SPLIT_KEYS}
            validate_training_cache(split)
            assert float(split['heldout_frac']) == heldout_frac
            assert Path(str(d['h5_path'].item())).resolve() == h5.resolve()
            assert Path(str(d['ckpt_dir'].item())).resolve() == weights.parent.resolve()
            assert len(d['ep_len']) == len(split['episode_id'])
        return True
    plan = []
    def command(argv, gpu=0, log=None, extra=None):
        child = os.environ.copy()
        child.update(extra or {})
        # EGL uses the physical device index; CUDA renumbers this single device to 0.
        child.update(CUDA_VISIBLE_DEVICES=str(args.gpu_offset+gpu),
                     MUJOCO_EGL_DEVICE_ID=str(args.gpu_offset+gpu))
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as handle:
                subprocess.run(argv, cwd=ROOT, env=child, stdout=handle, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(argv, cwd=ROOT, env=child, check=True)
    def stage(name, check, argv, stale=None, force=False):
        ok = not force and available(check)
        plan.append({'stage': name, 'action': 'SKIP' if ok else 'RUN', 'command': argv})
        print(f"{'SKIP' if ok else 'RUN ':4} {name}", flush=True)
        if not ok and args.run:
            if stale:
                archive(stale, out)
            command(argv, log=out/'logs'/'pipeline'/f'{name}.log')
            assert available(check), f'Stage output failed validation: {name}'
        return ok
    py = sys.executable
    stage('encode', cache_ok, [py, 'scripts/gas_mpc_prepare.py', 'encode','--heldout-frac',str(heldout_frac)],
          stale=out/'cache_train.npz' if (out/'cache_train.npz').is_symlink() and not (out/'cache_train.npz').exists() else None)
    def tdr_ok(s):
        assert split is not None
        ck = torch.load(out/f'tdr_full_s{s}.pt', map_location='cpu', weights_only=False)
        used = np.r_[ck['train_episode_ids'], ck['diagnostic_episode_ids']]
        return (ck['done'] and ck['step'] >= tdr_steps and ck['cfg']['seed'] == s and
                ck['cfg'].get('diagnostic_scope') == 'training_only' and
                np.array_equal(ck['excluded_episode_ids'], split['heldout_episode_ids']) and
                np.isin(used, split['episode_id']).all())
    changed_tdr = set()
    for s in SEEDS:
        # Native preparation resumes partial TDRs. Legacy completed assets must be archived first.
        path = out/f'tdr_full_s{s}.pt'
        completed = available(lambda: torch.load(path,map_location='cpu',weights_only=False)['done'])
        ok = stage(f'tdr_s{s}', lambda s=s: tdr_ok(s),
                   [py, 'scripts/gas_mpc_prepare.py', 'tdr', '--seed', str(s),
                    '--heldout-frac',str(heldout_frac),'--tdr-steps',str(tdr_steps),
                    '--tdr-batch','16' if args.smoke else '1024'],
                   stale=path if completed else None)
        if not ok: changed_tdr.add(s)
    def expected_calibration():
        assert tdr_ok(0)
        ck = torch.load(out/'tdr_full_s0.pt', map_location='cpu', weights_only=False)
        assert ck['cfg']['diagnostic_scope'] == 'training_only'
        med = {int(k):float(v) for k,v in ck['history'][-1]['median_by_gap'].items()}
        gaps = sorted(med)
        h = 8.0 if env == 'pusht' else float(np.interp(12, gaps, [med[g] for g in gaps]))
        return dict(scope='training_only', h_td=round(h,2), lookahead=round(med[25],2),
                    htd_steps=12, median_by_gap=med, xneg_min_dist=0.0)
    def calib_ok():
        got = json.loads((out/'calib.json').read_text()); want = expected_calibration()
        return all(got.get(k) == want[k] for k in ('scope','h_td','lookahead','htd_steps'))
    calib_code = """import json, numpy as np, torch
from pathlib import Path
from gas_mpc_prepare import OUT, ENV
c=torch.load(OUT/'tdr_full_s0.pt',map_location='cpu',weights_only=False)
assert c['cfg']['diagnostic_scope']=='training_only'
m={int(k):float(v) for k,v in c['history'][-1]['median_by_gap'].items()}; g=sorted(m)
h=8.0 if ENV=='pusht' else float(np.interp(12,g,[m[x] for x in g]))
(OUT/'calib.json').write_text(json.dumps(dict(scope='training_only',h_td=round(h,2),lookahead=round(m[25],2),htd_steps=12,median_by_gap=m,xneg_min_dist=0.0),indent=2))
"""
    snippet = lambda code: [py, '-c', "import sys; sys.path.insert(0,'scripts');\n"+code]
    stage('calibration', calib_ok, snippet(calib_code), stale=out/'calib.json')
    calibration = expected_calibration() if available(lambda: tdr_ok(0)) else None
    htd = calibration['h_td'] if calibration else None
    la = calibration['lookahead'] if calibration else None
    print(f'  OUR: h_td={htd}, lookahead=final_thresh={la}, final_metric=l2, critic_beta=1, critic_cost=et, compose=std', flush=True)
    for s in SEEDS:
        def psi_ok(s=s):
            assert split is not None and tdr_ok(s)
            with np.load(out/'cache_train.npz') as d: rows = int(d['ep_len'].sum())
            ck = torch.load(out/f'tdr_full_s{s}.pt', map_location='cpu', weights_only=False)
            return np.load(out/f'psi_train_s{s}.npy', mmap_mode='r').shape == (rows, ck['cfg']['tdr_dim'])
        psi_path = out/f'psi_train_s{s}.npy'
        changed = s in changed_tdr
        stage(f'psi_s{s}', psi_ok,
              snippet(f'from gas_mpc_prepare import ensure_psi; ensure_psi({s})'), stale=psi_path, force=changed)
        def gap_ok(s=s):
            d = json.loads((out/f'gap_calib_s{s}.json').read_text())
            return d['scope'] == 'training_only' and len(d['gaps']) == len(d['tdr']) == len(d['l2'])
        stage(f'gap_calibration_s{s}', gap_ok,
              snippet(f'from gas_mpc_eval import gap_calibration; gap_calibration({s})'),
              stale=out/f'gap_calib_s{s}.json', force=changed)
        graph = out/f'graph_full_s{s}_htd{htd:g}_te0.9.pkl' if htd is not None else None
        def graph_ok(s=s, graph=graph):
            assert graph is not None and split is not None and tdr_ok(s)
            with graph.open('rb') as f: g = pickle.load(f)
            return (g['training_only'] and g['seed'] == s and g['h_td'] == htd and g['te_threshold'] == 0.9 and
                    np.array_equal(g['excluded_episode_ids'], split['heldout_episode_ids']))
        stage(f'graph_s{s}', graph_ok,
              [py, 'scripts/gas_mpc_prepare.py', 'graph', '--seed', str(s), '--h-td', str(htd), '--te', '0.9'],
              stale=graph, force=changed)
    def pools_ok():
        assert split is not None
        for proto in PROTOCOLS:
            suffix = 'cross_episode' if proto == 'cross' else f'same_episode_off{proto[4:]}'
            d = json.loads((out/'pairs'/f'pairs_{suffix}_{pool_name}.json').read_text())
            assert d['n'] == task_count and d['seed'] == TASK_SEED and d['nonoverlap']
            assert np.array_equal(np.sort(d['heldout_eps']), split['heldout_episode_ids'])
            assert len(d['start_ep']) == len(d['goal_ep']) == task_count
            assert np.isin(d['start_ep']+d['goal_ep'], split['heldout_episode_ids']).all()
        return True
    stage(f'{pool_name}_pools', pools_ok, [py, 'scripts/gas_mpc_make_tasks.py', '--tasks',str(task_count),'--asset-seed','0'])
    def ck_ok(s):
        assert split is not None and tdr_ok(s)
        c = torch.load(out/f'critic_s{s}_tdr_holdout/critic.pt', map_location='cpu', weights_only=False)
        assert c['args']['env'] == env
        assert Path(c['cache']).resolve() == (out/'cache_train.npz').resolve()
        return critic_ok(c, split, s,critic_steps) and all(torch.isfinite(v).all() for v in c['critic'].values() if torch.is_tensor(v))
    training = []
    critic_ready = {}
    for s in SEEDS:
        argv = [py, 'scripts/viability_train.py', '--env',env,'--cache',str(out/'cache_train.npz'),
                '--out',str(out/f'critic_s{s}_tdr_holdout'),'--seed',str(s),'--steps',str(critic_steps),
                '--log-every','200','--eval-every','2000','--label-source','tdr',
                '--tdr',str(out/f'tdr_full_s{s}.pt'),'--xneg-tdr-factor','1.25','--exclude-tdr-holdout']
        if args.smoke: argv += ['--batch-hindsight','8','--batch-cross','2','--batch-imagined','2',
                                '--batch-bellman','2','--bellman-actions','2','--imagine-blocks','1','--eval-n','16']
        ok = s not in changed_tdr and available(lambda s=s: ck_ok(s))
        critic_ready[s] = ok
        plan.append({'stage':f'critic_s{s}','action':'SKIP' if ok else 'RUN','command':argv})
        print(f"{'SKIP' if ok else 'RUN ':4} critic_s{s}", flush=True)
        if not ok: training.append((s,argv))
    worker_count = args.gpus * TASKS_PER_GPU
    def train_queue(slot):
        gpu = slot % args.gpus
        for s,argv in training[slot::worker_count]:
            archive(out/f'critic_s{s}_tdr_holdout',out)
            command(argv, gpu=gpu, log=out/'logs'/'pipeline'/f'critic_s{s}.log')
            assert ck_ok(s), f'Critic seed {s} failed holdout/completion checks'
            critic_ready[s] = True
    if args.run:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(train_queue, range(worker_count)))
    # Content-dependent evaluation tags prevent reuse after assets, task pools or code change.
    hashes = {}
    def sha(path):
        if path not in hashes:
            if not path.exists(): return 'pending'
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
            hashes[path] = h.hexdigest()
        return hashes[path]
    source_paths = [ROOT/'scripts'/k for k in ('gas_mpc_eval.py','gas_mpc_run.sh','viability_eval_audit.py',
                                              'viability_cross_episode_baseline.py')]
    source_paths += [ROOT/k for k in ('eval.py','jepa.py','utils.py') if (ROOT/k).exists()]
    source_paths += sorted((ROOT/'scripts'/'common').glob('*.py'))
    source_paths += sorted((ROOT/'config').rglob('*.yaml'))
    shared = {str(q.relative_to(ROOT)):sha(q) for q in source_paths}
    shared.update(world_model=sha(weights), world_config=sha(weights.with_name('config.json')),
                  action_interface=sha(out/'calib.json'), dataset_identity=[str(h5.resolve()),h5.stat().st_size,h5.stat().st_mtime_ns],
                  recipe='OUR_v1_final_l2_et_beta1_std')
    if split is not None:
        shared['heldout'] = split['heldout_episode_ids'].tolist()
        with np.load(out/'cache_train.npz') as d:
            shared['act_mean']=d['act_mean'].tolist(); shared['act_std']=d['act_std'].tolist()
    jobs = []
    counts = {'l2':{'RUN':0,'SKIP':0},'OUR':{'RUN':0,'SKIP':0}}
    for label in ('l2','OUR'):
        for proto in PROTOCOLS:
            for s in SEEDS:
                method = 'l2' if label == 'l2' else 'subgoal_tdr'
                suffix = 'cross_episode' if proto == 'cross' else f'same_episode_off{proto[4:]}'
                pool = out/'pairs'/f'pairs_{suffix}_{pool_name}.json'
                assets = dict(shared, pool=sha(pool), n=args.n, chunk=25, seed=s, env=env, protocol=proto, label=label)
                if label == 'OUR':
                    for key,path in [('tdr',out/f'tdr_full_s{s}.pt'),('critic',out/f'critic_s{s}_tdr_holdout/critic.pt'),
                                     ('graph',out/f'graph_full_s{s}_htd{htd:g}_te0.9.pkl' if htd is not None else out/'pending'),
                                     ('gap_calibration',out/f'gap_calib_s{s}.json')]: assets[key]=sha(path)
                fingerprint=hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest()[:16]
                m=OmegaConf.create(dict(method=method, lookahead=(la or 0.0) if label=='OUR' else 0.0,
                    h_td=htd if htd is not None else 8.0, subgoal_threshold=htd if htd is not None else 8.0,
                    te=0.9,support_lambda=0.0,receding=5,retrieval=False,
                    final_thresh=la if label=='OUR' else None,final_metric='l2' if label=='OUR' else 'same',
                    critic_beta=1.0 if label=='OUR' else 0.0,critic_cost='et' if label=='OUR' else 'nlv',
                    compose='std',critic_filter=0.0,critic_final=True,step_units=None,
                    budget=5 if args.smoke else None,tag=('_SMOKE_' if args.smoke else '_pipe_')+fingerprint))
                tag=method_tag(m)
                result=out/'eval'/f'{tag}__{proto}__{pool_name}__s{s}__n{args.n}__heldout_disjoint__trainonly_assets.json'
                budget=5 if args.smoke else protocols[proto]['budget']
                ok=available(lambda:eval_ok(json.loads(result.read_text()),method,proto,s,args.n,tag,budget,args.smoke))
                if label=='OUR' and not critic_ready[s]: ok=False
                argv=['bash','scripts/gas_mpc_run.sh',f'[{len(jobs)+1}/40]',method,proto,
                      f'+mpc.graph_seed={s}',f'+mpc.tag={m.tag}']
                if label=='OUR': argv += [f'+mpc.h_td={htd}', '+mpc.te=0.9', f'+mpc.lookahead={la}',
                    f'+mpc.final_thresh={la}','+mpc.final_metric=l2','+mpc.critic_beta=1','+mpc.critic_cost=et',
                    '+mpc.compose=std',f'+mpc.critic={out}/critic_s{s}_tdr_holdout/critic.pt']
                if args.smoke: argv += ['+mpc.budget=5','solver.num_samples=8','solver.topk=2','solver.n_steps=1',
                                        f'hydra.run.dir={out}/hydra_runs/{label}_{proto}_s{s}']
                action='SKIP' if ok else 'RUN'; counts[label][action]+=1
                plan.append({'stage':f'eval_{label}_{proto}_s{s}','action':action,'command':argv,'result':str(result)})
                jobs.append((ok,argv,result,method,proto,s,tag,budget))
    print('EVAL',json.dumps(counts),flush=True)
    def eval_queue(slot):
        gpu = slot % args.gpus
        from contextlib import redirect_stdout,redirect_stderr
        if args.smoke:
            from hydra import compose,initialize_config_dir
            from gas_mpc_eval import main as evaluate
        for ok,argv,result,method,proto,s,tag,budget in jobs[slot::worker_count]:
            if ok: continue
            if result.exists(): archive(result,out)
            log=out/'logs'/'pipeline'/f'eval_{method}_{proto}_s{s}.log'
            # Smoke batches avoid repeated imports; exercise the actual Bash runner
            # once for each method, then invoke the same evaluator with Hydra configs.
            if args.smoke and not (proto=='same25' and s==0):
                overrides=[f'+mpc.method={method}',f'+mpc.protocol={proto}',f'+mpc.seed={s}',
                           f'eval.num_eval={args.n}']+argv[5:]
                with initialize_config_dir(version_base=None,config_dir=str(ROOT/'config/eval')):
                    cfg=compose(config_name=env,overrides=overrides)
                log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as handle,redirect_stdout(handle),redirect_stderr(handle):
                    evaluate(cfg)
            else:
                command(argv,gpu=gpu,log=log,extra={'PY':py,'SEED':str(s),'N':str(args.n)})
            assert eval_ok(json.loads(result.read_text()),method,proto,s,args.n,tag,budget,args.smoke)
    if args.run:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(eval_queue,range(worker_count)))
    print(f"SUMMARY {env}: critics to train={[s for s,_ in training]}; eval to run={sum(v['RUN'] for v in counts.values())}/40",flush=True)
    if args.plan_dir:
        dest=Path(args.plan_dir); dest.mkdir(parents=True,exist_ok=True)
        (dest/f'{env}.json').write_text(json.dumps(plan,indent=2))


def self_test():
    import numpy as np
    split={'episode_id':np.array([0,1,2]),'heldout_episode_ids':np.array([3])}
    ck={'step':60000,'args':{'seed':1,'training_only':True,'exclude_tdr_holdout':True,'label_source':'tdr'},
        'train_episodes':[0,1],'val_episodes':[2],'evaluation_episodes':[3]}
    assert critic_ok(ck,split,1)
    assert not critic_ok(dict(ck,val_episodes=[3]),split,1)
    assert not critic_ok(dict(ck,train_episodes=[0,3]),split,1)
    assert not critic_ok(dict(ck,step=1000),split,1)
    assert not critic_ok(ck,split,2)
    r={'method':'tag','protocol':'same25','seed':1,'n':2,'tasks':200,'pool':'task200u',
       'budget':50,'mpc':{'method':'l2','graph_seed':1},'first_hit_step':[10,-1],
       'n_success':1,'success_rate':50.0}
    assert eval_ok(r,'l2','same25',1,2,'tag',50)
    assert not eval_ok(dict(r,n=50),'l2','same25',1,2,'tag',50)
    assert not eval_ok(dict(r,pool='task200'),'l2','same25',1,2,'tag',50)
    assert not eval_ok(dict(r,first_hit_step=[51,-1]),'l2','same25',1,2,'tag',50)
    assert not eval_ok(dict(r,smoke_test=True),'l2','same25',1,2,'tag',50)
    print('Holdout/completion self-check passed')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true',help='execute missing stages')
    mode.add_argument('--dry-run',action='store_true',help='inspect only (default)')
    mode.add_argument('--smoke',action='store_true',help='execute a tiny isolated end-to-end rehearsal')
    ap.add_argument('--envs',nargs='+',choices=['pusht','reacher','cube'],default=['pusht','reacher','cube'])
    ap.add_argument('--gpus',type=int,default=4,help='visible physical GPUs; queues run two tasks per GPU')
    ap.add_argument('--n',type=int,default=50,help='fixed-pool prefix; default is first 50 of task200u')
    ap.add_argument('--train-cache-root',default='/ssd_scratch/mayaank.ashok/planning_trainonly')
    ap.add_argument('--plan-dir',help='write detailed JSON plans with all commands')
    ap.add_argument('--worker-env',choices=['pusht','reacher','cube'],help=argparse.SUPPRESS)
    ap.add_argument('--gpu-offset',type=int,default=0,help=argparse.SUPPRESS)
    ap.add_argument('--smoke-root',help=argparse.SUPPRESS)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test: self_test(); return
    if args.smoke:
        args.run=True
        args.n=1
        if not args.smoke_root:
            args.smoke_root='/ssd_scratch/mayaank.ashok/pipeline_smoke/'+datetime.now().strftime('%Y%m%d_%H%M%S')
        assert 'pipeline_smoke' in Path(args.smoke_root).parts
    if not 1<=args.n<=200 or args.gpus<1: ap.error('Require 1 <= n <= 200 and gpus >= 1')
    if args.worker_env: environment(args); return
    def worker(pair):
        index,env=pair
        # Process isolation keeps environment-specific module globals separate.
        argv=[sys.executable,str(Path(__file__).resolve()),'--worker-env',env,
              '--smoke' if args.smoke else ('--run' if args.run else '--dry-run'),
              '--gpus','1' if args.smoke else str(args.gpus),'--n',str(args.n),
              '--train-cache-root',args.train_cache_root]
        if args.smoke: argv += ['--smoke-root',args.smoke_root,'--gpu-offset',str(index % args.gpus)]
        if args.plan_dir: argv += ['--plan-dir',str(Path(args.plan_dir).resolve())]
        if args.smoke:
            log=Path(args.smoke_root)/f'{env}_driver.log'; log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f: subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            print(f'SMOKE PASSED {env}: full 5-seed/4-protocol/L2+OUR path; {log}',flush=True)
        else: subprocess.run(argv,cwd=ROOT,check=True)
    if args.smoke:
        print(f'SMOKE ROOT {args.smoke_root}; these artifacts cannot satisfy main-run checks',flush=True)
        with ThreadPoolExecutor(max_workers=min(args.gpus,len(args.envs))) as executor:
            list(executor.map(worker,enumerate(args.envs)))
    else:
        for pair in enumerate(args.envs): worker(pair)


if __name__=='__main__':
    main()
