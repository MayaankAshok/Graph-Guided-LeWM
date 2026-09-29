"""One config-driven entrypoint for the active GAS-MPC preparation and evaluation flow.

Preview: python evaluator.py --config config/evaluations/pusht_headline.yaml
Run:     python evaluator.py --config config/evaluations/pusht_headline.yaml --run

Config `extends` entries compose left to right. Dataset and pretrained LeWM weights
may be supplied as cached paths or with explicit `prepare` argv lists in the config.
Split train/test encoding, TDR, graph, critic, task pools, and evaluation are then
loaded from verified caches or produced in dependency order.
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

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent
PROTOCOLS = ("same25", "same50", "same100", "cross")


def compose(path, stack=()):
    path = (ROOT / path).resolve()
    if path in stack:
        raise ValueError(f"config cycle: {' -> '.join(map(str, (*stack, path)))}")
    config = OmegaConf.load(path)
    merged = OmegaConf.create()
    for parent in config.get("extends", []):
        merged = OmegaConf.merge(merged, compose(path.parent / parent, (*stack, path)))
    config.pop("extends", None)
    return OmegaConf.merge(merged, config)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def check_config(cfg):
    if cfg.environment not in ("pusht", "reacher", "cube"):
        raise ValueError("active evaluator supports pusht, reacher, and cube")
    if cfg.evaluation.pool != "task200u":
        raise ValueError("headline evaluations require the task200u pool")
    if not 1 <= int(cfg.evaluation.num_eval) <= 200:
        raise ValueError("num_eval must be a prefix of the 200-task pool")
    if not cfg.evaluation.gpu_slots or len(set(cfg.evaluation.gpu_slots)) != len(cfg.evaluation.gpu_slots):
        raise ValueError("gpu_slots must contain distinct physical GPU IDs")
    if int(cfg.evaluation.workers_per_gpu) < 1:
        raise ValueError("workers_per_gpu must be positive")
    if any(p not in PROTOCOLS for p in cfg.evaluation.protocols):
        raise ValueError("unknown evaluation protocol")
    if any(m not in cfg.methods for m in cfg.evaluation.methods):
        raise ValueError("evaluation method has no composed definition")
    if any(not set(cfg.methods[m].requires) <= {"dataset", "model", "embeddings_train", "embeddings_test",
                                               "tdr", "graph", "critic", "tasks"}
           for m in cfg.evaluation.methods):
        raise ValueError("unknown method requirement")


def verify_encoded(path, cfg):
    import h5py
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    from common.heldout_tasks import validate_training_cache
    with np.load(path) as cache:
        meta = {k: cache[k] for k in ("training_only", "n_source_episodes", "heldout_frac",
                                     "episode_id", "heldout_episode_ids")}
        validate_training_cache(meta)
        if float(cache["heldout_frac"]) != float(cfg.training.heldout_frac):
            raise ValueError("encoded cache has a different holdout fraction")
        if absolute(str(cache["h5_path"].item())).resolve() != absolute(cfg.dataset.path).resolve():
            raise ValueError("encoded cache belongs to a different dataset")
        if absolute(str(cache["ckpt_dir"].item())).resolve() != absolute(cfg.model.directory).resolve():
            raise ValueError("encoded cache belongs to a different checkpoint")
        with h5py.File(absolute(cfg.dataset.path), "r") as dataset:
            actions = dataset["action"][:]
        if not (np.allclose(cache["act_mean"], actions.mean(axis=0), atol=1e-6) and
                np.allclose(cache["act_std"], actions.std(axis=0), atol=1e-6)):
            raise ValueError("encoded cache action statistics are not from the full dataset")


def verify_test_encoded(path, cfg):
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    from common.heldout_tasks import validate_test_cache
    with np.load(path) as cache:
        meta = {k: cache[k] for k in ("heldout_only", "n_source_episodes", "heldout_frac",
                                     "episode_id", "training_episode_ids")}
        validate_test_cache(meta)
        if float(cache["heldout_frac"]) != float(cfg.training.heldout_frac):
            raise ValueError("test embeddings have a different holdout fraction")
        if absolute(str(cache["h5_path"].item())).resolve() != absolute(cfg.dataset.path).resolve():
            raise ValueError("test embeddings belong to a different dataset")
        if absolute(str(cache["ckpt_dir"].item())).resolve() != absolute(cfg.model.directory).resolve():
            raise ValueError("test embeddings belong to a different checkpoint")
        if len(cache["z"]) != int(cache["ep_len"].sum()):
            raise ValueError("test embedding rows do not match held-out episodes")
        with np.load(absolute(cfg.output_root) / "cache_train.npz") as training:
            if not (np.array_equal(cache["training_episode_ids"], training["episode_id"]) and
                    np.allclose(cache["act_mean"], training["act_mean"]) and
                    np.allclose(cache["act_std"], training["act_std"])):
                raise ValueError("training and test cache metadata disagree")


def verify_tdr(path, seed):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck.get("done") or ck["cfg"].get("diagnostic_scope") != "training_only":
        raise ValueError(f"TDR seed {seed} is incomplete or lacks training-only provenance")


def verify_graph(path, seed, cfg):
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if (not graph.get("training_only") or graph["seed"] != seed or
            graph["h_td"] != float(cfg.training.h_td) or graph["te_threshold"] != float(cfg.training.te)):
        raise ValueError(f"graph seed {seed} does not match the configured training-only asset")


def verify_critic(path, seed, cfg):
    import numpy as np
    import torch
    with np.load(absolute(cfg.output_root) / "cache_train.npz") as cache:
        held = cache["heldout_episode_ids"]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    used = np.r_[ck["train_episodes"], ck["val_episodes"]]
    if (ck["step"] < int(cfg.training.critic_steps) or ck["args"].get("seed") != seed or
            not ck["args"].get("training_only") or ck["args"].get("label_source") != "tdr" or
            np.intersect1d(used, held).size or not np.array_equal(ck["evaluation_episodes"], held)):
        raise ValueError(f"critic seed {seed} is incomplete or overlaps evaluation episodes")


def verify_tasks(path, cfg):
    import numpy as np
    data = json.loads(path.read_text(encoding="utf-8"))
    with np.load(absolute(cfg.output_root) / "cache_test.npz") as cache:
        heldout = set(map(int, cache["episode_id"]))
    if (data["n"] != 200 or data["seed"] != 20260915 or not data["nonoverlap"] or
            not data["reject_solved"] or set(map(int, data["heldout_eps"])) != heldout or
            not set(map(int, data["start_ep"] + data["goal_ep"])) <= heldout or
            len(data["start_ep"]) != 200 or len(data["goal_ep"]) != 200):
        raise ValueError(f"invalid task200u pool: {path}")


def command(argv, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {' '.join(map(str, argv))} -> {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(list(map(str, argv)), cwd=ROOT, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def archive_stale(path, root):
    if not path.absolute().is_relative_to(root.absolute()):
        raise ValueError(f"refusing to archive outside {root}: {path}")
    destination = root / "bak" / "evaluator" / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / path.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)
    print(f"ARCHIVED stale asset {path} -> {destination}")


def fingerprint(cfg, name, protocol, seed, paths):
    # ponytail: size/mtime identifies local cache revisions; hash file content if assets are copied preserving both.
    files = {}
    for path in paths:
        path = Path(path)
        stat = path.stat() if path.exists() else None
        files[str(path.resolve())] = None if stat is None else [stat.st_size, stat.st_mtime_ns]
    payload = dict(config=OmegaConf.to_container(cfg, resolve=True), method=name,
                   protocol=protocol, seed=seed, files=files)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="composable YAML evaluation set")
    parser.add_argument("--run", action="store_true", help="produce missing assets and evaluate; default previews")
    args = parser.parse_args()
    cfg = compose(args.config)
    check_config(cfg)
    out = absolute(cfg.output_root)
    train_cache = absolute(cfg.train_cache_dir)
    dataset = absolute(cfg.dataset.path)
    model = absolute(cfg.model.directory)
    run_env = os.environ.copy()
    run_env.update(GAS_MPC_ENV=cfg.environment, GAS_MPC_OUT=str(out),
                   GAS_MPC_TRAIN_CACHE_DIR=str(train_cache),
                   MUJOCO_GL="egl")
    h5_var = {"pusht": "PUSHT_H5_PATH", "reacher": "REACHER_H5_PATH", "cube": "CUBE_H5_PATH"}[cfg.environment]
    run_env[h5_var] = str(dataset)
    run_env[cfg.environment.upper() + "_CKPT_DIR"] = str(model)
    gpu_slots = list(map(int, cfg.evaluation.gpu_slots))
    for stage in ("dataset", "model"):
        paths = [dataset] if stage == "dataset" else [model / "weights.pt", model / "config.json"]
        if all(path.exists() for path in paths):
            print(f"CACHED {stage}: {', '.join(map(str, paths))}")
            continue
        producer = cfg[stage].get("prepare", [])
        if not producer:
            if args.run:
                raise FileNotFoundError(f"{stage} is missing {paths}; supply cached files or configure {stage}.prepare")
            print(f"MISSING {stage}: {paths} (configure {stage}.prepare to build)")
            continue
        print(f"RUN {stage}: {len(producer)} preparation command(s)")
        if args.run:
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
            context = dict(dataset=str(dataset), checkpoint_dir=str(model), output_root=str(out),
                           python=sys.executable)
            for argv in producer:
                command([str(part).format_map(context) for part in argv], run_env,
                        out / "logs" / "evaluator" / f"{stage}.log")
            if not all(path.exists() for path in paths):
                raise FileNotFoundError(f"{stage} producer did not create {paths}")

    needed = {requirement for name in cfg.evaluation.methods for requirement in cfg.methods[name].requires}
    asset_env = dict(run_env, CUDA_VISIBLE_DEVICES=str(gpu_slots[0]), MUJOCO_EGL_DEVICE_ID=str(gpu_slots[0]))

    def stage(name, path, argv, verify=None):
        if path.exists() or path.is_symlink():
            if args.run and verify:
                try:
                    verify(path)
                except (ValueError, KeyError, OSError, EOFError, pickle.UnpicklingError):
                    target = path.resolve() if name == "encoded" and path.is_symlink() else None
                    archive_stale(path, out)
                    if target and target.exists():
                        archive_stale(target, train_cache)
            if path.exists():
                print(f"CACHED {name}: {path}")
                return
        print(f"RUN {name}: {path}")
        if args.run:
            command(argv, asset_env, out / "logs" / "evaluator" / f"{name}.log")
            if not path.exists():
                raise FileNotFoundError(f"{name} did not produce {path}")
            if verify:
                verify(path)

    encoded = out / "cache_train.npz"
    stage("encoded", encoded,
          [sys.executable, ROOT / "scripts/gas_mpc/gas_mpc_prepare.py", "encode", "--heldout-frac", cfg.training.heldout_frac],
          lambda path: verify_encoded(path, cfg))
    test_encoded = out / "cache_test.npz"
    stage("test_encoded", test_encoded,
          [sys.executable, ROOT / "scripts/gas_mpc/gas_mpc_prepare.py", "encode", "--heldout-frac", cfg.training.heldout_frac],
          lambda path: verify_test_encoded(path, cfg))
    seeds = list(map(int, cfg.evaluation.seeds))
    asset_seeds = sorted(set(seeds) | {0}) if {"tdr", "graph", "critic"} & needed else [0]
    if {"tdr", "graph", "critic"} & needed:
        for seed in asset_seeds:
            tdr = out / f"tdr_full_s{seed}.pt"
            stage(f"tdr_s{seed}", tdr,
                  [sys.executable, ROOT / "scripts/gas_mpc/gas_mpc_prepare.py", "tdr", "--seed", seed,
                   "--heldout-frac", cfg.training.heldout_frac, "--tdr-steps", cfg.training.tdr_steps],
                  lambda path, seed=seed: verify_tdr(path, seed))
    if "graph" in needed:
        for seed in seeds:
            graph = out / f"graph_full_s{seed}_htd{float(cfg.training.h_td):g}_te{float(cfg.training.te):g}.pkl"
            stage(f"graph_s{seed}", graph,
                  [sys.executable, ROOT / "scripts/gas_mpc/gas_mpc_prepare.py", "graph", "--seed", seed,
                   "--h-td", cfg.training.h_td, "--te", cfg.training.te],
                  lambda path, seed=seed: verify_graph(path, seed, cfg))
    if "critic" in needed:
        for seed in seeds:
            critic = out / f"critic_s{seed}_tdr_holdout" / "critic.pt"
            stage(f"critic_s{seed}", critic,
                  [sys.executable, ROOT / "scripts/critics/viability_train.py", "--env", cfg.environment,
                   "--cache", encoded, "--out", critic.parent, "--seed", seed,
                   "--steps", cfg.training.critic_steps, "--label-source", "tdr",
                   "--tdr", out / f"tdr_full_s{seed}.pt", "--xneg-tdr-factor", 1.25,
                   "--exclude-tdr-holdout"],
                  lambda path, seed=seed: verify_critic(path, seed, cfg))
    if "tasks" in needed:
        for protocol in cfg.evaluation.protocols:
            tag = "cross_episode" if protocol == "cross" else f"same_episode_off{protocol[4:]}"
            path = out / "pairs" / f"pairs_{tag}_task200u.json"
            stage(f"tasks_{protocol}", path,
                  [sys.executable, ROOT / "scripts/gas_mpc/gas_mpc_make_tasks.py"],
                  lambda path: verify_tasks(path, cfg))

    jobs = [(name, protocol, seed) for name in cfg.evaluation.methods
            for protocol in cfg.evaluation.protocols for seed in seeds]
    workers_per_gpu = int(cfg.evaluation.workers_per_gpu)
    workers = len(gpu_slots) * workers_per_gpu
    def evaluate(job, gpu):
        name, protocol, seed = job
        env = dict(run_env, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
        requirements = set(cfg.methods[name].requires)
        pair_tag = "cross_episode" if protocol == "cross" else f"same_episode_off{protocol[4:]}"
        identity = [dataset, model / "weights.pt", model / "config.json", encoded, test_encoded,
                    out / "pairs" / f"pairs_{pair_tag}_task200u.json",
                    ROOT / "evaluator.py", ROOT / "scripts/gas_mpc/gas_mpc_eval.py", ROOT / "scripts/gas_mpc/gas_mpc_prepare.py"]
        if "tdr" in requirements or "graph" in requirements or "critic" in requirements:
            identity.append(out / f"tdr_full_s{seed}.pt")
        if "graph" in requirements:
            identity.append(out / f"graph_full_s{seed}_htd{float(cfg.training.h_td):g}_te{float(cfg.training.te):g}.pkl")
        if "critic" in requirements:
            identity.append(out / f"critic_s{seed}_tdr_holdout" / "critic.pt")
        tag = fingerprint(cfg, name, protocol, seed, identity)
        overrides = [
            f"+mpc.h_td={float(cfg.training.h_td):g}", f"+mpc.te={float(cfg.training.te):g}",
            *cfg.methods[name].overrides,
            f"+mpc.protocol={protocol}", f"+mpc.seed={seed}", f"+mpc.graph_seed={seed}",
            f"+mpc.tag=_cfg{tag}",
            f"eval.num_eval={int(cfg.evaluation.num_eval)}",
        ]
        if "critic" in cfg.methods[name].requires:
            overrides.append(f"+mpc.critic={out / f'critic_s{seed}_tdr_holdout' / 'critic.pt'}")
        argv = [sys.executable, ROOT / "scripts/gas_mpc/gas_mpc_eval.py", *overrides]
        print(f"EVAL {name} {protocol} s{seed} gpu{gpu}", flush=True)
        if args.run:
            command(argv, env, out / "logs" / "evaluator" / f"{name}_{protocol}_s{seed}.log")
    if args.run:
        def worker(slot):
            gpu = gpu_slots[slot // workers_per_gpu]
            for job in jobs[slot::workers]:
                evaluate(job, gpu)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(worker, range(workers)))
    else:
        for index, job in enumerate(jobs):
            evaluate(job, gpu_slots[(index % workers) // workers_per_gpu])
    print(f"{'Completed' if args.run else 'Planned'} {len(jobs)} evaluations")


if __name__ == "__main__":
    main()
