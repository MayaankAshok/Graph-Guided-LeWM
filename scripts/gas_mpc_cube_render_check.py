"""Live-env plumbing check for the cube GAS-MPC eval stage (ada_gas_mpc_cube.sh): builds
swm/OGBCube-v0 through swm.World with CubeMechanics.world_kwargs, restores three dataset rows
through config/eval/cube.yaml's callables (set_state + set_target_pos), and compares the live
render to the stored pixels and the live info state to the h5 state. Same idea as Reacher's
render_check, which caught a mujoco-version render mismatch. Exit 1 if the mean abs pixel
diff exceeds 4 on any row. 2026-09-18: mujoco 3.5.0 + ogbench 1.2.1 give [0.0, 0.08, 0.2].
"""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
import h5py, hdf5plugin, numpy as np, mujoco
import stable_worldmodel as swm
from stable_worldmodel.world.world import _apply_callables
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from common.envs import ENV_MECHANICS
m = ENV_MECHANICS["cube"]
t0 = time.time()
from pathlib import Path
f = h5py.File(m.h5_path(Path(__file__).resolve().parent.parent), "r")
world = swm.World(env_name="swm/OGBCube-v0", num_envs=1, max_episode_steps=10, image_shape=(224, 224), **m.world_kwargs)
world.reset(seed=None)
print("world ok", round(time.time() - t0, 1), "s; info keys sample:", [k for k in world.infos if "block" in k or "effector" in k][:4])
env = world.envs.envs[0].unwrapped
callables = [
    {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
    {"method": "set_target_pos", "args": {"cube_id": {"value": 0, "in_dataset": False},
                                          "target_pos": {"value": "goal_privileged_block_0_pos"},
                                          "target_quat": {"value": "goal_privileged_block_0_quat"}}},
]
w = world.envs.envs[0]
while not hasattr(w, "_get_pixels"):
    w = w.env
diffs = []
for r in [0, 500000, 1500000]:
    row = {"qpos": f["qpos"][r], "qvel": f["qvel"][r],
           "goal_privileged_block_0_pos": f["privileged_block_0_pos"][r + 25],
           "goal_privileged_block_0_quat": f["privileged_block_0_quat"][r + 25]}
    _apply_callables(env, callables, row)
    live = w._get_pixels()[0]["pixels"].astype(float)
    diffs.append(float(np.abs(live - f["pixels"][r].astype(float)).mean()))
    st_live = m.state_from_infos({k: np.asarray(v)[None, None] for k, v in env.get_step_info().items() if isinstance(v, np.ndarray)})
    print(f"row {r}: live state {st_live[0].round(3).tolist()} h5 {m.read_state_slice(f, r, 1)[0].round(3).tolist()}")
world.close()
ok = max(diffs) < 4
print(f"[render] mujoco {mujoco.__version__} ogbench: dataset-vs-live mean abs pixel diff {np.round(diffs, 2).tolist()} (Reacher: <4 good, ~18 bad)")
sys.exit(0 if ok else 1)
