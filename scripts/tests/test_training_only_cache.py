"""Check that encoding never reads a final-holdout frame, including during normalization."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from gas_mpc import gas_mpc_prepare as prep
from gas_mpc import gas_mpc_eval as evaluation
from common.gas import TDR
from common.heldout_tasks import (cache_rows, fixed_episode_split, source_rows, validate_test_cache,
                                  validate_training_cache)


train, held = fixed_episode_split(np.arange(10), .2)
lengths = np.full(10, 5)
offsets = np.arange(10) * 5
allowed = np.repeat(np.isin(np.arange(10), train), lengths)


class GuardedColumn:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        rows = np.arange(len(self.values))[key]
        return self.values[key]


class FakeH5(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


data = np.arange(50, dtype=np.float32)[:, None]
fake = FakeH5(ep_offset=offsets, ep_len=lengths,
              pixels=GuardedColumn(data), state=GuardedColumn(data), action=GuardedColumn(data))
encoder_module = ModuleType("diagnostics.planning_cost_gate")
encoder_module.make_encode_frame = lambda model, batch: lambda pixels: pixels
mechanics = SimpleNamespace(h5_path=lambda root: root / "fake.h5", ckpt_dir=lambda root: root / "model",
                            state_dim=1, action_dim=1,
                            read_state_slice=lambda f, lo, n: f["state"][lo:lo + n])

with tempfile.TemporaryDirectory() as directory:
    out = Path(directory)
    args = SimpleNamespace(batch=8, heldout_frac=.2, seed=0)
    with patch.object(prep, "OUT", out), patch.object(prep, "MECH", mechanics), \
         patch.object(prep.h5py, "File", return_value=fake), \
         patch("common.lewm_loader.load_lewm", return_value=None), \
         patch.dict(sys.modules, {"diagnostics.planning_cost_gate": encoder_module}), \
         patch.dict(os.environ, {"GAS_MPC_TRAIN_CACHE_DIR": directory}):
        prep.cmd_encode(args)
        cache = prep.load_cache()
        validate_training_cache(cache)
        np.testing.assert_array_equal(cache["episode_id"], train)
        np.testing.assert_array_equal(cache["heldout_episode_ids"], held)
        np.testing.assert_array_equal(source_rows(cache, np.arange(len(cache["z"]))),
                                      np.flatnonzero(allowed))
        np.testing.assert_array_equal(cache_rows(cache, np.flatnonzero(allowed)), np.arange(len(cache["z"])))
        np.testing.assert_array_equal(cache["z"][:, 0], np.flatnonzero(allowed))
        with np.load(out / "cache_test.npz") as test:
            validate_test_cache({k: test[k] for k in (
                "heldout_only", "n_source_episodes", "heldout_frac", "episode_id", "training_episode_ids")})
            np.testing.assert_array_equal(test["episode_id"], held)
            np.testing.assert_array_equal(test["z"][:, 0], np.flatnonzero(~allowed))
            np.testing.assert_array_equal(cache_rows(test, np.flatnonzero(~allowed)), np.arange(len(test["z"])))
            assert not np.intersect1d(cache["episode_id"], test["episode_id"]).size
            try:
                cache_rows(test, [int(np.flatnonzero(allowed)[0])])
            except ValueError:
                pass
            else:
                raise AssertionError("A training row was accepted as a test embedding")
        prep.cmd_encode(args)  # Resuming verifies the split without reading frame columns.
        np.testing.assert_array_equal(prep.task_heldout_episodes(np.repeat(np.arange(10), 5)), held)
        model = TDR(cache["z"].shape[1], 2, 8, 1)
        original_weights = model.state_dict()
        torch.save(dict(done=True, step=50000, tdr=original_weights, history=[{"old_holdout_metric": 123}],
                        cfg=dict(tdr_dim=2, tdr_hidden=8, tdr_layers=1, heldout_frac=.2, seed=0)),
                   prep.tdr_path(0))
        prep.cmd_refresh_tdr(args)
        refreshed = torch.load(prep.tdr_path(0), map_location="cpu", weights_only=False)
        assert refreshed["cfg"]["diagnostic_scope"] == "training_only"
        assert "old_holdout_metric" not in refreshed["history"][0]
        for key in original_weights:
            torch.testing.assert_close(original_weights[key], refreshed["tdr"][key])
        features = prep.ensure_psi(0)
        assert features.shape == (40, 2), "Final-holdout frames entered the feature cache"
        with patch.object(evaluation, "OUT", out), patch.object(evaluation, "GAP_CALIB_GAPS", [1, 2]):
            table = evaluation.gap_calibration(0, per_gap=20)
            assert table["scope"] == "training_only"
            path = out / f"gap_calib_s0{prep.TAG}.json"
            path.write_text('{"scope": "full_dataset"}')
            try:
                evaluation.gap_calibration(0)
            except ValueError:
                pass
            else:
                raise AssertionError("A stale full-data calibration was accepted")

try:
    validate_training_cache({"training_only": False})
except ValueError:
    pass
else:
    raise AssertionError("An unsafe full-data cache was accepted")
print("Training/test embedding caches are disjoint and match the fixed episode split")
