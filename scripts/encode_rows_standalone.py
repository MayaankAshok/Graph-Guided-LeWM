"""Encode arbitrary h5 rows to LeWM latents. Used to backfill z_start for the Experiment-2
audit npz files (which store start_row but not z_start -- the audit only needed the
predicted/true endpoint latents, not the starting one).

    python scripts/encode_rows_standalone.py --rows-json rows.json --out z_start.npz

rows.json: {"25": [150798, 160502, ...], "50": [32872, 209505, ...]}
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.gas import DEV
from common.lewm_loader import load_lewm
from common.log_util import log
from gas_mpc_prepare import MECH, ROOT
from planning_cost_gate import make_encode_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mech = MECH
    h5_path, ckpt_dir = mech.h5_path(ROOT), mech.ckpt_dir(ROOT)
    rows_by_key = json.loads(Path(args.rows_json).read_text())

    model = load_lewm(Path(ckpt_dir), device=DEV)
    encode = make_encode_frame(model, batch=256)

    out = {}
    with h5py.File(h5_path, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as f:
        for key, rows in rows_by_key.items():
            rows = np.asarray(rows, dtype=np.int64)
            order = np.argsort(rows)
            pixels = f["pixels"][rows[order]]
            z = encode(pixels)
            z_out = np.empty_like(z)
            z_out[order] = z
            out[f"z_start_{key}"] = z_out
            log(f"[encode_rows] {key}: {len(rows)} rows encoded")

    np.savez(args.out, **out)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
