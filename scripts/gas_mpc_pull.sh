#!/usr/bin/env bash
# Pull the Ada screen results (method-level + chunk JSONs, not the traj npz) into the local
# outputs/pusht/eval/ and regenerate docs/gas-mpc/results_table.tex.
cd "$(dirname "$0")/.." || exit 1
R=/home2/mayaank.ashok/lewm_research/outputs/pusht
mkdir -p outputs/pusht/eval outputs/pusht/logs_ada
scp -q "ada:$R/eval/*.json" outputs/pusht/eval/ 2>/dev/null
scp -q "ada:$R/logs/*.log" outputs/pusht/logs_ada/ 2>/dev/null
scp -q "ada:/home2/mayaank.ashok/lewm_research/outputs/pusht/pairs/pairs_*_n50_s*.json" outputs/pusht/pairs/ 2>/dev/null
.venv/Scripts/python.exe scripts/gas_mpc_report.py 2>&1 | grep -v Warning | grep "&\|wrote"
