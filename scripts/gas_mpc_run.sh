#!/usr/bin/env bash
# Sequential driver for scripts/gas_mpc_eval.py: one line per (method, protocol, extra
# hydra overrides). Every eval is resumable (per-chunk caches under outputs/pusht/eval/),
# so re-running this script only does the missing work. Logs: outputs/pusht/logs/.
#
#   bash scripts/gas_mpc_run.sh                       # runs the QUEUE below
#   bash scripts/gas_mpc_run.sh l2 same50             # one combo
#   bash scripts/gas_mpc_run.sh subgoal cross +mpc.lookahead=20

cd "$(dirname "$0")/.." || exit 1
# python: local Windows venv, or the Ada venv (override with PY=...)
if [ -z "$PY" ]; then
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
  elif [ -x /home2/mayaank.ashok/.venv/bin/python ]; then PY=/home2/mayaank.ashok/.venv/bin/python
  else PY=python; fi
fi
N=${N:-50}
SEED=${SEED:-0}
# Optional htop-only progress prefix supplied by parallel drivers, e.g. "[3/12]".
[[ ${1:-} =~ ^\[[0-9]+/[0-9]+\]$ ]] && shift
# GAS_MPC_ENV (default pusht) picks the environment; a non-Push-T env keeps everything
# (critic, logs, eval results) under outputs/<env>/ -- see gas_mpc_prepare.py
ENV=${GAS_MPC_ENV:-pusht}
OUTD=${GAS_MPC_OUT:-outputs/$ENV}
CRITIC_TPL="$OUTD"'/critic_s${SEED}_tdr_holdout/critic.pt'
mkdir -p "$OUTD/logs"

run() {
  local method=$1 proto=$2; shift 2
  local extra=("$@")
  # Match the learned models (TDR/graph/critic) to the CEM seed, unless the caller already
  # pinned graph_seed explicitly. Safe no-op for l2 (it never loads TDR/graph/critic).
  if [[ "${extra[*]}" != *"mpc.graph_seed="* ]]; then
    extra+=("+mpc.graph_seed=$SEED")
  fi
  if [[ ( "${extra[*]}" == *"mpc.critic_beta="* || "${extra[*]}" == *"mpc.critic_filter="* ) && "${extra[*]}" != *"mpc.critic="* ]]; then
    extra+=("+mpc.critic=$(eval echo "$CRITIC_TPL")")
  fi
  local tag; tag=$(echo "${method}_${extra[*]}" | tr -c 'A-Za-z0-9_.=-' '_' | sed 's/_*$//')
  # long override lists (e.g. B + critic_final + an explicit critic path) exceed the 255-byte
  # filename limit: keep a readable prefix and a short hash of the full tag
  if [ ${#tag} -gt 150 ]; then tag="${tag:0:120}_$(printf '%s' "$tag" | md5sum | cut -c1-8)"; fi
  local logf="$OUTD/logs/${tag}__${proto}__s${SEED}__n${N}.log"
  echo "[$(date +%H:%M:%S)] $method $proto ${extra[*]} -> $logf"
  $PY scripts/gas_mpc_eval.py +mpc.method="$method" +mpc.protocol="$proto" +mpc.seed="$SEED" \
      eval.num_eval="$N" "${extra[@]}" >> "$logf" 2>&1
  grep "^=== .*success=" "$logf" | tail -1
}

if [ $# -ge 2 ]; then
  run "$@"
  exit
fi

QUEUE=(
  "l2 same25"
  "l2 same50"
  "l2 cross"
  "tdr same25"
  "tdr same50"
  "tdr cross"
  "ctg same25"
  "ctg same50"
  "ctg cross"
)
for line in "${QUEUE[@]}"; do
  # shellcheck disable=SC2086
  run $line
done
