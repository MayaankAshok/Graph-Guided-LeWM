#!/usr/bin/env bash
# Incrementally rsync this repo's code to a remote machine.
#
# Only source code, configs, and small tracked assets are sent — see
# rsync_excludes.txt for what's left out (datasets, run outputs, venvs, etc).
# rsync's delta-transfer means re-running this only ships what changed.
#
# Configure once by editing the defaults below, or override per-run:
#   REMOTE_USER=alice REMOTE_HOST=gpu-box REMOTE_PATH=~/LEWM ./scripts/sync_to_remote.sh
#   ./scripts/sync_to_remote.sh alice@gpu-box:~/LEWM
#
# Add --dry-run to preview without transferring:
#   ./scripts/sync_to_remote.sh --dry-run

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-mayaank.ashok}"
REMOTE_HOST="${REMOTE_HOST:-ada.iiit.ac.in}"
REMOTE_PATH="${REMOTE_PATH:-~/LEWM}"
SSH_PORT="${SSH_PORT:-22}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXTRA_ARGS=()
TARGET=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) EXTRA_ARGS+=(--dry-run) ;;
    *) TARGET="$arg" ;;
  esac
done
TARGET="${TARGET:-${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}}"
REMOTE_DEST_PATH="${TARGET#*:}"

rsync -avz --delete --human-readable --progress \
  -e "ssh -p ${SSH_PORT}" \
  --rsync-path="mkdir -p '${REMOTE_DEST_PATH}' && rsync" \
  --exclude-from="${SCRIPT_DIR}/rsync_excludes.txt" \
  "${EXTRA_ARGS[@]}" \
  "${REPO_ROOT}/" "${TARGET}/"
