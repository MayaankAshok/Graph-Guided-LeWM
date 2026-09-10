"""Crash-safe checkpoint read/write -- environment-agnostic, already pure and path-
parameterized before this move (no coupling to any specific tier/environment naming), so
this is a straight relocation from tworoom_b3_datatiers.py (kept there as a re-export shim).
"""

import os
import time

import torch

from common.log_util import log


def save_checkpoint(path, step, v_net, v_target, q_net, q_target, opt, history, np_rng, done, extra=None):
    """extra: optional dict merged into the payload (e.g. an actor net's state_dict and a
    peak-step snapshot) -- lets callers that train more than just V/Q/opt reuse this same
    hardened write path instead of duplicating it."""
    tmp = path.with_suffix(".pt.tmp")
    payload = dict(
        step=step, done=done, history=history,
        v_net=v_net.state_dict(), v_target=v_target.state_dict(),
        q_net=q_net.state_dict(), q_target=q_target.state_dict(),
        opt=opt.state_dict(),
        numpy_rng=np_rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
        torch_cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )
    if extra:
        payload.update(extra)
    # Verify the tmp write BEFORE replacing the previous good checkpoint -- found necessary
    # after an actual bad-kill during a session teardown left a *destination* .pt file that
    # was valid-looking (right size, right mtime) but entirely zero bytes (no PK\x03\x04 zip
    # magic), i.e. os.replace's atomicity guarantee doesn't help if what got renamed in was
    # itself corrupt. A cheap magic-byte check on tmp, retried before we ever touch the real
    # destination, means a bad write never promotes over a known-good checkpoint.
    for attempt in range(5):
        # fsync forces the bytes physically to disk before we trust them -- added after a
        # .pt AND its .pt.bak both turned up corrupt at the identical timestamp, which points
        # to the environment's abrupt teardown losing not-yet-flushed writes across multiple
        # files at once (not a single-rename race the earlier magic-byte check alone can
        # catch). This can't fully close the gap (the directory-entry update from os.replace
        # itself isn't separately fsync'd, no portable way to do that on Windows), but it's
        # the strongest additional durability guarantee available here.
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        with open(tmp, "rb") as fh:
            magic = fh.read(4)
        if magic == b"PK\x03\x04":
            break
        log(f"[checkpoint] WARNING: write to {tmp} produced bad magic {magic!r}, "
            f"retrying (attempt {attempt+1}/5)")
    else:
        raise IOError(f"failed to write a valid checkpoint to {tmp} after 5 attempts")

    # Even with the write-side verification above, a checkpoint has still turned up corrupt
    # (zero-byte, no zip magic) TWICE now during an abrupt environment-level teardown --
    # meaning corruption can apparently strike during/around the rename itself, not only
    # during the write to tmp. There's no way to verify a rename after the fact (by the time
    # we could check, the old good bytes are already gone), so the only real protection is
    # keeping one generation of backup: demote the current (already-verified-good) path to
    # path.pt.bak before promoting tmp over it. If THIS rename is what gets corrupted, the
    # resumer falls back to .bak and loses at most one checkpoint interval instead of the
    # whole run.
    bak = path.with_suffix(".pt.bak")
    if path.exists():
        for attempt in range(5):
            try:
                os.replace(path, bak)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))

    # os.replace is atomic in the sense that a reader never sees a half-written file, but on
    # Windows it can still raise WinError 5 ("Access is denied") if AV/indexing briefly holds
    # a handle on the destination; that's transient, so retry a few times before giving up.
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def load_checkpoint(path):
    """Load path, falling back to path.pt.bak if path is missing/corrupt (see the backup
    note in save_checkpoint). Returns None if neither is usable (fresh run)."""
    bak = path.with_suffix(".pt.bak")
    if path.exists():
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            log(f"[checkpoint] WARNING: {path} failed to load ({e!r}), trying {bak}")
    if bak.exists():
        try:
            ckpt = torch.load(bak, map_location="cpu", weights_only=False)
            log(f"[checkpoint] loaded fallback {bak} (step={ckpt['step']}) -- "
                f"primary checkpoint was missing or corrupt")
            return ckpt
        except Exception as e:
            log(f"[checkpoint] WARNING: fallback {bak} ALSO failed to load ({e!r}) -- starting fresh")
    return None
