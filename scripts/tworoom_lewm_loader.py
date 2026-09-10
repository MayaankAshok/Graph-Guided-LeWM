"""Re-export shim -- this module's actual content moved to common/lewm_loader.py (the
"tworoom" prefix was always a misnomer once Push-T started reusing load_tworoom_lewm
unchanged). Kept so existing imports (`from tworoom_lewm_loader import load_tworoom_lewm`,
used by ~20 scripts) keep working without modification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.lewm_loader import CKPT_DIR, REPO_ROOT, load_lewm, load_tworoom_lewm  # noqa: F401
