"""Path setup so a step can be run directly: `python3 steps/stepN.py`.

T15 is the first topic that depends on *other* topics' code at import time —
T4's `GPT` and T30's FinTok — so this bootstrap adds every topic's `src` to the
path, exactly as the repo-root `conftest.py` does for pytest. (The AlphaDesk
registry is how topics depend on each other at *runtime*; a training script that
needs the architecture class is a build-time dependency and is allowed to import
it.)
"""

import pathlib
import sys

TOPIC = pathlib.Path(__file__).resolve().parent.parent
ROOT = TOPIC.parents[2]
sys.path.insert(0, str(ROOT))
for src in sorted((ROOT / "phases").glob("*/*/src")):
    if src.is_dir():
        sys.path.insert(0, str(src))
sys.path.insert(0, str(TOPIC / "src"))
