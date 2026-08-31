"""Path setup so a step can be run directly: `python3 steps/stepN.py`."""

import pathlib
import sys

TOPIC = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOPIC / "src"))
sys.path.insert(0, str(TOPIC.parents[2]))  # repo root, for `common`
