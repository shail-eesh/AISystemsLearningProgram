"""Make every topic's implementation importable without packaging ceremony.

Each topic ships `src/<unique_pkg>/`; this adds every `src` directory to
`sys.path` so `from p0_1_oms import Order` works from the repo root, from the
topic folder, and from an individual step script. Unique package names (rather
than a shared `src`) are what keep 51 topics from colliding in `sys.modules`.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for src in sorted((ROOT / "phases").glob("*/*/src")):
    if src.is_dir():
        sys.path.insert(0, str(src))
