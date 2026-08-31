"""The list of topic modules that register components into AlphaDesk.

This grows by one line per topic. Keeping it as *data* (rather than imports at
the top of `__init__.py`) is what lets the desk boot with any subset of the
course built: `load_all()` reports what failed instead of raising.
"""

from __future__ import annotations

#: module path -> the topic that owns it. Append as topics land.
TOPIC_MODULES: dict[str, str] = {
    "p0_1_oms.alphadesk_hook": "P0.1",
    "p0_2_indicators.alphadesk_hook": "P0.2",
    "p0_3_training.alphadesk_hook": "P0.3",
    "t31_autograd.alphadesk_hook": "T31",
    "t16a_matmul.alphadesk_hook": "T16A",
}
