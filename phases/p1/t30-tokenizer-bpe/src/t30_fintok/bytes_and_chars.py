"""Step 1: UTF-8, and the three different numbers people call "length".

A .NET string is UTF-16, so `"".Length` counts 16-bit code units and a single
emoji is 2. A Python `str` is a sequence of *code points*, so `len()` gives 1.
Neither is the number a tokenizer cares about, which is **bytes**.

Getting this straight is not pedantry. It is why byte-level BPE exists: build
the base vocabulary out of the 256 byte values and there is no such thing as an
out-of-vocabulary string, in any language, ever. Build it out of characters and
you inherit Unicode's entire surface area as a source of production bugs.
"""

from __future__ import annotations

import unicodedata

EXAMPLES = [
    ("ascii", "NSE"),
    ("accented (precomposed)", "café"),
    ("accented (combining)", "café"),
    ("devanagari", "रिलायंस"),
    ("cjk", "東京証券取引所"),
    ("emoji", "📈"),
    ("emoji zwj sequence", "👩‍💻"),
    ("currency", "₹1,20,000"),
]


def utf8_lengths(text: str) -> dict[str, int]:
    return {
        "python_len_code_points": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "utf16_code_units": len(text.encode("utf-16-le")) // 2,
        "grapheme_clusters_approx": len(
            [c for c in text if not unicodedata.combining(c)]
        ),
    }


def byte_view(text: str, limit: int = 24) -> str:
    raw = text.encode("utf-8")[:limit]
    return " ".join(f"{b:02x}" for b in raw)


def char_vs_byte_table() -> list[dict]:
    rows = []
    for label, text in EXAMPLES:
        row = {"label": label, "text": text, "bytes_hex": byte_view(text)}
        row.update(utf8_lengths(text))
        rows.append(row)
    return rows


def normalisation_changes_the_answer() -> dict:
    """NFC vs NFD: the same visible string, different byte sequences.

    A tokenizer that does not fix a normalisation form will assign different
    token ids to strings a user cannot tell apart — and then the model behaves
    differently for reasons nobody can see in the logs.
    """
    composed = "café"
    decomposed = unicodedata.normalize("NFD", composed)
    return {
        "look_identical": composed == unicodedata.normalize("NFC", decomposed),
        "same_code_points": composed == decomposed,
        "nfc_bytes": len(composed.encode("utf-8")),
        "nfd_bytes": len(decomposed.encode("utf-8")),
    }
