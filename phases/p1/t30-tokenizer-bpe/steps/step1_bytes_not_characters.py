#!/usr/bin/env python3
"""Step 1 — "length" is three different numbers, and only one of them matters.

Run:  python3 steps/step1_bytes_not_characters.py

A .NET `string` is UTF-16, so `"📈".Length` is 2. A Python `str` is code points,
so `len("📈")` is 1. UTF-8 says 4 bytes. A human says one character.

Byte-level BPE picks bytes and never looks back, and that single choice deletes
the entire "unknown token" problem: every string has a UTF-8 encoding, and every
byte of every UTF-8 encoding is already in the vocabulary.
"""

import _bootstrap  # noqa: F401
from t30_fintok import byte_view, char_vs_byte_table
from t30_fintok.bytes_and_chars import normalisation_changes_the_answer


def the_table() -> None:
    print(f"  {'':<24} {'py len':>7} {'utf8':>6} {'utf16':>6}  bytes")
    for row in char_vs_byte_table():
        print(f"  {row['label']:<24} {row['python_len_code_points']:>7} "
              f"{row['utf8_bytes']:>6} {row['utf16_code_units']:>6}  {row['bytes_hex']}")


def normalisation() -> None:
    r = normalisation_changes_the_answer()
    print("\n  'café' typed two ways:")
    print(f"    NFC (precomposed): {r['nfc_bytes']} bytes")
    print(f"    NFD (combining):   {r['nfd_bytes']} bytes")
    print(f"    equal as Python strings? {r['same_code_points']}")
    print("\n  Two byte sequences a user cannot tell apart become two different")
    print("  token sequences. Fix a normalisation form at the door or spend a")
    print("  Friday explaining why the same prompt gives two answers.")


def why_bytes_win() -> None:
    print("\n  the base vocabulary of a byte-level BPE is these 256 tokens:")
    print("    " + " ".join(f"{i:02x}" for i in range(16)) + " ...")
    print("  and therefore:")
    for text in ("hello", "रिलायंस", "👩‍💻", "\x00\xff"):
        raw = text.encode("utf-8")
        print(f"    {text!r:<14} -> {len(raw):>2} base tokens, 0 unknowns  [{byte_view(text)}]")


if __name__ == "__main__":
    print("== three lengths ==")
    the_table()
    normalisation()
    why_bytes_win()
