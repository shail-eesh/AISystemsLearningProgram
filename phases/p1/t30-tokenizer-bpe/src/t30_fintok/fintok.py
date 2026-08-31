"""Step 5: FinTok — the tokenizer every later model in this course uses.

FinTok is a byte-level BPE with a finance-aware pre-tokenizer and six special
tokens, trained on the synthetic financial corpus in `corpus.py`. It is the
tokenizer AlphaSLM (T15) is pretrained with, which means every downstream topic
— LoRA (T17), DPO (T19), quantization (T8), serving (T3) — inherits it. Changing
it later means retraining everything, so it is worth getting the *interface*
right now even at a small vocabulary.

**A deviation from the plan, stated plainly.** The master plan asks for
FinTok-16k. Asking this trainer for 16,384 merges on the committed corpus yields
3,233: the corpus is templated synthetic text and simply runs out of pairs that
occur twice. Rather than pad the vocabulary with junk merges of frequency 1, the
shipped artefact is **FinTok-4k** and `bench/compression.py` measures the
saturation curve that explains why. Point the same trainer at a real filings
corpus (`FORGE_ALLOW_NETWORK=1`, see `common/data/loaders.py`) and it scales.
"""

from __future__ import annotations

from pathlib import Path

from .bpe import Tokenizer, TrainStats
from .corpus import financial_corpus, split

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FINTOK_PATH = ARTIFACTS / "fintok.json"

#: Special tokens, in a fixed order — their ids are part of the model contract.
SPECIAL_TOKENS = [
    "<|endoftext|>",     # document boundary; also the pretraining separator
    "<|pad|>",           # batch padding, masked out of the loss
    "<|filing|>",        # a regulatory filing section follows
    "<|commentary|>",    # market commentary follows
    "<|announcement|>",  # an exchange announcement follows
    "<|order|>",         # a (simulated, paper) order ticket follows
]

DEFAULT_VOCAB_SIZE = 4096
DEFAULT_DOCS = 3000


def train_fintok(
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    docs: int = DEFAULT_DOCS,
    pattern: str = "finance",
    verbose_every: int = 0,
) -> tuple[Tokenizer, TrainStats, list[str]]:
    """Train FinTok and return it with the held-out split it was *not* trained on."""
    train_docs, holdout = split(financial_corpus(docs=docs))
    tok, stats = Tokenizer.train(
        train_docs,
        vocab_size,
        pattern=pattern,
        special_tokens=SPECIAL_TOKENS,
        name="fintok",
        verbose_every=verbose_every,
    )
    return tok, stats, holdout


def load_fintok() -> Tokenizer:
    """Load the committed artefact, training it once if it is missing."""
    if not FINTOK_PATH.exists():
        tok, _, _ = train_fintok()
        tok.save(FINTOK_PATH)
        return tok
    return Tokenizer.load(FINTOK_PATH)


def build_artifact() -> Path:
    """Regenerate `artifacts/fintok.json`. Deterministic — the corpus is seeded."""
    tok, _, _ = train_fintok()
    return tok.save(FINTOK_PATH)


if __name__ == "__main__":  # pragma: no cover
    path = build_artifact()
    tok = Tokenizer.load(path)
    print(f"wrote {path} — vocab {tok.vocab_size}, {len(tok.merges)} merges")
