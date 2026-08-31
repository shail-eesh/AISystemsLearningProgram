"""The AlphaSLM pretraining corpus — bigger, tagged, and reproducible offline.

Policy, unchanged from T30 and `common/data`: **offline first, nothing licensed
redistributed.** Every document is generated deterministically from templates
with fictional issuers, or derived from the sample files already committed to
this repository. The whole corpus rebuilds from a seed on a machine with no
network.

Four document types, each prefixed with its FinTok special token so the model
learns that a filing and a market-commentary line are different registers:

    <|filing|>        section-structured filing prose with numbers
    <|commentary|>    daily market commentary
    <|announcement|>  exchange announcements
    <|order|>         order tickets from the committed sample
    (plus the raw price tape, which carries no tag — it is not prose)

The split is **by document, held out before any shard is written**, and the
held-out set is filings only, because that is what T15's perplexity claim is
about. Splitting after packing would leak: a document straddling the boundary
puts its own first half in the training set.
"""

from __future__ import annotations

import csv
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[5]
SAMPLES = REPO / "common" / "data" / "samples"
T30_SRC = REPO / "phases" / "p1" / "t30-tokenizer-bpe" / "src"

TAGS = {
    "filing": "<|filing|>",
    "commentary": "<|commentary|>",
    "announcement": "<|announcement|>",
    "order": "<|order|>",
}


def _t30():
    """T30's corpus generator, imported by path rather than by package.

    Topics do not import each other in this course — that is what the AlphaDesk
    registry is for. But a *corpus generator* is data, not a component, and
    duplicating 200 lines of templates to avoid one path insert would be worse.
    """
    if str(T30_SRC) not in sys.path:
        sys.path.insert(0, str(T30_SRC))
    from t30_fintok import corpus as t30_corpus

    return t30_corpus


def _tag_of(doc: str) -> str:
    head = doc.split(".", 1)[0].lower()
    if "commentary" in head:
        return TAGS["commentary"]
    if any(w in head for w in ("intimation", "disclosure", "certificate", "outcome",
                               "record date")):
        return TAGS["announcement"]
    return TAGS["filing"]


def order_tickets() -> list[str]:
    """The committed order sample, rendered as sentences a model can read.

    These are *simulated* tickets from a fictional venue. AlphaDesk never places
    a real order; the tickets exist so the desk's model has seen the vocabulary
    of an order blotter.
    """
    out: list[str] = []
    with (SAMPLES / "orders_sample.csv").open() as fh:
        for row in csv.DictReader(fh):
            out.append(
                f"{TAGS['order']} Order {row['order_id']} dated {row['date']}: "
                f"{row['side']} {row['quantity']} {row['symbol']} as a "
                f"{row['order_type'].lower()} order at Rs {row['limit_price']}. "
                f"Filled {row['filled_quantity']} at an average of Rs "
                f"{row['avg_fill_price']}; status {row['status']} on venue "
                f"{row['venue']}. Simulated order, educational use only."
            )
    return out


def filing_excerpts() -> list[str]:
    """The committed synthetic filings, one document per section."""
    out: list[str] = []
    for line in (SAMPLES / "filings_sample.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.append(
            f"{TAGS['filing']} {rec['issuer']} {rec['form']} {rec['fiscal_period']} "
            f"{rec['section'].replace('_', ' ')}. {rec['text']}"
        )
    return out


def price_tape() -> list[str]:
    """One document per symbol: its whole year of OHLCV as a tape.

    Grouping by symbol rather than by date is deliberate — it gives the model
    long stretches of a single symbol's price behaviour, which is the only way a
    context of a few hundred tokens can contain anything resembling a series.
    """
    rows: dict[str, list[str]] = {}
    with (SAMPLES / "ohlcv_sample.csv").open() as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["symbol"], []).append(
                f"{r['date']} O {float(r['open']):.2f} H {float(r['high']):.2f} "
                f"L {float(r['low']):.2f} C {float(r['close']):.2f} V {int(r['volume'])}"
            )
    return [f"Price tape {sym}\n" + "\n".join(lines) for sym, lines in sorted(rows.items())]


def build_corpus(*, docs: int = 30_000, seed: int = 15) -> list[str]:
    """The full pretraining corpus as a list of documents.

    ``docs`` controls only the generated portion; the sample-derived documents
    are always included. At the default the corpus is roughly 12 MB of text —
    about 2.4M FinTok tokens, which packs in two seconds and is large enough
    that the held-out perplexity is measuring generalisation rather than how
    much of a 12 MB file a 6M-parameter model can memorise.
    """
    gen = _t30().financial_corpus(docs=docs, seed=seed)
    tagged = [f"{_tag_of(d)} {d}" for d in gen]
    everything = tagged + filing_excerpts() + order_tickets() + price_tape()
    random.Random(seed).shuffle(everything)
    return everything


def split_documents(documents: list[str], *, holdout: float = 0.1, seed: int = 15,
                    ) -> tuple[list[str], list[str]]:
    """Hold out whole documents, never token ranges.

    The held-out set is drawn only from ``<|filing|>`` documents: the capsule's
    claim is "perplexity on held-out filings", and a validation set diluted with
    price tape would measure something easier and different.
    """
    rng = random.Random(seed + 1)
    filings = [d for d in documents if d.startswith(TAGS["filing"])]
    n_val = max(1, int(len(filings) * holdout))
    val = set(rng.sample(range(len(filings)), n_val))
    val_docs = [d for i, d in enumerate(filings) if i in val]
    val_set = set(val_docs)
    train_docs = [d for d in documents if d not in val_set]
    return train_docs, val_docs


def corpus_stats(documents: list[str]) -> dict:
    chars = sum(len(d) for d in documents)
    by_tag: dict[str, int] = {}
    for d in documents:
        tag = d.split(" ", 1)[0] if d.startswith("<|") else "<|tape|>"
        by_tag[tag] = by_tag.get(tag, 0) + 1
    return {
        "documents": len(documents),
        "chars": chars,
        "mean_doc_chars": chars / max(len(documents), 1),
        "by_tag": dict(sorted(by_tag.items())),
        "distinct_words": len({w for d in documents for w in d.split()}),
    }
