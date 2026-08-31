"""Perplexity where it means something, and generation you can judge.

Two things this module refuses to do, both of which are how model reports get
written:

* **Report one perplexity number.** Perplexity on a mixed corpus is dominated by
  whichever register is most predictable — here, the price tape, which is nearly
  deterministic. Broken out by document tag, the same model can be at
  perplexity 2 on tape and 12 on filing prose, and only the second number is
  about language.
* **Compare perplexities across tokenizers.** Perplexity is per *token*, so a
  tokenizer with a larger vocabulary gets a better-looking number for free.
  ``bits_per_char`` is reported alongside, because it is comparable.
"""

from __future__ import annotations

import math

import torch

TAG_NAMES = {
    "<|filing|>": "filing",
    "<|commentary|>": "commentary",
    "<|announcement|>": "announcement",
    "<|order|>": "order",
}


@torch.no_grad()
def perplexity_on_documents(model, tokenizer, documents: list[str], *,
                            device: str = "cpu", max_docs: int | None = None) -> dict:
    """Token-level cross-entropy over whole documents, one at a time.

    Documents longer than the context are scored in non-overlapping chunks, and
    a document shorter than two tokens is skipped rather than contributing a
    degenerate zero.
    """
    model.eval().to(device)
    block = model.config.block_size
    total_nll, total_tokens, total_chars, scored = 0.0, 0, 0, 0
    for doc in documents[:max_docs]:
        ids = tokenizer.encode(doc)
        if len(ids) < 2:
            continue
        scored += 1
        total_chars += len(doc)
        for start in range(0, len(ids) - 1, block):
            chunk = ids[start : start + block + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk[:-1]], device=device)
            y = torch.tensor([chunk[1:]], device=device)
            _, loss = model(x, y)
            total_nll += float(loss.detach()) * y.numel()
            total_tokens += y.numel()
    mean = total_nll / max(total_tokens, 1)
    return {
        "documents": scored,
        "tokens": total_tokens,
        "loss": mean,
        "perplexity": math.exp(mean),
        "bits_per_char": total_nll / math.log(2) / max(total_chars, 1),
    }


def perplexity_by_tag(model, tokenizer, documents: list[str], *, device: str = "cpu",
                      max_docs_per_tag: int = 60) -> dict:
    """The same measurement, split by register."""
    buckets: dict[str, list[str]] = {}
    for doc in documents:
        tag = doc.split(" ", 1)[0] if doc.startswith("<|") else "<|tape|>"
        buckets.setdefault(TAG_NAMES.get(tag, "tape"), []).append(doc)
    return {name: perplexity_on_documents(model, tokenizer, docs, device=device,
                                          max_docs=max_docs_per_tag)
            for name, docs in sorted(buckets.items())}


@torch.no_grad()
def sample_commentary(model, tokenizer, *, prompt: str = "<|commentary|> Market commentary for 2024-06-12.",
                      max_new_tokens: int = 120, temperature: float = 0.8,
                      top_p: float = 0.9, seed: int = 7, device: str = "cpu") -> str:
    """Qualitative check: does it produce something that reads like the register?

    AlphaDesk is a fictional educational simulation. Generated commentary is a
    language-model artefact about fictional issuers — never a market view, never
    advice, and never routed to anything that could act on it.
    """
    model.eval().to(device)
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids[-model.config.block_size :]], device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    out = model.generate(x, max_new_tokens, temperature=temperature, top_p=top_p,
                         generator=g)
    return tokenizer.decode(out[0].tolist())


def compare_models(models: dict, tokenizer, documents: list[str], *, device: str = "cpu",
                   max_docs: int = 80) -> dict:
    """Rung name -> held-out perplexity, plus the pairwise margins."""
    scores = {name: perplexity_on_documents(m, tokenizer, documents, device=device,
                                            max_docs=max_docs)
              for name, m in models.items()}
    names = list(scores)
    margins = {
        f"{a} vs {b}": {
            "loss_delta": scores[a]["loss"] - scores[b]["loss"],
            "perplexity_ratio": scores[a]["perplexity"] / scores[b]["perplexity"],
        }
        for i, a in enumerate(names) for b in names[i + 1 :]
    }
    return {"scores": scores, "margins": margins}
