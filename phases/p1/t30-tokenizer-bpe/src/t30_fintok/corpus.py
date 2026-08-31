"""The corpora FinTok is trained on, and the control it is measured against.

Same policy as `common/data`: **offline first, nothing licensed redistributed.**
Both corpora are generated deterministically from templates and a fixed seed,
with fictional issuers, so the whole experiment reproduces on any machine with
no network and no rights questions.

Two corpora, because the claim being tested is about *domain*, not about
tokenizers in general:

* :func:`financial_corpus` — filing sections, market commentary, exchange
  announcements, order tickets, earnings-call fragments.
* :func:`general_corpus` — ordinary English prose about food, travel, weather,
  sport and biography, sharing as little vocabulary with the first as possible.

Training the *same algorithm* at the *same vocabulary size* on each and then
measuring both on held-out financial text isolates exactly one variable.

> The master plan names tiktoken's GPT-2 vocabulary as the intended external
> benchmark. `bench/compression.py` will use it when `tiktoken` is installed and
> can reach its vocabulary files; in this sandbox the download host is blocked,
> so the controlled two-corpus comparison above is what actually runs. It is the
> better-designed experiment anyway — GPT-2 differs from FinTok in corpus,
> vocabulary size *and* pre-tokenizer at once.
"""

from __future__ import annotations

import random

ISSUERS = [
    ("ALPHAINFRA", "Alpha Infrastructure Limited", "INE001A01011", "infrastructure"),
    ("BHARATCHEM", "Bharat Chemicals Limited", "INE002A01018", "specialty chemicals"),
    ("COASTBANK", "Coastal Bank Limited", "INE003A01025", "banking"),
    ("DECCANMOT", "Deccan Motors Limited", "INE004A01032", "automotive"),
    ("EASTPOWER", "Eastern Power Grid Limited", "INE005A01049", "power transmission"),
]

FILING_SECTIONS = [
    "Management's discussion and analysis of financial condition and results of operations",
    "Quantitative and qualitative disclosures about market risk",
    "Risk factors relating to the business and the industry",
    "Notes to the consolidated financial statements",
    "Report of the independent registered public accounting firm",
    "Related party transactions and balances outstanding",
]

RISK_CLAUSES = [
    "adverse changes in commodity prices may materially affect our margins",
    "our results may fluctuate as a result of foreign currency exchange rate movements",
    "we depend on a limited number of customers for a substantial portion of revenue",
    "regulatory changes in the markets in which we operate could increase compliance costs",
    "interruptions in the supply chain could delay deliveries and reduce revenue",
    "we may be unable to refinance our borrowings on commercially acceptable terms",
    "cybersecurity incidents could disrupt operations and expose us to liability",
]

COMMENTARY = [
    "opened firm and held the gains through the afternoon session",
    "drifted lower on thin volumes ahead of the policy announcement",
    "outperformed the benchmark index for the third consecutive session",
    "reversed early losses after the management commentary on margins",
    "traded in a narrow band with delivery volumes below the twenty-day average",
    "saw sustained institutional buying interest in the second half",
]

ANNOUNCEMENTS = [
    "intimation of board meeting to consider unaudited financial results",
    "disclosure under Regulation 30 of the Listing Obligations and Disclosure Requirements",
    "certificate under Regulation 74(5) of the Depositories and Participants Regulations",
    "outcome of board meeting held today at the registered office of the company",
    "record date fixed for the purpose of payment of interim dividend",
]

# The control corpus is generated from word banks rather than a handful of fixed
# sentences: a control with 200 distinct words would be trivially easy to
# compress and would flatter FinTok for the wrong reason. These banks give the
# general corpus a lexical diversity in the same range as the financial one,
# which is what makes the comparison mean anything.
GEN_NOUNS = ["morning", "kitchen", "river", "garden", "harbour", "village", "orchard", "meadow", "window", "doorway", "station", "platform", "carriage", "ferry", "lantern", "kettle", "skillet", "flour", "butter", "pepper", "garlic", "lemon", "mountain", "valley", "footpath", "ridge", "shoreline", "lighthouse", "cottage", "terrace", "courtyard", "chapel", "notebook", "letter", "postcard", "photograph", "album", "bicycle", "rucksack", "compass", "blanket", "sweater", "teacher", "gardener", "carpenter", "fisherman", "baker", "printer", "sailor", "weaver", "potter", "joiner", "summer", "autumn", "winter", "spring", "afternoon", "evening", "midnight", "daybreak", "twilight", "season", "cousin", "neighbour", "stranger", "visitor", "apprentice", "companion", "daughter", "grandmother", "uncle", "harvest", "festival", "market", "bakery", "library", "workshop", "bookshop", "pharmacy", "laundry", "brewery", "sparrow", "heron", "otter", "badger", "hedgehog", "swallow", "kestrel", "salmon", "beetle", "butterfly", "rainfall", "sunshine", "breeze", "thunder", "frost", "drizzle", "hailstorm", "humidity", "horizon", "shadow"]
GEN_VERBS = ["walked", "carried", "gathered", "opened", "folded", "stirred", "simmered", "kneaded", "painted", "mended", "climbed", "waited", "noticed", "remembered", "forgot", "borrowed", "returned", "promised", "whispered", "argued", "laughed", "listened", "wandered", "rested", "arrived", "departed", "settled", "unpacked", "repaired", "planted", "pruned", "harvested", "watered", "swept", "polished", "sharpened", "measured", "sketched", "practised", "taught", "learned", "copied", "translated", "recited", "hummed", "rowed", "sailed", "cycled", "hiked"]
GEN_ADJECTIVES = ["quiet", "patient", "stubborn", "cheerful", "damp", "golden", "narrow", "crooked", "ancient", "tidy", "restless", "generous", "careless", "fragrant", "bitter", "salty", "crisp", "tender", "sturdy", "brittle", "distant", "familiar", "peculiar", "ordinary", "splendid", "modest", "gloomy", "radiant", "windy", "misty", "scattered", "steady", "sudden", "gentle", "fierce", "shallow", "crowded", "empty", "faded", "polished"]
GEN_ADVERBS = ["slowly", "carefully", "suddenly", "quietly", "cheerfully", "reluctantly", "often", "rarely", "already", "almost", "hardly", "nearly", "perfectly", "badly", "quickly", "patiently", "gradually", "briefly"]
GEN_PLACES = ["Ashford", "Brackenhill", "Coldwater", "Dunmore", "Elmsworth", "Fernleigh", "Greystones", "Hollowfield", "Ivybridge", "Kestrelton", "Larkmead", "Mossgate", "Northwick", "Oakhaven", "Pinefall", "Quarrytown", "Redmarsh", "Stonebrook", "Thornbury", "Uppercross", "Westerly", "Yarrowdale"]
GEN_NAMES = ["Aline", "Bertram", "Cecily", "Duncan", "Edith", "Fergus", "Greta", "Hamish", "Imogen", "Jonas", "Kirsty", "Lachlan", "Marta", "Niall", "Orla", "Piers", "Rowena", "Silas", "Tessa", "Uist", "Verity", "Wilfred"]

GEN_TEMPLATES = [
    "The {adj} {noun} near {place} was {adv} {verb} by {name} that {noun2}.",
    "{name} {verb} the {adj} {noun} and left the {noun2} on the {noun3}.",
    "By {adv} the {noun} had turned {adj}, and nobody in {place} {verb} at all.",
    "There was a {adj} {noun} beyond the {noun2}, {adv} {verb} since the {noun3}.",
    "{name} and {name2} {verb} together every {noun}, whatever the {noun2}.",
    "It is {adj} work, {verb} the {noun} before the {noun2} turns {adj2}.",
    "The road out of {place} runs past a {adj} {noun} and a {adj2} {noun2}.",
    "Nobody {verb} much that {noun}; the {noun2} was {adj} and the {noun3} {adj2}.",
    "{adv}, {name} {verb} the {noun2} and never mentioned the {noun} again.",
    "A {adj} {noun} arrived from {place}, {adv} {verb} and addressed to {name}.",
]


def financial_corpus(*, docs: int = 900, seed: int = 30) -> list[str]:
    """Synthetic financial text: filings, commentary, announcements, tickets."""
    rng = random.Random(seed)
    out: list[str] = []
    for i in range(docs):
        sym, name, isin, sector = rng.choice(ISSUERS)
        year = rng.choice([2022, 2023, 2024])
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        date = f"{year}-{month:02d}-{day:02d}"
        price = round(rng.uniform(80, 3200), 2)
        qty = rng.choice([50, 100, 250, 500, 1000, 2500, 10000])
        pct = round(rng.uniform(-4.5, 4.5), 2)
        crore = round(rng.uniform(12, 8400), 1)

        kind = i % 4
        if kind == 0:
            body = (
                f"{rng.choice(FILING_SECTIONS)}. {name} ({sym}, ISIN {isin}) reported "
                f"consolidated revenue of Rs {crore} crore for the quarter ended {date}, "
                f"an increase of {abs(pct)}% over the corresponding period of the previous year. "
                f"EBITDA margin stood at {round(rng.uniform(6, 28), 1)}% against "
                f"{round(rng.uniform(6, 28), 1)}% a year earlier. "
                f"Risk factors: {rng.choice(RISK_CLAUSES)}; {rng.choice(RISK_CLAUSES)}. "
                f"The {sector} segment contributed {round(rng.uniform(20, 95), 1)}% of "
                f"consolidated revenue during the period under review."
            )
        elif kind == 1:
            body = (
                f"Market commentary for {date}. {sym} {rng.choice(COMMENTARY)}, "
                f"settling at Rs {price} ({pct:+.2f}%) on turnover of Rs {crore} crore. "
                f"The stock traded between Rs {round(price * 0.985, 2)} and "
                f"Rs {round(price * 1.014, 2)} with delivery at "
                f"{round(rng.uniform(18, 72), 1)}% of traded quantity. "
                f"{rng.choice(ISSUERS)[0]} and {rng.choice(ISSUERS)[0]} were the other "
                f"notable movers in the {sector} space."
            )
        elif kind == 2:
            body = (
                f"NSE/CM/{rng.randint(10000, 99999)} dated {date}: {name} - "
                f"{rng.choice(ANNOUNCEMENTS)}. Symbol: {sym}. ISIN: {isin}. "
                f"Series: EQ. The company has informed the exchange that the board "
                f"will meet on {year}-{(month % 12) + 1:02d}-{day:02d} to consider and "
                f"approve the unaudited financial results for the quarter ended {date}."
            )
        else:
            side = rng.choice(["BUY", "SELL"])
            body = (
                f"Order ticket {rng.randint(100000, 999999)}: {side} {qty} {sym} "
                f"@ LIMIT {price} TIF DAY, routed {date} 09:{rng.randint(15, 59)}:"
                f"{rng.randint(10, 59)}. Filled {qty} at average price "
                f"{round(price * rng.uniform(0.999, 1.001), 2)}, "
                f"realised slippage {round(rng.uniform(-8, 8), 2)} bps against arrival. "
                f"Risk check: notional Rs {round(price * qty / 100000, 2)} lakh within limit."
            )
        out.append(body)
    return out


def general_corpus(*, docs: int = 900, seed: int = 31) -> list[str]:
    """Ordinary English, deliberately sharing little vocabulary with the above.

    Generated by slot-filling from word banks so the control has comparable
    lexical diversity. A control with a few hundred distinct words would be
    trivially compressible and would make FinTok look good for the wrong reason.
    """
    rng = random.Random(seed)

    def sentence() -> str:
        t = rng.choice(GEN_TEMPLATES)
        return t.format(
            adj=rng.choice(GEN_ADJECTIVES), adj2=rng.choice(GEN_ADJECTIVES),
            noun=rng.choice(GEN_NOUNS), noun2=rng.choice(GEN_NOUNS), noun3=rng.choice(GEN_NOUNS),
            verb=rng.choice(GEN_VERBS), adv=rng.choice(GEN_ADVERBS),
            place=rng.choice(GEN_PLACES), name=rng.choice(GEN_NAMES), name2=rng.choice(GEN_NAMES),
        )

    return [" ".join(sentence() for _ in range(rng.randint(6, 12))) for _ in range(docs)]


def split(texts: list[str], *, holdout: float = 0.15) -> tuple[list[str], list[str]]:
    """Deterministic train/holdout split — the compression claim must be on unseen text."""
    cut = int(len(texts) * (1 - holdout))
    return texts[:cut], texts[cut:]


def corpus_stats(texts: list[str]) -> dict:
    joined = "".join(texts)
    return {
        "docs": len(texts),
        "chars": len(joined),
        "bytes": len(joined.encode("utf-8")),
        "distinct_whitespace_words": len({w for t in texts for w in t.split()}),
    }
