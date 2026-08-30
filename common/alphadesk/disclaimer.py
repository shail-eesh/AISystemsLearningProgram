"""The one string every AlphaDesk surface must show."""

DISCLAIMER = (
    "AlphaDesk is a fictional educational simulation. No real orders, no real money, "
    "no brokerage connectivity, no redistribution of licensed market data."
)


def banner(width: int = 88) -> str:
    """A boxed disclaimer for CLI surfaces."""
    inner = width - 4
    words, lines, cur = DISCLAIMER.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > inner:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    rule = "+" + "-" * (width - 2) + "+"
    body = "\n".join(f"| {ln.ljust(inner)} |" for ln in lines)
    return f"{rule}\n{body}\n{rule}"
