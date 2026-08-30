"""Exception hierarchy.

.NET translation: there is no checked-exception ceremony and no
`ArgumentNullException` culture. Python's convention is a small hierarchy
rooted at one package exception so callers can `except OrderError:` and catch
everything this module raises.
"""

from __future__ import annotations


class OrderError(Exception):
    """Base for every error this package raises."""


class ValidationError(OrderError, ValueError):
    """A domain invariant was violated (the C# ctor-guard equivalent).

    Inheriting from ValueError too means callers who only know the stdlib
    hierarchy still catch it — Python leans on structural expectations more
    than .NET does.
    """


class RiskLimitBreached(OrderError):
    """A pre-trade risk check rejected the order. Paper trading only."""
