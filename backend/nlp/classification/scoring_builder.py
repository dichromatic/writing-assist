# Diagram omitted - utility module with no significant information flow.

"""
Mutable helper for assembling classification evidence.

Classification scorers all follow the same bookkeeping pattern: accumulate a
numeric score, append reasons and vetoes in branch order, clamp the final
score, and materialize a ``ClassEvidence`` result. This helper centralizes
that bookkeeping so scorer modules can focus on the branch conditions that are
actually category-specific.
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.types import LexiconCategory


class ScoringBuilder:
    """Accumulate score, reasons, and vetoes for one category scorer.

    Args:
        category: Classification category being assembled.
    """

    def __init__(self, category: LexiconCategory) -> None:
        self._category = category
        self._score = 0.0
        self._reasons: list[str] = []
        self._vetoes: list[str] = []

    @property
    def score(self) -> float:
        """Return the current un-clamped score."""
        return self._score

    def add(self, score: float, reason: str) -> None:
        """Increase the score and append its supporting reason."""
        self._score += score
        self._reasons.append(reason)

    def set(self, score: float, reason: str | None = None) -> None:
        """Replace the score, optionally appending the branch reason.

        This is used for branches whose score is authoritative rather than
        additive, such as the base group-suffix heuristic.
        """
        self._score = score
        if reason is not None:
            self._reasons.append(reason)

    def veto(self, reason: str) -> None:
        """Append a veto without changing the current score."""
        self._vetoes.append(reason)

    def veto_and_zero(self, reason: str) -> None:
        """Append a veto and clear the score entirely."""
        self._vetoes.append(reason)
        self._score = 0.0

    def cap(self, ceiling: float) -> None:
        """Clamp the running score downward without adding new explanations."""
        self._score = min(self._score, ceiling)

    def merge(self, score: float, reasons: list[str], vetoes: list[str]) -> None:
        """Merge precomputed score parts while preserving list order."""
        self._score += score
        self._reasons.extend(reasons)
        self._vetoes.extend(vetoes)

    def build(self) -> ClassEvidence:
        """Materialize the final evidence record with standard score clamping."""
        return ClassEvidence(
            category=self._category,
            score=min(self._score, 1.0),
            reasons=self._reasons,
            vetoes=self._vetoes,
        )
