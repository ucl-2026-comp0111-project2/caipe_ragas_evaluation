"""
Custom Ragas metrics for short-answer datasets (e.g. HotpotQA).

Standard Ragas metrics (FactualCorrectness, ContextPrecision) use LLM-based
NLI claim comparison, which breaks when the reference is a short factual answer
("yes", "Chief of Protocol") rather than a long paragraph.

Metrics defined here:
  - ContainsAnswer: checks whether the normalised reference answer appears
    anywhere in the normalised model response (substring match).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Optional

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import SingleTurnMetric, MetricType


_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> str:
    """Lowercase, strip articles, punctuation, and collapse whitespace."""
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


@dataclass
class ContainsAnswer(SingleTurnMetric):
    """
    Short-answer correctness metric: 1.0 if the normalised reference answer
    is a substring of the normalised model response, else 0.0.

    Designed for HotpotQA-style datasets where references are short factual
    strings ("yes", "American", "Chief of Protocol") and the model may produce
    verbose answers ("Yes, both were American directors.").

    Requires: response, reference.
    Does NOT require an LLM or embeddings.
    """

    name: str = "contains_answer"
    _required_columns: dict = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"response", "reference"}
        }
    )

    def init(self, run_config: Optional[object] = None) -> None:
        """No-op: ContainsAnswer requires no LLM or embeddings."""

    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Optional[object] = None,
    ) -> float:
        response = sample.response or ""
        reference = sample.reference or ""

        if not reference.strip():
            return 0.0

        norm_response = _normalize(response)
        norm_reference = _normalize(reference)

        return 1.0 if norm_reference in norm_response else 0.0
