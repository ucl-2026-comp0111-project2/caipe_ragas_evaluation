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


import json
import logging
import re
import string
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import SingleTurnMetric, MetricType

logger = logging.getLogger(__name__)

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


@dataclass
class MeanReciprocalRank(SingleTurnMetric):
    name: str = "mrr"
    _required_columns: dict = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: set()
        }
    )

    def init(self, run_config: Optional[object] = None) -> None:
        """No-op: Deterministic calculation requires no LLM or embeddings."""

    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Optional[object] = None,
    ) -> float:
        # Extract fields out of the rubrics dictionary
        rubrics = getattr(sample, "rubrics", None) or {}
        
        try:
            retrieved_ids = json.loads(rubrics.get("retrieved_doc_ids", "[]"))
            expected_ids = set(json.loads(rubrics.get("expected_doc_ids", "[]")))
        except Exception:
            logger.warning("Failed to parse retrieved or expected doc IDs from rubrics.")
            return 0.0

        # Sanitize items to strings
        retrieved_ids = [str(d) for d in retrieved_ids]
        expected_ids = {str(d) for d in expected_ids}

        if not expected_ids:
            return 0.0

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                return 1.0 / rank
        return 0.0


@dataclass
class NDCGAtK(SingleTurnMetric):
    name: str = "ndcg_at_k"
    k: int = 5
    _required_columns: dict = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: set()
        }
    )

    def init(self, run_config: Optional[object] = None) -> None:
        """No-op: Deterministic calculation requires no LLM or embeddings."""

    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Optional[object] = None,
    ) -> float:
        rubrics = getattr(sample, "rubrics", None) or {}
        
        try:
            retrieved_ids = json.loads(rubrics.get("retrieved_doc_ids", "[]"))
            expected_ids = set(json.loads(rubrics.get("expected_doc_ids", "[]")))
        except Exception:
            logger.warning("Failed to parse retrieved or expected doc IDs from rubrics.")
            return 0.0

        retrieved_ids = [str(d) for d in retrieved_ids]
        expected_ids = {str(d) for d in expected_ids}

        if not expected_ids or not retrieved_ids:
            logger.warning("No valid doc IDs found in rubrics.")
            return 0.0

        retrieved_k = retrieved_ids[:self.k]
        
        dcg = sum((1.0 / np.log2(i + 2)) for i, doc_id in enumerate(retrieved_k) if doc_id in expected_ids)
        if dcg <= 0.0:
            return 0.0

        ideal_hits = min(len(expected_ids), self.k)
        idcg = sum((1.0 / np.log2(i + 2)) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0.0 else 0.0