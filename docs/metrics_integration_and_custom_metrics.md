# Metrics Integration & Custom Metrics Guide

This document explains how evaluation metrics are integrated into the RAG Evaluation System, how the execution engine interacts with them, and how you can define and add your own custom metrics.

---

## 1. Ragas Metrics Integration

The system uses **Ragas 0.3.5** to evaluate RAG outputs across multiple quality dimensions.

### Standard Metrics Used

* **FactualCorrectness**: Measures semantic alignment of the generated answer with the ground-truth reference.
* **Faithfulness**: Measures whether the generated answer is strictly grounded in the retrieved contexts (detecting hallucinations).
* **AnswerRelevancy**: Measures how relevant the generated answer is to the original question (uses embeddings).
* **ContextPrecision**: Evaluates whether the retrieved documents that are most relevant are ranked at the top.
* **ContextRecall**: Verifies if the retrieved context is sufficient to answer the reference ground-truth.
* **Retrieval Recall**: Custom exact document ID recall, measuring the ratio of correct reference document IDs retrieved in the top-k results.
* **Retrieval Precision**: Custom exact document ID precision, measuring the ratio of retrieved document IDs that are correct ground-truth reference documents.

### Initialization & Execution Flow

1. **Initialization ([`_init_metrics`](../src/ragas_eval/evals.py) in `evals.py`)**:
   Metrics are selected based on the active run mode flags (`--retrieval-only`, `--generation-only`, `--short-answer`):

   ```python
   def _init_metrics(config, ragas_llm, ragas_embeddings):
       retrieval_only = config["rag_eval_retrieval_only"]
       generation_only = config["rag_eval_generation_only"]
       short_answer = config.get("rag_eval_short_answer", False)

       if retrieval_only:
           return [
               ContextPrecision(llm=ragas_llm),
               ContextRecall(llm=ragas_llm),
           ]
       if generation_only:
           return [
               FactualCorrectness(llm=ragas_llm),
               AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
           ]
       # default full stack
       return [
           FactualCorrectness(llm=ragas_llm),
           Faithfulness(llm=ragas_llm),
           AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
           ContextPrecision(llm=ragas_llm),
           ContextRecall(llm=ragas_llm),
       ]
   ```

2. **Evaluation Execution ([`_run_evaluation`](../src/ragas_eval/evals.py) in `evals.py`)**:
   Converts datasets into Ragas `EvaluationDataset` / `SingleTurnSample` structures and calls `evaluate()`.

3. **Resilience & Key Normalization**:
   Ragas expects precise JSON outputs from the LLM evaluator (e.g. `claims`, `statements`, `verdict`). To prevent crashes due to formatting issues, a custom interceptor `patched_ragas_evaluator_llm_create` intercepts the LLM output, repairs the JSON structure using `json_repair`, and normalizes schemas/values.

---

## 2. Creating Custom Metrics

Ragas 0.3 supports defining custom metrics by subclassing `SingleTurnMetric`. This is the same approach used for `ContainsAnswer` in this project.

### Subclassing `SingleTurnMetric`

Subclass `SingleTurnMetric` from `ragas.metrics.base` and implement `_single_turn_ascore`. This is the correct Ragas 0.3.5 API — do **not** use the older `_ascore` / `EvaluationMode` API.

The `ContainsAnswer` metric in [`src/ragas_eval/metrics.py`](../src/ragas_eval/metrics.py) is a working example:

```python
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
    name: str = "contains_answer"
    _required_columns: dict = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"response", "reference"}
        }
    )

    def init(self, run_config=None) -> None:
        """No-op: ContainsAnswer requires no LLM or embeddings."""

    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks=None,
    ) -> float:
        response = sample.response or ""
        reference = sample.reference or ""

        if not reference.strip():
            return 0.0

        norm_response = _normalize(response)
        norm_reference = _normalize(reference)

        return 1.0 if norm_reference in norm_response else 0.0
```

Key implementation notes:

- **`_required_columns`** — declares which `SingleTurnSample` fields the metric needs. Use the `MetricType.SINGLE_TURN` enum member as the key.
- **`init()`** — required abstract method. No-op for metrics that need no LLM or embeddings.
- **`_single_turn_ascore()`** — the scoring method. Must be `async` and return a `float`.
- **Import from `ragas.metrics`** directly, never from `ragas.metrics.collections` — see the [Ragas import constraint KI](../README.md).

### Registering Your Custom Metric

Add an instance to the list returned by `_init_metrics` in [`evals.py`](../src/ragas_eval/evals.py):

```python
from ragas_eval.metrics import ContainsAnswer
from ragas.metrics import SemanticSimilarity

# Inside _init_metrics, for the --short-answer flag branch:
if short_answer:
    return [
        SemanticSimilarity(embeddings=ragas_embeddings),
        ContainsAnswer(),
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextRecall(llm=ragas_llm),
    ]
```

---

## 3. HotpotQA Metric Compatibility

### The Problem: Short-Answer Reference Mismatch

Standard Ragas metrics assume **long-form, paragraph-style reference answers**. They extract semantic claims from the reference using an LLM and compare those claims via NLI. This breaks for **HotpotQA-style datasets** where references are short factual strings:

| Question | Reference |
|---|---|
| "Were Scott Derrickson and Ed Wood of the same nationality?" | `"yes"` |
| "What government position was held by Shirley Temple?" | `"Chief of Protocol"` |

**Affected metrics:**

| Metric | Why it breaks |
|---|---|
| `FactualCorrectness` | NLI finds no claims to extract from `"yes"` → always 0.0 |
| `ContextPrecision` | Depends on reference claims to rank context relevance → always 0.0 |
| `ContextRecall` | Partially affected — claim extraction on short references is unreliable |

### Recommended Metric Stack for HotpotQA

| Metric | Source | Reliable? | Notes |
|---|---|---|---|
| `SemanticSimilarity` | `ragas.metrics` | ✅ | Embedding cosine sim — handles verbose answers matching short refs |
| `ContainsAnswer` | `ragas_eval.metrics` | ✅ | Substring match after normalisation (see below) |
| `Faithfulness` | `ragas.metrics` | ✅ | Response ↔ context NLI — unaffected by ref length |
| `AnswerRelevancy` | `ragas.metrics` | ✅ | Response ↔ question — unaffected by ref length |
| `ContextRecall` | `ragas.metrics` | ⚠️ | Use with caution; still more reliable than FactualCorrectness |
| `retrieval_recall` | Custom (in `_analyze_failures`) | ✅ | Best retrieval signal for HotpotQA — uses `expected_doc_ids` |
| `FactualCorrectness` | `ragas.metrics` | ❌ | Do not use for HotpotQA |
| `ContextPrecision` | `ragas.metrics` | ❌ | Do not use for HotpotQA |

### The `ContainsAnswer` Custom Metric

Implemented in [`src/ragas_eval/metrics.py`](../src/ragas_eval/metrics.py).

Checks whether the **normalised** reference answer is a **substring** of the **normalised** model response.
Normalisation: lowercase → strip articles (`a`/`an`/`the`) → strip punctuation → collapse whitespace.

```
"Yes, both Scott Derrickson and Ed Wood were American."  →  "yes both scott derrickson ed wood were american"
"yes"                                                    →  "yes"
Result: 1.0  ✅
```

```
"Based on the documents, I cannot determine Shirley Temple's government role."
"chief of protocol"
Result: 0.0  ❌  (correctly flags retrieval/generation failure)
```

This mirrors the official HotpotQA evaluation script's substring-match correctness check.

### The `PrecomputedRAG` Retrieval Query Fix

`PrecomputedRAG` was designed to query CAIPE using the **reference answer** as a semantic proxy — valid for long-form references that overlap with source documents.

For HotpotQA short answers, querying CAIPE with `"yes"` returns random irrelevant documents.

**Fix** ([`precomputed_rag.py` L124](../src/ragas_eval/precomputed_rag.py)):

```python
# Use a combined query of question + reference to retrieve relevant docs from CAIPE.
# This leverages the semantic context from both, which works for both long-form references
# and short factual answers (e.g. HotpotQA "yes"/"no").
reference_query = f"{data['question']} {data['reference']}".strip()
```

By joining the question and reference, we ensure a strong retrieval signal for all dataset formats. For yes/no references, the question dominates. For short entities, both query terms contribute. For long-form responses, the reference context provides the primary signal.

### Eval Mode Guidance for HotpotQA

| Mode | Command | Suitable? |
|---|---|---|
| **Retrieval only** | `--retrieval-only` | ✅ Best fit — measures multi-hop retrieval quality |
| **End-to-end (short-answer metrics)** | `--short-answer` | ✅ Correct metric stack for HotpotQA |
| **End-to-end (default metrics)** | _(no flags)_ | ❌ `FactualCorrectness` scores 0.0 on short references |
| **Model answer eval** | `--compute-model-eval` | ❌ Not meaningful — gold answer as model output gives trivial correctness |

### Using `--short-answer`

Pass `--short-answer` whenever evaluating against a dataset with **short factual references** (single words or short phrases). It can also be set persistently in `.env` alongside your datasource config:

```bash
# .env — hotpotqa block
CAIPE_DATASOURCE_ID="hotpotqa_sample"
RAGAS_DATASOURCE="hotpotqa"
QUESTIONS_PATH="data/hotpotqa_full_questions.jsonl"
RAG_EVAL_SHORT_ANSWER=true        # activates short-answer metric stack automatically
```

Or passed explicitly at runtime:

```bash
# End-to-end evaluation with short-answer metric stack
./scripts/run_eval.sh --short-answer

# Retrieval-only (no metric change needed — context metrics are unaffected by ref length)
./scripts/run_eval.sh --retrieval-only

# Short-answer + generation-only
./scripts/run_eval.sh --short-answer --generation-only
```

The flag is **dataset-agnostic** — it works for HotpotQA, TriviaQA, SQuAD, or any other short-answer benchmark, without being tied to a specific datasource name.

---

## 4. Legacy Embedding Compatibility Issues

### The Problem
During evaluation execution, some metrics (primarily older or custom Ragas/LangChain components) expect the legacy LangChain-compatible `Embeddings` interface, requiring:
- `embed_query(text: str) -> list[float]`
- `embed_documents(texts: list[str]) -> list[list[float]]`

Conversely, modern Ragas 0.3.5 embedding factories (like `embedding_factory`) produce instances conforming to the modern `BaseRagasEmbedding` interface, requiring:
- `embed_text(text: str) -> list[float]`
- `aembed_text(text: str) -> list[float]` (asynchronous)

If a synchronous client or custom engine is wrapped directly in Ragas, calling `aembed_text` will throw a `TypeError` or `AttributeError` during runtime, crashing the evaluation pipeline.

### The Solution: `LegacyEmbeddingsWrapper`
To bridge these interfaces and prevent runtime crashes, we implement the `LegacyEmbeddingsWrapper` decorator inside [`src/ragas_eval/evals.py`](../src/ragas_eval/evals.py):

```python
class LegacyEmbeddingsWrapper:
    """Wrapper class to adapt Ragas embeddings to LangChain-compatible interface."""

    def __init__(self, emb: Any):
        self.emb = emb

    def embed_query(self, text: str) -> List[float]:
        return self.emb.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.emb.embed_texts(texts)

    def embed_text(self, text: str) -> List[float]:
        return self.emb.embed_text(text)

    async def aembed_text(self, text: str) -> List[float]:
        try:
            return await self.emb.aembed_text(text)
        except (TypeError, AttributeError):
            return self.emb.embed_text(text)  # safe sync fallback
```

### Regression Enforcement (Unit Tests)
To prevent developers or AI assistants from accidentally removing or breaking this wrapper in future updates, the pipeline contract is guarded by the following unit tests in [`tests/test_evals_units.py`](../tests/test_evals_units.py):
- `test_legacy_embeddings_wrapper_methods`: Asserts standard LangChain forward mapping methods compile and match correct signatures.
- `test_legacy_embeddings_wrapper_methods_negative`: Asserts empty/invalid input checks and exceptions.
- `test_legacy_embeddings_wrapper_aembed_standard`: Verifies async execution when a modern async-compatible embedding client is present.
- `test_legacy_embeddings_wrapper_aembed_fallback`: Verifies synchronous fallback catching when a synchronous embedding client is wrapped, avoiding uncaught `TypeError` crashes.

---

## 5. Sample Evaluation Results

When running evaluations, the console output and saved summary JSONs (e.g. `{experiment_name}_summary.json`) present a breakdown of both operational metrics and quality indicators.

### Sample Console Output

```text
--- RUN CONFIGURATION ---
top_k: 5
retrieval_only: False
generation_only: False
limit_per_category: 10
compute_model_eval: False
short_answer: True


--- OPERATIONAL BEHAVIOR ---
P50 Latency: 2.30s
P95 Latency: 3.78s
Average Tokens: 902.5
Ragas Evaluator LLM Token Usage:
  Prompt Tokens: 15854
  Completion Tokens: 2196
  Total Evaluator Tokens: 18050

--- QUALITY METRICS AVERAGE ---
Average semantic_similarity: 0.58
Average contains_answer: 0.85
Average faithfulness: 0.92
Average answer_relevancy: 0.67
Average context_recall: 0.70
Average retrieval_recall: 0.82
Average retrieval_precision: 0.34

--- FAILURE CAUSE ANALYSIS ---
failure_cause
none              13
poor_retrieval     6
hallucination      1
Name: count, dtype: int64
```

### Explaining the Metrics

1. **Operational Metrics**:
   * **Latency (P50/P95)**: Tracks user response speeds (50% and 95% of queries finished within 2.30s and 3.78s respectively).
   * **Average Tokens**: Average tokens used by the RAG generation pipeline.
   * **Ragas Evaluator Tokens**: Tracks the prompt and completion tokens consumed by the LLM running the Ragas evaluations (very helpful for cost monitoring).

2. **Quality Metrics**:
   * **semantic_similarity (0.58)**: Compares the generated answer semantic similarity to the reference ground truth (scale: `0.0` to `1.0`).
   * **contains_answer (0.85)**: Verifies whether the generated answer contains the normalized text of the ground-truth short answer.
   * **faithfulness (0.92)**: High score indicates generated answers are heavily grounded in retrieved contexts, preventing hallucinations.
   * **answer_relevancy (0.67)**: Evaluates if the answer is directly addressing the question without going off-topic.
   * **context_recall (0.70)**: Evaluates RAG contexts retrieved against the reference answer.
   * **retrieval_recall (0.82)**: Measures exact document ID retrieval recall (82% of ground truth reference documents were retrieved in the top-k results).
   * **retrieval_precision (0.34)**: Measures exact document ID retrieval precision (34% of retrieved document IDs in the top-k results were correct ground truth reference documents).

3. **Failure Cause Analysis**:
   * Classifies query runs into `none` (successful), `poor_retrieval` (reference documents missed during retrieval phase), and `hallucination` (reference retrieved but generated answer contains ungrounded info).

