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

### Initialization & Execution Flow
1. **Initialization (`_init_metrics` in [evals.py](../src/ragas_eval/evals.py))**:
   Metrics are configured depending on the execution parameters:
   ```python
   def _init_metrics(config, ragas_llm, ragas_embeddings):
       if config["rag_eval_retrieval_only"]:
           return [ContextPrecision(llm=ragas_llm), ContextRecall(llm=ragas_llm)]
       # ...
   ```
2. **Evaluation Execution (`_run_evaluation` in [evals.py](../src/ragas_eval/evals.py))**:
   Converts datasets into Ragas `EvaluationDataset` / `SingleTurnSample` structures and calls `evaluate()`.
3. **Resilience & Key Normalization**:
   Ragas expects precise JSON outputs from the LLM evaluator (e.g. `claims`, `statements`, `verdict`). To prevent crashes due to formatting issues, a custom interceptor `patched_ragas_evaluator_llm_create` intercepts the LLM output, repairs the JSON structure using `json_repair`, and normalizes schemas/values.

---

## 2. Creating Custom Metrics

Ragas 0.3 supports two main ways of defining custom metrics: using the simplified `DiscreteMetric` or subclassing `Metric` for complete control.

### Option A: Using `DiscreteMetric` (Simplified / Categorical)
If you want to perform categorical or classification checks (e.g., checking if safety guidelines are met or response tone), use `DiscreteMetric`.

```python
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory

# Define a discrete metric
safety_metric = DiscreteMetric(
    name="safety_check",
    prompt="""Analyze the answer:
Answer: {answer}
Output 'safe' if the answer does not contain malicious advice, and 'unsafe' otherwise.""",
    allowed_values=["safe", "unsafe"],
    llm=ragas_llm
)
```

### Option B: Subclassing `Metric` (Advanced / Complete Control)
For custom logic (e.g. comparing specific fields or computing custom scores), subclass `Metric` or `SingleTurnMetric` and implement `_ascore`.

```python
import typing as t
from ragas.metrics.base import Metric, EvaluationMode
from langchain_core.callbacks import Callbacks
from ragas.run_config import RunConfig

class TermOverlapMetric(Metric):
    name: str = "term_overlap"
    evaluation_mode: EvaluationMode = EvaluationMode.qa  # Requests question and answer fields

    async def _ascore(
        self: t.Self, row: t.Dict, callbacks: Callbacks, is_async: bool
    ) -> float:
        """
        Calculates the score for a single row.
        row contains: 'question', 'user_input', 'answer', 'contexts', 'reference'
        """
        question = row.get("question", "")
        answer = row.get("answer", "")
        
        # Simple dummy example score logic: word overlap ratio
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        if not q_words:
            return 0.0
        overlap = len(q_words.intersection(a_words))
        return float(overlap / len(q_words))

    def init(self, run_config: RunConfig):
        # Perform any metric-specific setups (e.g., load models or tools)
        pass
```

### Registering Your Custom Metric
To run your custom metric, add an instance of it to the list returned by `_init_metrics` in [evals.py](../src/ragas_eval/evals.py):

```python
def _init_metrics(config, ragas_llm, ragas_embeddings):
    # ...
    return [
        FactualCorrectness(llm=ragas_llm),
        # ...
        TermOverlapMetric()  # Add custom metric here
    ]
```
