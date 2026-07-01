# RAG Evaluation

Evaluate a CAIPE RAG system end-to-end using [Ragas](https://docs.ragas.io) metrics. Supports multiple benchmark datasources, configurable eval modes, and custom metrics for short-answer datasets.

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure via `.env`

Copy the relevant block into `.env` for your datasource:

```bash
# OpenAI / LiteLLM gateway
OPENAI_API_KEY="your-key"
OPENAI_ENDPOINT="http://localhost:4000/v1"
OPENAI_MODEL_NAME="your-model"
EMBEDDINGS_MODEL="your-embeddings-model"

# CAIPE Knowledge Base
CAIPE_QUERY_ENDPOINT="http://localhost:9446/v1/query"

# --- Datasource: enterprise_rag_bench ---
CAIPE_DATASOURCE_ID="enterprise_rag_bench"
RAGAS_DATASOURCE="enterprise_rag_bench"
QUESTIONS_PATH="data/enterprise_rag_bench_questions.jsonl"
# RAG_EVAL_SHORT_ANSWER=false   # default — uses FactualCorrectness

# --- Datasource: hotpotqa ---
# CAIPE_DATASOURCE_ID="hotpotqa_sample"
# RAGAS_DATASOURCE="hotpotqa"
# QUESTIONS_PATH="data/hotpotqa_full_questions.jsonl"
# RAG_EVAL_SHORT_ANSWER=true    # use SemanticSimilarity + ContainsAnswer for short answers
```

### 3. Ingest Data

```bash
# Ingest Enterprise RAG Bench
./scripts/ingest_enterprise_rag_bench.sh

# Ingest HotpotQA
./scripts/ingest_hotpotqa.sh
```

### 4. Run Evaluation

```bash
# Standard end-to-end evaluation
./scripts/run_eval.sh

# HotpotQA — short-answer metric stack
./scripts/run_eval.sh --short-answer

# Retrieval quality only (context_precision + context_recall)
./scripts/run_eval.sh --retrieval-only

# Generation quality only (no context metrics)
./scripts/run_eval.sh --generation-only

# Evaluate pre-existing model answers from the datasource
./scripts/run_model_eval.sh
```

## Evaluation Flags

| Flag | Description |
|---|---|
| `--short-answer` | Swap `FactualCorrectness` → `SemanticSimilarity + ContainsAnswer`. Required for short-answer datasets (HotpotQA, TriviaQA). Can also be set via `RAG_EVAL_SHORT_ANSWER=true` in `.env`. |
| `--retrieval-only` | Run only retrieval metrics (`context_precision`, `context_recall`). |
| `--generation-only` | Run only generation metrics (skip context metrics). |
| `--compute-model-eval` | Use `PrecomputedRAG` — evaluates pre-existing reference answers as model output. Best for `enterprise_rag_bench`. |
| `--top-k N` | Number of documents to retrieve (default: 3). |
| `--limit-per-category N` | Cap questions per category (useful for quick smoke tests). |
| `--datasource NAME` | Override the active datasource at runtime. |

## Datasource Guide

| Datasource | `RAGAS_DATASOURCE` | `--short-answer` | Best eval mode |
|---|---|---|---|
| Enterprise RAG Bench | `enterprise_rag_bench` | No | End-to-end or `--compute-model-eval` |
| HotpotQA | `hotpotqa` | **Yes** | End-to-end `--short-answer` or `--retrieval-only` |

## Metric Reference

### Default metric stack

Used when `--short-answer` is **not** set:

| Metric | What it measures |
|---|---|
| `factual_correctness` | Answer accuracy vs. reference (NLI claim comparison) |
| `faithfulness` | Answer grounded in retrieved context (hallucination detection) |
| `answer_relevancy` | Answer relevance to the question |
| `context_precision` | Most relevant docs ranked at top |
| `context_recall` | Retrieved context covers the reference |

### Short-answer metric stack (`--short-answer`)

Used for HotpotQA and similar short-answer benchmarks:

| Metric | What it measures |
|---|---|
| `semantic_similarity` | Embedding cosine similarity between response and reference |
| `contains_answer` | Normalised reference is a substring of the response |
| `faithfulness` | Answer grounded in retrieved context |
| `answer_relevancy` | Answer relevance to the question |
| `context_recall` | Retrieved context covers the reference |

> `factual_correctness` and `context_precision` are excluded — they rely on LLM NLI claim extraction which produces 0.0 on short references like `"yes"` or `"Chief of Protocol"`. See [docs/metrics_integration_and_custom_metrics.md](docs/metrics_integration_and_custom_metrics.md) for full explanation.

## Project Structure

```
rag_eval/
├── README.md
├── pyproject.toml
├── .env                          # Active configuration (gitignored)
├── data/                         # Benchmark question JSONL files
├── docs/                         # Design docs and metric guides
├── scripts/
│   ├── run_eval.sh               # Main evaluation runner
│   ├── run_model_eval.sh         # Pre-computed answer evaluation
│   ├── ingest_enterprise_rag_bench.sh
│   └── ingest_hotpotqa.sh
├── src/ragas_eval/
│   ├── config.py                 # Pydantic settings (env → config)
│   ├── evals.py                  # Core evaluation engine
│   ├── rag.py                    # RAG pipeline (CaipeRetriever)
│   ├── precomputed_rag.py        # PrecomputedRAG for model eval
│   ├── metrics.py                # Custom metrics (ContainsAnswer)
│   ├── hotpotqa_rag_ingest.py    # HotpotQA ingestion pipeline
│   └── enterprise_rag_bench_ingest.py
└── evals/
    ├── experiments/              # CSV + JSON results per run
    └── logs/                     # RAG trace logs per query
```

## Documentation

- [Metrics Integration & Custom Metrics](docs/metrics_integration_and_custom_metrics.md)
- [RAG Eval Design](docs/rag_eval_design_docs.md)
- [Retrieval & Answering Integration](docs/retrieval_and_answering_integration.md)
- [Ragas Documentation](https://docs.ragas.io)
