# RAG Evaluation System: Design and Data Flow Documentation

This document explains the architecture, execution pipeline, and data flow of the RAG (Retrieval-Augmented Generation) Evaluation system.

---

## 1. Architecture Overview

The system consists of three main components:
1. **Knowledge Ingestion Pipeline**: Ingests test corpus documents (such as Confluence, Jira, and Slack datasets) into a CAIPE RAG Knowledge Base.
2. **RAG Client Pipeline**: Connects to the CAIPE knowledge base to retrieve context and generates answers using a configured LLM.
3. **Ragas Evaluation & Analysis Engine**: Orchestrates execution across test datasets, runs evaluations using multi-dimensional Ragas metrics, and provides operational and failure cause analysis.

```mermaid
graph TD
    A[Data Ingestion] -->|Populates| B[CAIPE Knowledge Base]
    C[Dataset / Questions] -->|Triggers Pipeline| D[RAG Pipeline]
    B -->|Context Retrieval| D
    D -->|Generates Answers & Latency| E[Ragas Evaluation Engine]
    E -->|Grades Output| F[Reports & Failures Analysis]
    F -->|Outputs CSV/JSON| G[(Experiment Results)]
```

---

## 2. Component Design & Code Structure

* **Configuration Management ([config.py](../config.py))**: Consolidates settings such as LLM models, endpoints, datasource IDs, retrieval limits, and evaluation parameters using Pydantic Settings.
* **Document Ingestion ([enterprise_rag_bench_ingest.py](../enterprise_rag_bench_ingest.py))**: Downloads dataset files, obtains OIDC tokens, and uploads batches to CAIPE.
* **RAG Pipeline Implementation ([rag.py](../rag.py))**: Core RAG system implementing:
  * [CaipeRetriever](../rag.py#L80): Queries CAIPE Knowledge Base using OIDC authentication.
  * [SimpleKeywordRetriever](../rag.py#L50): Keyword matching fallback.
  * [BaseRAG](../rag.py#L216): Coordinates document retrieval, prompt formulation, LLM completion, and telemetry tracing.
* **Evaluation Orchestrator ([evals.py](../evals.py))**: Evaluates RAG performance, computes quality metrics, and handles LLM output formatting repairs via custom monkey-patches.

---

## 3. End-to-End Data Flow

### Phase 1: Ingestion Flow
1. The ingestion script reads the slice configurations and downloads the source dataset `.zip` files from GitHub Releases.
2. It calls `kubectl` to fetch OIDC Client ID/Secret credentials from Keycloak secret mappings.
3. An access token is requested from Keycloak OIDC.
4. Documents are batched and POSTed to `/v1/ingest` under the target `datasource_id`.

```mermaid
sequenceDiagram
    participant Script as enterprise_rag_bench_ingest.py
    participant K8s as Kubernetes API
    participant KC as Keycloak OIDC
    participant CAIPE as CAIPE Ingestion API
    
    Script->>K8s: Fetch client secrets
    K8s-->>Script: Return Client ID & Secret
    Script->>KC: POST /openid-connect/token (credentials)
    KC-->>Script: Return Access Token
    Script->>Script: Parse and batch Zip documents
    loop For each batch
        Script->>CAIPE: POST /v1/ingest (with token)
        CAIPE-->>Script: HTTP 200 (Success)
    end
```

### Phase 2: RAG Pipeline execution
1. `run_eval.sh` retrieves the Keycloak OIDC token from the Kubernetes secret store and exports it as `CAIPE_OIDC_TOKEN`.
2. `evals.py` reads test samples from `enterprise_rag_bench_questions.jsonl`.
3. For each query, `BaseRAG` triggers [retrieve_documents](../rag.py#L341) to fetch context from CAIPE. The retriever uses the token supplied by `settings.caipe_oidc_token` (derived from the `CAIPE_OIDC_TOKEN` environment variable).
4. The retrieved contexts are formatted into the system prompt template.
5. The RAG application requests a completion from the generation model (via LiteLLM / OpenAI proxy) to output the answer.
6. The latency, raw tokens, retrieved document IDs, and logs are tracked.


```mermaid
sequenceDiagram
    participant Eval as evals.py
    participant RAG as "rag.py (BaseRAG)"
    participant CAIPE as "CAIPE Query API (/v1/query)"
    participant LLM as LiteLLM Server
    
    Eval->>RAG: query(question)
    RAG->>CAIPE: POST /v1/query (limit=k)
    CAIPE-->>RAG: Return retrieved documents & doc_ids
    RAG->>RAG: Format context prompt
    RAG->>LLM: chat.completions.create (user prompt)
    LLM-->>RAG: Return generated answer + usage stats
    RAG-->>Eval: Return answer, retrieved contexts, doc_ids, latency, tokens
```

### Phase 3: Ragas Evaluation, Output Repair & Diagnostics
1. Ragas `evaluate` is invoked across the batched evaluation dataset.
2. During the evaluation process, Ragas calls the evaluation LLM to analyze correctness and faithfulness.
3. **Output Repairing**: A custom interceptor monkey-patch (`patched_ragas_evaluator_llm_create`) catches JSON responses, corrects formatting issues, normalizes keys (e.g. `claims`, `statements`, `verdict`), and sanitizes values (e.g., converting yes/no/null outputs into integers) to prevent Ragas validation failures.
4. **Failure Cause Analysis**:
   * Evaluates standard metrics: `FactualCorrectness`, `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, and `ContextRecall`.
   * Evaluates exact document overlap: checks `retrieved_doc_ids` against `expected_doc_ids` to calculate `retrieval_recall`.
   * Uses metric score thresholds (scores < 0.5) to categorize root-cause failures:
     * `ContextRecall < 0.5` $\rightarrow$ **poor_retrieval**
     * `Faithfulness < 0.5` $\rightarrow$ **hallucination**
     * `FactualCorrectness < 0.5` $\rightarrow$ **incorrect_generation**
5. All results are consolidated and saved as a CSV experiment run sheet (with averages summary) and a summary JSON.

```mermaid
flowchart TD
    subgraph Evaluation Loop
        E1[Build EvaluationDataset] --> E2[Run ragas.evaluate]
        E2 --> E3{Ragas LLM Call?}
        E3 -->|Yes| E4[patched_ragas_evaluator_llm_create]
        E4 --> E5[json_repair & normalize keys/values]
        E5 --> E2
        E2 --> E6[Receive Metric Scores]
    end
    
    subgraph Analysis & Diagnostic Phase
        E6 --> D1[Calculate exact Retrieval Recall]
        D1 --> D2{"ContextRecall < 0.5?"}
        D2 -->|Yes| F1[poor_retrieval]
        D2 -->|No| D3{"Faithfulness < 0.5?"}
        D3 -->|Yes| F2[hallucination]
        D3 -->|No| D4{"FactualCorrectness < 0.5?"}
        D4 -->|Yes| F3[incorrect_generation]
        D4 -->|No| F4[none]
    end
    
    F1 & F2 & F3 & F4 --> S1[Compile Averages]
    S1 --> S2[Save CSV and JSON reports]
```

---

## 4. Key Metrics Summary

The engine generates the following metrics:
* **Latency (P50 & P95)**: Tracks generation and retrieval speed.
* **Token Usage**: Measures prompt and completion tokens for both the RAG application and the Ragas evaluator.
* **FactualCorrectness**: Verifies semantic alignment of generated answer with the ground-truth reference.
* **Faithfulness**: Measures whether the generated answer is grounded in the retrieved documents (checking for hallucinations).
* **AnswerRelevancy**: Measures how relevant the generated answer is to the original question.
* **ContextPrecision**: Checks if the most relevant retrieved documents are ranked at the top.
* **ContextRecall**: Verifies whether all ground-truth reference statements can be answered using the retrieved documents.
* **Retrieval Recall (Custom)**: Compares retrieved doc IDs against expected doc IDs for direct vector search quality.

---

## 5. Detailed Subsystem Documentation

For more in-depth explanations on the system's core capabilities, see:
* **[Metrics Integration & Custom Metrics Guide](metrics_integration_and_custom_metrics.md)**: Explains built-in Ragas metrics and how to implement custom evaluators.
* **[Retrieval & Answering Integration Guide](retrieval_and_answering_integration.md)**: Explains retrieval architectures (`CaipeRetriever`) and generation flows (`BaseRAG`).

