import json
import hashlib
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from ragas_eval.config import settings
from ragas_eval.rag import BaseRetriever, BaseRAG, TraceEvent

logger = logging.getLogger(__name__)

# ============================================================
# Agentic retrieval — queries caipe-supervisor's A2A endpoint
# instead of rag-server directly. Requires the rag_context
# patch applied to agent.py for full context metric scoring.
# ============================================================

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_SNIPPET_PREFIX_RE = re.compile(r"^\s*\*\*Snippet:\*\*\s*")
_ELLIPSIS_RE = re.compile(r"\.{3,}")


def clean_snippet_markdown(text: str) -> str:
    """Strip bold/ellipsis display markup from search tool snippets.

    The search tool returns UI-formatted snippets e.g.
    '**Snippet:** ...**CAIPE** uses nomic-embed-text...'.
    Stripping gives Ragas plain prose and avoids WAF 403s on the
    Outshift proxy when snippets are embedded in judge prompts.
    """
    if not text:
        return text
    cleaned = _SNIPPET_PREFIX_RE.sub("", text)
    cleaned = _ELLIPSIS_RE.sub(" ", cleaned)
    cleaned = _BOLD_RE.sub(r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_text_from_parts(parts: list) -> str:
    """Concatenate A2A response parts into a single string."""
    return "".join(p.get("text", "") for p in parts if p.get("kind") == "text")


def _parse_rag_context_artifact(text: str) -> list:
    """Parse a rag_context artifact into (content, doc_id) tuples.

    Handles both tool shapes:
      - search:         {"semantic_results": [...], "keyword_results": [...]}
      - fetch_document: [{"document": {"page_content": ..., "document_id": ...}}]

    Returns list of (content, doc_id) tuples. doc_id is None for search
    snippets since individual doc IDs are not exposed per snippet.
    """
    out = []
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return out

    if isinstance(data, dict):
        for key in ("semantic_results", "keyword_results"):
            for item in data.get(key, []) or []:
                txt = item.get("text_content")
                if txt:
                    # Extract document_id from the search metadata block where available.
                    meta = (
                        item.get("metadata", {})
                        if isinstance(item.get("metadata"), dict)
                        else {}
                    )
                    doc_id = (
                        meta.get("document_id")
                        or meta.get("doc_id")
                        or item.get("document_id")
                    )
                    # Convert to string before appending so doc IDs are consistently typed.
                    resolved_id = str(doc_id) if doc_id is not None else None
                    out.append((clean_snippet_markdown(txt), resolved_id))
                    logger.info(
                        "Snippet: %s | DocID: %s",
                        clean_snippet_markdown(txt),
                        resolved_id,
                    )
    elif isinstance(data, list):
        for item in data:
            doc = item.get("document", {}) if isinstance(item, dict) else {}
            txt = doc.get("page_content")
            if txt:
                doc_id = (
                    doc.get("document_id")
                    or doc.get("doc_id")
                    or item.get("document_id")
                    or item.get("doc_id")
                )
                out.append((txt, str(doc_id) if doc_id is not None else None))
    return out


def _dedupe_preserve_order(items: list) -> list:
    """Deduplicate (content, doc_id) tuples by content, preserving order."""
    seen = set()
    result = []
    for item in items:
        content = item[0] if isinstance(item, tuple) else item
        if content not in seen:
            seen.add(content)
            result.append(item)
    return result


class AgenticRetriever(BaseRetriever):
    """Retriever that queries caipe-supervisor's A2A endpoint.

    Routes queries through the agent instead of rag-server directly.
    Captures both retrieved contexts (from rag_context artifacts) and
    the generated answer (from final_result artifact) in a single call.

    Requires the rag_context patch applied to agent.py for context metrics.
    """

    def __init__(
        self,
        supervisor_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__()
        self.supervisor_url = (
            supervisor_url
            or getattr(settings, "caipe_supervisor_url", None)
            or "http://localhost:8000"
        )
        self.timeout = timeout
        self.last_answer: str = ""
        self.last_raw_response: Optional[dict] = None
        self.documents_metadata: List[Dict] = []

    def fit(self, documents: List[str]):
        """AgenticRetriever doesn't support local fitting."""
        self.documents = documents
        self.documents_metadata = [{} for _ in documents]

    def _call_supervisor(self, question: str) -> Optional[dict]:
        """Send a question to caipe-supervisor's A2A message/send endpoint."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": question}],
                    "messageId": str(uuid.uuid4()),
                }
            },
        }
        try:
            response = requests.post(
                self.supervisor_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.error(
                "Timeout calling caipe-supervisor (%.1fs) — increase --supervisor-timeout",
                self.timeout,
            )
            return None
        except requests.HTTPError as exc:
            logger.error(
                "HTTP %s from caipe-supervisor: %s",
                exc.response.status_code,
                exc,
            )
            return None
        except Exception:
            logger.exception("Unexpected error calling caipe-supervisor A2A endpoint")
            return None

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Query caipe-supervisor and extract contexts from rag_context artifacts.

        Populates self.documents, self.documents_metadata, and self.last_answer
        as side effects. Returns (idx, score) tuples — scores are 1.0 since
        the agent doesn't expose per-chunk scores in the A2A response.
        """
        self.documents = []
        self.documents_metadata = []
        self.last_answer = ""
        self.last_raw_response = None

        body = self._call_supervisor(query)
        if not body:
            return []

        self.last_raw_response = body
        artifacts = body.get("result", {}).get("artifacts", [])

        raw_contexts = []
        for art in artifacts:
            name = art.get("name", "")
            text = _extract_text_from_parts(art.get("parts", []))
            if name == "rag_context":
                raw_contexts.extend(_parse_rag_context_artifact(text))
            elif name == "final_result":
                self.last_answer = text

        raw_contexts = _dedupe_preserve_order(raw_contexts)[:k]

        for content, doc_id in raw_contexts:
            self.documents.append(content)
            self.documents_metadata.append({"doc_id": doc_id} if doc_id else {})

        return [(i, 1.0) for i in range(len(self.documents))]


class AgenticRAG(BaseRAG):
    """RAG pipeline that uses caipe-supervisor for both retrieval and generation.

    Collapses retrieval and generation into a single A2A call. Both the
    retrieved contexts and the generated answer are captured from the response.
    Use instead of BaseRAG when evaluating the full agentic pipeline end-to-end.

    Requires the rag_context patch applied to agent.py for context metrics.
    Without it, only answer-quality metrics (factual_correctness, answer_relevancy)
    can be scored.
    """

    def __init__(
        self,
        supervisor_url: Optional[str] = None,
        timeout: float = 120.0,
        logdir: str = "logs",
    ):
        # Pass dummy llm_client — generation is handled by the agent
        super().__init__(
            llm_client=None,
            model_name="agentic",
            retriever=AgenticRetriever(supervisor_url=supervisor_url, timeout=timeout),
            logdir=logdir,
        )

    @property
    def _agentic_retriever(self) -> AgenticRetriever:
        return self.retriever  # type: ignore

    def query(
        self, question: str, top_k: int = 3, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Single A2A call returns both contexts and answer.

        Returns same dict shape as BaseRAG.query() for drop-in compatibility.
        """
        if run_id is None:
            _q_hash = int(hashlib.md5(question.encode()).hexdigest(), 16) % 10000
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_q_hash:04d}"

        self.traces = []
        self.traces.append(
            TraceEvent(
                event_type="query_start",
                component="agentic_rag",
                data={
                    "run_id": run_id,
                    "question": question,
                    "supervisor_url": self._agentic_retriever.supervisor_url,
                },
            )
        )

        try:
            top_docs = self._agentic_retriever.get_top_k(question, k=top_k)

            retrieved_docs = [
                {
                    "content": self._agentic_retriever.documents[idx],
                    "similarity_score": score,
                    # Use real KB doc_id from fetch_document results where available;
                    # fall back to positional index for search snippets.
                    "document_id": (
                        self._agentic_retriever.documents_metadata[idx].get("doc_id")
                        if idx < len(self._agentic_retriever.documents_metadata)
                        and self._agentic_retriever.documents_metadata[idx].get(
                            "doc_id"
                        )
                        else idx
                    ),
                    "metadata": (
                        self._agentic_retriever.documents_metadata[idx]
                        if idx < len(self._agentic_retriever.documents_metadata)
                        else {}
                    ),
                }
                for idx, score in top_docs
                if idx < len(self._agentic_retriever.documents)
            ]

            # Expose doc IDs for _analyze_failures retrieval_recall/precision scoring
            retrieved_doc_ids = [doc["document_id"] for doc in retrieved_docs]
            answer = self._agentic_retriever.last_answer

            if not retrieved_docs and answer:
                logger.warning(
                    "AgenticRAG [%s]: no rag_context artifacts in response — context "
                    "metrics cannot be scored. Apply the rag_context patch to agent.py "
                    "for full 4-metric eval.",
                    run_id,
                )
            elif not retrieved_docs and not answer:
                logger.warning(
                    "AgenticRAG [%s]: no contexts and no answer — check caipe-supervisor "
                    "is running at %s.",
                    run_id,
                    self._agentic_retriever.supervisor_url,
                )

            self.traces.append(
                TraceEvent(
                    event_type="query_complete",
                    component="agentic_rag",
                    data={
                        "run_id": run_id,
                        "success": True,
                        "num_retrieved": len(retrieved_docs),
                        "answer_length": len(answer),
                    },
                )
            )

            logs_path = self.export_traces_to_log(
                run_id,
                question,
                {
                    "answer": answer,
                    "retrieved_docs": retrieved_docs,
                },
            )

            return {
                "answer": answer,
                "run_id": run_id,
                "retrieved_docs": retrieved_docs,
                "retrieved_doc_ids": retrieved_doc_ids,
                "usage": None,
                "logs": logs_path,
            }

        except Exception as e:
            logger.exception(
                "AgenticRAG [%s]: unhandled exception during query", run_id
            )
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="agentic_rag",
                    data={"run_id": run_id, "error": str(e)},
                )
            )
            logs_path = self.export_traces_to_log(run_id, question, None)
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
                "retrieved_docs": [],
                "retrieved_doc_ids": [],
                "usage": None,
                "logs": logs_path,
            }


def default_agentic_rag_client(
    logdir: str = "logs",
    supervisor_url: Optional[str] = None,
    timeout: float = 120.0,
) -> AgenticRAG:
    """Create an AgenticRAG client that routes queries through caipe-supervisor.

    Drop-in replacement for default_rag_client() for agentic eval.
    """
    return AgenticRAG(supervisor_url=supervisor_url, timeout=timeout, logdir=logdir)
