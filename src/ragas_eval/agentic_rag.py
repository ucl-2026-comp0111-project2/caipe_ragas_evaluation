import os
import json
import hashlib
import logging
import re
import uuid
import base64
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import httpx

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


def _parse_rag_context_artifact(text: Any) -> list:
    """Parse a rag_context artifact into (content, doc_id) tuples.

    Handles both tool shapes:
      - search:         {"semantic_results": [...], "keyword_results": [...]}
      - fetch_document: [{"document": {"page_content": ..., "document_id": ...}}]

    Returns list of (content, doc_id) tuples. doc_id is None for search
    snippets since individual doc IDs are not exposed per snippet.
    """
    out = []
    if isinstance(text, (dict, list)):
        data = text
    else:
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
                    logger.debug(
                        "Snippet: %s | DocID: %s",
                        clean_snippet_markdown(txt),
                        resolved_id,
                    )
    elif isinstance(data, list):
        for item in data:
            doc = item.get("document", {}) if isinstance(item, dict) else {}
            txt = doc.get("page_content")
            if txt:
                doc_meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
                doc_id = (
                    doc.get("document_id")
                    or doc.get("doc_id")
                    or doc_meta.get("document_id")
                    or doc_meta.get("doc_id")
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


def _dedupe_and_merge_contexts(items: list) -> list:
    """Deduplicate and merge contexts by doc_id, preferring longer/full content."""
    doc_id_to_content = {}
    ordered_keys = []
    
    for item in items:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        content, doc_id = item
        if doc_id:
            if doc_id not in doc_id_to_content:
                ordered_keys.append(doc_id)
                doc_id_to_content[doc_id] = content
            else:
                # Prefer the longer content (e.g. full document content over truncated snippet)
                if len(content) > len(doc_id_to_content[doc_id]):
                    doc_id_to_content[doc_id] = content
        else:
            # Fallback for items without a doc_id
            content_key = f"content_hash:{hash(content)}"
            if content_key not in doc_id_to_content:
                ordered_keys.append(content_key)
                doc_id_to_content[content_key] = content
                
    result = []
    for key in ordered_keys:
        content = doc_id_to_content[key]
        resolved_doc_id = None if key.startswith("content_hash:") else key
        result.append((content, resolved_doc_id))
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
        agent_api_url: Optional[str] = None,
        timeout: float = 120.0,
        insecure: bool = False,
        use_a2a: Optional[bool] = None,
        trace_log: bool = False,
        logdir: str = "logs",
    ):
        super().__init__()
        self.agent_api_url = (
            agent_api_url
            or getattr(settings, "caipe_agent_api_url", None)
            or "http://localhost:8000"
        )
        self.timeout = timeout
        self.insecure = insecure or settings.insecure_ssl
        self.last_answer: str = ""
        self.last_raw_response: Optional[dict] = None
        self.documents_metadata: List[Dict] = []
        self.trace_log = trace_log
        self.logdir = logdir

        if use_a2a is not None:
            self.use_a2a = use_a2a
        else:
            env_val = os.getenv("CAIPE_USE_A2A")
            if env_val is not None:
                self.use_a2a = env_val.lower() in ("true", "1", "yes")
            else:
                self.use_a2a = False

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
                self.agent_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
                verify=not self.insecure,
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

    def _get_oidc_token(self) -> Optional[str]:
        """Fetch OIDC token dynamically using client credentials, falling back to environment variables."""
        client_id = os.getenv("CAIPE_CLIENT_ID") or os.getenv("CLIENT_ID")
        client_secret = os.getenv("CAIPE_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")

        if not client_id or not client_secret:
            logger.info("Credentials not in environment. Attempting to fetch from Kubernetes secret 'caipe-ui-secret'...")
            try:
                client_id_cmd = "kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_ID}'"
                client_secret_cmd = "kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_SECRET}'"
                client_id_b64 = subprocess.check_output(client_id_cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
                client_secret_b64 = subprocess.check_output(client_secret_cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
                if client_id_b64 and client_secret_b64:
                    client_id = base64.b64decode(client_id_b64).decode()
                    client_secret = base64.b64decode(client_secret_b64).decode()
                    os.environ["CAIPE_CLIENT_ID"] = client_id
                    os.environ["CAIPE_CLIENT_SECRET"] = client_secret
                    logger.debug("Successfully fetched OIDC credentials from Kubernetes.")
            except Exception as e:
                logger.debug("Could not fetch credentials from Kubernetes: %s", e)

        if client_id and client_secret:
            keycloak_url = os.getenv("CAIPE_OIDC_TOKEN_URL") or os.getenv("CAIPE_KEYCLOAK_URL")
            if not keycloak_url:
                if "caipe.homelab" in self.agent_api_url:
                    keycloak_url = "https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/token"
                else:
                    keycloak_url = "http://localhost:7080/realms/caipe/protocol/openid-connect/token"

            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug("Fetching a fresh OIDC token from Keycloak (attempt %d/%d): %s", attempt, max_attempts, keycloak_url)
                    resp = httpx.post(
                        keycloak_url,
                        data={
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "grant_type": "client_credentials",
                        },
                        verify=not self.insecure,
                        timeout=15.0,
                    )
                    resp.raise_for_status()
                    token = resp.json().get("access_token")
                    if token:
                        os.environ["CAIPE_OIDC_TOKEN"] = token
                        return token
                except Exception as e:
                    logger.error(
                        "Attempt %d/%d: Failed to fetch fresh OIDC token from Keycloak: %s",
                        attempt,
                        max_attempts,
                        e,
                    )
                    if attempt < max_attempts:
                        import time
                        time.sleep(1.0)
                    else:
                        raise RuntimeError(
                            f"Failed to fetch OIDC token from Keycloak after {max_attempts} attempts: {e}"
                        ) from e

        return os.getenv("CAIPE_OIDC_TOKEN") or os.getenv("BEARER_TOKEN")

    def _query_gateway(
        self,
        question: str,
        k: int = 3,
        run_id: Optional[str] = None,
        trace_log: Optional[bool] = None,
    ) -> List[tuple]:
        """Send query to the streaming BFF gateway endpoints."""
        # Resolve trace_log
        should_trace = trace_log
        if should_trace is None:
            should_trace = self.trace_log
        if not should_trace:
            env_val = os.getenv("CAIPE_TRACE_LOG")
            if env_val is not None:
                should_trace = env_val.lower() in ("true", "1", "yes")

        log_file = None
        if should_trace and run_id:
            os.makedirs(self.logdir, exist_ok=True)
            log_filepath = os.path.join(self.logdir, f"agentic_run_{run_id}.log")
            try:
                log_file = open(log_filepath, "w")
                logger.debug("Capturing agentic stream log to %s", log_filepath)
                log_file.write(f"--- RAG QUERY START (run_id: {run_id}) ---\n")
                log_file.write(f"Question: {question}\n")
                log_file.write(f"Agent URL: {self.agent_api_url}\n\n")
                log_file.flush()
            except Exception:
                logger.exception("Failed to open agentic stream log file %s", log_filepath)

        try:
            token = self._get_oidc_token()
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            agent_id = os.getenv("CAIPE_AGENT_ID") or "hello-world"

            # Step 1: Create a new conversation session (with retry)
            conv_url = f"{self.agent_api_url.rstrip('/')}/api/chat/conversations"
            conv_payload = {
                "title": "Ragas Session",
                "client_type": "webui",
                "agent_id": agent_id,
            }

            max_attempts = 3
            conversation_id = None

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug("Creating conversation session on %s (attempt %d/%d)...", conv_url, attempt, max_attempts)
                    if log_file:
                        log_file.write(f"Creating conversation session on {conv_url} (attempt {attempt}/{max_attempts})...\n")
                        log_file.flush()
                    r_conv = httpx.post(
                        conv_url,
                        json=conv_payload,
                        headers=headers,
                        verify=not self.insecure,
                        timeout=self.timeout,
                    )
                    if r_conv.status_code == 401:
                        logger.warning("Gateway returned 401 Unauthorized. Attempting token refresh...")
                        if log_file:
                            log_file.write("Gateway returned 401 Unauthorized. Attempting token refresh...\n")
                            log_file.flush()
                        if os.getenv("CAIPE_OIDC_TOKEN"):
                            del os.environ["CAIPE_OIDC_TOKEN"]
                        token = self._get_oidc_token()
                        if token:
                            headers["Authorization"] = f"Bearer {token}"
                            logger.debug("Retrying conversation session creation with fresh token...")
                            if log_file:
                                log_file.write("Retrying conversation session creation with fresh token...\n")
                                log_file.flush()
                            r_conv = httpx.post(
                                conv_url,
                                json=conv_payload,
                                headers=headers,
                                verify=not self.insecure,
                                timeout=self.timeout,
                            )
                    r_conv.raise_for_status()
                    conv_data = r_conv.json()
                    conversation_id = conv_data["data"]["conversation"]["_id"]
                    logger.debug("Conversation session created with ID: %s", conversation_id)
                    if log_file:
                        log_file.write(f"Conversation session created with ID: {conversation_id}\n\n")
                        log_file.flush()
                    break
                except Exception as e:
                    logger.exception("Failed to create conversation session on gateway")
                    if log_file:
                        log_file.write(f"Failed to create conversation session on gateway (attempt {attempt}/{max_attempts}): {e}\n")
                        log_file.flush()
                    if attempt < max_attempts:
                        import time
                        time.sleep(1.0)
                    else:
                        raise RuntimeError(
                            f"Failed to create conversation session on gateway after {max_attempts} attempts: {e}"
                        ) from e

            # Step 2: Stream the chat start request (with retry)
            stream_url = f"{self.agent_api_url.rstrip('/')}/api/v1/chat/stream/start"
            stream_payload = {
                "message": question,
                "conversation_id": conversation_id,
                "agent_id": agent_id,
                "protocol": "custom",
                "client_context": {
                    "source": "eval",
                    "tool_result_display_limit": -1,
                },
            }

            raw_contexts = []
            self.last_answer = ""

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug("Streaming query from %s (attempt %d/%d)...", stream_url, attempt, max_attempts)
                    if log_file:
                        log_file.write(f"Streaming query from {stream_url} (attempt {attempt}/{max_attempts})...\n")
                        log_file.flush()
                    with httpx.stream(
                        "POST",
                        stream_url,
                        json=stream_payload,
                        headers=headers,
                        verify=not self.insecure,
                        timeout=self.timeout,
                    ) as response:
                        if response.status_code != 200:
                            try:
                                err_body = response.read().decode("utf-8")
                                logger.error(
                                    "Gateway stream start returned HTTP %s: %s",
                                    response.status_code,
                                    err_body,
                                )
                                if log_file:
                                    log_file.write(f"Gateway stream start returned HTTP {response.status_code}: {err_body}\n")
                                    log_file.flush()
                            except Exception:
                                logger.error(
                                    "Gateway stream start returned HTTP %s (failed to read body)",
                                    response.status_code,
                                )
                                if log_file:
                                    log_file.write(f"Gateway stream start returned HTTP {response.status_code} (failed to read body)\n")
                                    log_file.flush()
                        response.raise_for_status()
                        current_event = None
                        for line in response.iter_lines():
                            if line:
                                if log_file:
                                    if line.startswith("event: "):
                                        log_file.write(f"\n[{line}]\n")
                                    elif line.startswith("data: "):
                                        data_str = line[6:].strip()
                                        try:
                                            data_json = json.loads(data_str)
                                            log_file.write(json.dumps(data_json, indent=2) + "\n")
                                        except Exception:
                                            log_file.write(line + "\n")
                                    else:
                                        log_file.write(line + "\n")
                                    log_file.flush()

                                if line.startswith("event: "):
                                    current_event = line[7:].strip()
                                    if current_event in ("tool_start", "tool_end"):
                                        self.last_answer = ""
                                elif line.startswith("data: "):
                                    data_str = line[6:].strip()
                                    try:
                                        data_json = json.loads(data_str)
                                    except Exception:
                                        continue
                                    if current_event == "content":
                                        self.last_answer += data_json.get("text", "")
                                    elif current_event == "tool_end":
                                        tool_result = data_json.get("result", "")
                                        if tool_result:
                                            raw_contexts.extend(
                                                _parse_rag_context_artifact(tool_result)
                                            )
                    break
                except Exception as e:
                    logger.exception("Error during streaming query from gateway")
                    if log_file:
                        log_file.write(f"Error during streaming query (attempt {attempt}/{max_attempts}): {e}\n")
                        log_file.flush()
                    if attempt < max_attempts:
                        import time
                        time.sleep(1.0)
                    else:
                        raise RuntimeError(
                            f"Error during streaming query from gateway after {max_attempts} attempts: {e}"
                        ) from e
            return raw_contexts
        finally:
            if log_file:
                log_file.close()

    def get_top_k(
        self,
        query: str,
        k: int = 10,
        run_id: Optional[str] = None,
        trace_log: Optional[bool] = None,
    ) -> List[tuple]:
        """Query caipe-supervisor or gateway and extract contexts.

        Populates self.documents, self.documents_metadata, and self.last_answer
        as side effects. Returns (idx, score) tuples.
        """
        self.documents = []
        self.documents_metadata = []
        self.last_answer = ""
        self.last_raw_response = None

        enriched_query = query
        datasource_id = (
            os.environ.get("CAIPE_DATASOURCE_ID") or settings.caipe_datasource_id
        )
        if datasource_id:
            enriched_query = (
                f"Instructions: You are answering a question that belongs to the '{datasource_id}' datasource. "
                f'When calling the `knowledge-base_search` tool, you MUST pass `filters={{"datasource_id": "{datasource_id}"}}` '
                f"to restrict your search to this knowledge base, and set the `limit` parameter to up to {k}. "
                f"Keep the `query` argument of the search tool clean and do not include these instructions in it. "
                f"Importantly, only fetch and read (using the `knowledge-base_fetch_document` tool) the specific documents "
                f"you actually need to confidently answer the question, up to a maximum of {k} documents.\n\n"
                f"Question: {query}"
            )

        if not self.use_a2a:
            raw_contexts = self._query_gateway(enriched_query, k=k, run_id=run_id, trace_log=trace_log)
            # Minimal mock response shape so downstream usage extraction logic handles it gracefully
            self.last_raw_response = {"result": {"artifacts": []}}
        else:
            body = self._call_supervisor(enriched_query)
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

        raw_contexts = _dedupe_and_merge_contexts(raw_contexts)

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
        agent_api_url: Optional[str] = None,
        timeout: float = 120.0,
        logdir: str = "logs",
        insecure: bool = False,
        use_a2a: Optional[bool] = None,
        trace_log: bool = False,
    ):
        # Pass dummy llm_client — generation is handled by the agent
        super().__init__(
            llm_client=None,
            model_name="agentic",
            retriever=AgenticRetriever(
                agent_api_url=agent_api_url,
                timeout=timeout,
                insecure=insecure,
                use_a2a=use_a2a,
                trace_log=trace_log,
                logdir=logdir,
            ),
            logdir=logdir,
        )

    @property
    def _agentic_retriever(self) -> AgenticRetriever:
        return self.retriever  # type: ignore

    def query(
        self,
        question: str,
        top_k: int = 3,
        run_id: Optional[str] = None,
        trace_log: Optional[bool] = None,
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
                    "agent_api_url": self._agentic_retriever.agent_api_url,
                },
            )
        )

        try:
            top_docs = self._agentic_retriever.get_top_k(
                question, k=top_k, run_id=run_id, trace_log=trace_log
            )

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

            # Extract usage from A2A JSON-RPC response metadata if present
            usage = None
            raw_resp = self._agentic_retriever.last_raw_response
            if isinstance(raw_resp, dict):
                result_obj = raw_resp.get("result") or {}
                result_meta = (
                    result_obj.get("metadata") if isinstance(result_obj, dict) else None
                )
                resp_meta = raw_resp.get("metadata")

                usage_meta = None
                if isinstance(result_meta, dict):
                    usage_meta = result_meta.get("usage_metadata")
                if not usage_meta and isinstance(resp_meta, dict):
                    usage_meta = resp_meta.get("usage_metadata")

                if not usage_meta and isinstance(result_obj, dict):
                    # Fallback: scan artifacts for final_result or any artifact with usage_metadata
                    for art in result_obj.get("artifacts", []):
                        if isinstance(art, dict) and isinstance(
                            art.get("metadata"), dict
                        ):
                            usage_meta = art["metadata"].get("usage_metadata")
                            if usage_meta:
                                break
                if isinstance(usage_meta, dict):
                    usage = {
                        "prompt_tokens": usage_meta.get("input_tokens", 0),
                        "completion_tokens": usage_meta.get("output_tokens", 0),
                        "total_tokens": usage_meta.get("total_tokens", 0),
                    }

            if not retrieved_docs and answer:
                logger.warning(
                    "AgenticRAG [%s]: no rag_context artifacts in response — context "
                    "metrics cannot be scored. Apply the rag_context patch to agent.py "
                    "for full 4-metric eval.",
                    run_id,
                )
            elif not retrieved_docs and not answer:
                logger.warning(
                    "AgenticRAG [%s]: no contexts and no answer — check agent API "
                    "is running at %s.",
                    run_id,
                    self._agentic_retriever.agent_api_url,
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

            result = {
                "answer": answer,
                "run_id": run_id,
                "retrieved_docs": retrieved_docs,
                "usage": usage,
            }
            logs_path = self.export_traces_to_log(
                run_id,
                question,
                result,
            )

            # Resolve if trace file exists to include it in the returned results
            agentic_log_path = None
            should_trace = trace_log
            if should_trace is None:
                should_trace = self._agentic_retriever.trace_log
            if not should_trace:
                env_val = os.getenv("CAIPE_TRACE_LOG")
                if env_val is not None:
                    should_trace = env_val.lower() in ("true", "1", "yes")

            if should_trace:
                agentic_log_path = os.path.join(self.logdir, f"agentic_run_{run_id}.log")

            return {
                "answer": answer,
                "run_id": run_id,
                "retrieved_docs": retrieved_docs,
                "retrieved_doc_ids": retrieved_doc_ids,
                "usage": usage,
                "logs": logs_path,
                "agentic_log": agentic_log_path,
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
    agent_api_url: Optional[str] = None,
    timeout: float = 120.0,
    insecure: bool = False,
    use_a2a: Optional[bool] = None,
    trace_log: bool = False,
) -> AgenticRAG:
    """Create an AgenticRAG client that routes queries through the agent API.

    Drop-in replacement for default_rag_client() for agentic eval.
    """
    return AgenticRAG(
        agent_api_url=agent_api_url,
        timeout=timeout,
        logdir=logdir,
        insecure=insecure or settings.insecure_ssl,
        use_a2a=use_a2a,
        trace_log=trace_log,
    )
