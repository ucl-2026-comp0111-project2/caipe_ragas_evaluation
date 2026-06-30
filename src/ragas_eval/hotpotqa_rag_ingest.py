"""
Ingest a sample of the HotpotQA dataset into a local CAIPE RAG server via its
public REST API, then run sample queries to verify retrieval.

Flow: ingestor heartbeat -> create/replace datasource -> create job
      -> POST /v1/ingest (batched) -> mark job completed -> /v1/query.
      Also writes a golden question set for whatever rows were fetched, with expected_doc_ids aligned to the
      same document_id hashing used at ingestion time.

Usage:
    python3 scripts/hotpotqa_rag_ingest.py --limit 1000

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import time
from typing import Any

import requests

HF_DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "hotpotqa/hotpot_qa"
HF_PAGE_SIZE = 100  # HF datasets-server's max `length` per request

INGESTOR_TYPE = "hotpotqa"
INGESTOR_NAME = "hotpotqa-eval-script"
DEFAULT_QUESTIONS_FILE = "data/hotpotqa_full_questions.jsonl"

logger = logging.getLogger(__name__)


def _get_oidc_token(
    oidc_token_url: str, client_id: str | None = None, client_secret: str | None = None
) -> str:
    """Fetch OIDC token for CAIPE ingestion service using client credentials"""
    if not client_id or not client_secret:
        raise ValueError(
            "Both client_id and client_secret must be provided to fetch OIDC token "
            "when not explicitly providing a token."
        )
    try:
        response = requests.post(
            oidc_token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception:
        logger.exception("Error fetching token")
        raise


def _check(resp: requests.Response) -> requests.Response:
    """Raise with the response body printed, for easier debugging of 4xx/5xx."""
    if not resp.ok:
        logger.error(f"\nHTTP {resp.status_code} for {resp.request.method} {resp.request.url}")
        logger.error(resp.text)
        resp.raise_for_status()
    return resp


def _setup_session(args: argparse.Namespace) -> requests.Session:
    """Set up requests.Session with optional OIDC auto-refresh authentication."""
    session = requests.Session()
    if not args.use_oidc:
        # No Authorization header is sent on this session - on a local dev stack with
        # CAIPE_UNSAFE_RBAC_BYPASS=true this is what makes rag-server treat the
        # caller as an admin.
        return session

    token = getattr(args, "oidc_token", None)
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
        logger.info("Provided OIDC token set successfully.")
    else:
        logger.warning(
            "No initial OIDC token provided. Authentication token will be fetched "
            "dynamically using client credentials on the first request."
        )

    # Set up auto-refresh wrapper for session requests on 401 Unauthorized
    oidc_token_url = args.oidc_token_url
    oidc_client_id = args.oidc_client_id
    oidc_client_secret = args.oidc_client_secret

    orig_request = session.request

    def auto_refresh_request(*req_args, **req_kwargs):
        """Requests wrapper to automatically refresh expired OIDC tokens upon 401 Unauthorized status."""
        is_retry = req_kwargs.pop("_is_retry", False)
        resp = orig_request(*req_args, **req_kwargs)
        if resp.status_code == 401 and not is_retry:
            logger.warning("Token Expired. Fetching a new OIDC token and retrying...")
            try:
                new_token = _get_oidc_token(
                    oidc_token_url,
                    oidc_client_id,
                    oidc_client_secret,
                )
                session.headers.update({"Authorization": f"Bearer {new_token}"})
                if "headers" in req_kwargs and "Authorization" in req_kwargs["headers"]:
                    req_kwargs["headers"]["Authorization"] = f"Bearer {new_token}"
                req_kwargs["_is_retry"] = True
                resp = orig_request(*req_args, **req_kwargs)
            except Exception:
                logger.exception("Failed to refresh OIDC token")
        return resp

    session.request = auto_refresh_request
    return session


def _wait_seconds_from_headers(resp: requests.Response, attempt: int) -> float:
    """
    Prefer the server's own guidance on how long to wait, if present
    (standard `Retry-After`, or HF's IETF-draft `RateLimit` header e.g.
    '"api";r=0;t=189' where t= seconds until reset). Falls back to
    exponential backoff (capped higher than before, since HF's windows
    are 5 minutes) if no usable header is found.
    """
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    ratelimit = resp.headers.get("RateLimit") or resp.headers.get("ratelimit")
    if ratelimit and "t=" in ratelimit:
        try:
            t_part = ratelimit.split("t=")[1].split(";")[0].split(",")[0]
            return float(t_part)
        except (ValueError, IndexError):
            pass

    # Fallback: exponential backoff, capped at 60s. HF rate-limit windows
    # are 5 minutes, so a short fixed cap can still fail repeatedly — this
    # is only used when the server gives us no explicit guidance.
    return min(2 ** attempt, 60)


def fetch_hotpotqa_rows(
    config: str, split: str, limit: int | None = None, start_offset: int = 0, max_retries: int = 8
) -> list[dict[str, Any]]:
    """
    Fetch `limit` rows of HotpotQA via the HuggingFace datasets-server REST
    API, paginating in batches of HF_PAGE_SIZE (the API's per-request cap —
    requesting length > 100 in one call returns a 422). Stops early if the
    split has fewer than `limit` rows. If limit is None, fetches all rows.

    On 429, waits according to the server's own Retry-After/RateLimit
    headers when present (HF's rate-limit windows are 5 minutes, so a
    short fixed backoff can still fail repeatedly); otherwise falls back
    to capped exponential backoff. `start_offset` allows resuming a
    previously interrupted pull (see --resume-offset in main()).
    """
    rows: list[dict[str, Any]] = []
    offset = start_offset
    target = (start_offset + limit) if limit is not None else float("inf")
    while offset < target:
        batch_size = HF_PAGE_SIZE
        if limit is not None:
            batch_size = min(HF_PAGE_SIZE, target - offset)

        for attempt in range(max_retries):
            resp = requests.get(
                HF_DATASETS_SERVER,
                params={"dataset": HF_DATASET, "config": config, "split": split, "offset": offset, "length": batch_size},
                timeout=60,
            )
            if resp.status_code == 429:
                wait = _wait_seconds_from_headers(resp, attempt)
                logger.warning(f"  Rate limited (429) at offset {offset}, retrying in {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            print(f"\nGave up after {max_retries} retries at offset={offset}.")
            print(f"Fetched {len(rows)} rows so far (offsets {start_offset}-{offset - 1}).")
            print(f"To resume from here, re-run with: --resume-offset {offset} --limit {target - offset}")
            print(f"(then concatenate with the {len(rows)} rows already saved, if you saved them)")
            raise RuntimeError(f"Gave up after repeated 429s fetching offset={offset}")

        batch = [r["row"] for r in resp.json()["rows"]]
        if not batch:
            break  # reached the end of the split (fewer rows available than `limit`)
        rows.extend(batch)
        offset += len(batch)

        if offset < target:
            time.sleep(1.0)  # be polite between pages, even when not rate-limited

    return rows


def load_documents_from_file(path: str) -> list[dict[str, Any]]:
    """Load and parse HotpotQA document pool from a local JSONL file."""
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            documents.append({
                "document_id": item.get("document_id"),
                "title": item.get("title"),
                "content": item.get("content"),
            })
    return documents


def load_questions_from_file(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load questions from a local questions JSONL file and normalize them to the internal HF row format."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sf = item.get("supporting_facts", [])
            sf_dict = {"title": [], "sent_id": []}
            if isinstance(sf, list):
                for fact in sf:
                    sf_dict["title"].append(fact.get("title"))
                    sf_dict["sent_id"].append(fact.get("sent_id"))
            elif isinstance(sf, dict):
                sf_dict = sf

            rows.append({
                "id": item.get("question_id"),
                "question": item.get("user_input") or item.get("question"),
                "answer": item.get("reference") or item.get("answer"),
                "type": item.get("category") or item.get("type"),
                "level": item.get("level"),
                "supporting_facts": sf_dict,
                "context": item.get("context", {"title": [], "sentences": []}),
            })
            if limit is not None and len(rows) >= limit:
                break
    return rows


def get_existing_doc_ids(
    session: requests.Session, rag_url: str, datasource_id: str
) -> set[str]:
    """Fetch unique document IDs already stored in the datasource to avoid re-embedding them."""
    doc_ids = set()
    offset = 0
    limit = 1000
    while True:
        if offset + limit >= 16384:
            logger.warning(
                "  Warning: reached Milvus pagination limit (16384 chunks). Cannot fetch more existing document IDs."
            )
            break
        resp = session.get(
            f"{rag_url}/v1/datasource/{datasource_id}/documents",
            params={"offset": offset, "limit": limit},
        )
        if resp.status_code == 404:
            break
        if not resp.ok:
            logger.warning(
                f"  Warning: failed to fetch existing documents: HTTP {resp.status_code}"
            )
            break
        data = resp.json()
        docs = data.get("documents", [])
        if not docs:
            break
        for doc in docs:
            doc_ids.add(doc["document_id"])
        if not data.get("has_more"):
            break
        offset += limit
    return doc_ids


def compute_document_id(title: str) -> str:
    """
    Deterministic document_id for a HotpotQA context paragraph, based on its
    title. Used both when building Documents for ingestion AND when building
    expected_doc_ids for the golden question set, so the two always line up.
    """
    return f"hotpotqa_{hashlib.md5(title.encode()).hexdigest()[:16]}"


def build_documents(rows: list[dict[str, Any]], datasource_id: str, ingestor_id: str) -> list[dict[str, Any]]:
    """Convert HotpotQA context paragraphs into deduplicated CAIPE Document dicts.

    Each HotpotQA row has a `context` with parallel `title` / `sentences` lists
    (one Wikipedia paragraph per entry). The same paragraph is referenced by
    many questions, so we dedupe by a hash of the title into one Document per
    unique paragraph.
    """
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        context = row.get("context", {})
        if "title" not in context or "sentences" not in context:
            continue
        for title, sentences in zip(context["title"], context["sentences"]):
            document_id = compute_document_id(title)
            if document_id in documents:
                continue
            documents[document_id] = {
                "page_content": f"{title}\n\n{''.join(sentences)}",
                "type": "Document",
                "metadata": {
                    "document_id": document_id,
                    "datasource_id": datasource_id,
                    "ingestor_id": ingestor_id,
                    "title": title,
                    "description": "",
                    "is_structured_entity": False,
                    "document_type": "text",
                    "document_ingested_at": None,
                    "fresh_until": None,
                    "metadata": {"source": "hotpotqa"},
                },
            }
    return list(documents.values())


def build_documents_from_local_pool(
    local_docs: list[dict[str, Any]],
    datasource_id: str,
    ingestor_id: str,
    reference_doc_ids: set[str] | None = None,
    exclude_doc_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert local document pool items into CAIPE Document dicts, prioritizing reference docs."""
    ref_docs = []
    other_docs = []
    for item in local_docs:
        title = item["title"]
        content = item["content"]
        document_id = item.get("document_id") or compute_document_id(title)
        if exclude_doc_ids and document_id in exclude_doc_ids:
            continue
        doc_entry = {
            "page_content": f"{title}\n\n{content}",
            "type": "Document",
            "metadata": {
                "document_id": document_id,
                "datasource_id": datasource_id,
                "ingestor_id": ingestor_id,
                "title": title,
                "description": "",
                "is_structured_entity": False,
                "document_type": "text",
                "document_ingested_at": None,
                "fresh_until": None,
                "metadata": {"source": "hotpotqa"},
            },
        }
        if reference_doc_ids and document_id in reference_doc_ids:
            ref_docs.append(doc_entry)
        else:
            other_docs.append(doc_entry)
    return ref_docs + other_docs


def build_golden_question_set(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    golden = []
    for row in rows:
        sf = row["supporting_facts"]
        expected_doc_ids = [compute_document_id(t) for t in sf["title"]]
        golden.append({
            "question_id": row["id"],
            "user_input": row["question"],
            "reference": row["answer"],
            "category": f"{row['type']}_{row['level']}",  # Combined type and level
            "level": row["level"],    # "easy" / "medium" / "hard" — HotpotQA-specific
            "expected_doc_ids": expected_doc_ids,
            "source_types": ["hotpotqa"],
            "supporting_facts": [
                {"title": t, "sent_id": s} for t, s in zip(sf["title"], sf["sent_id"])
            ],
        })
    return golden


def write_golden_set_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    """Write the golden question set to a JSON Lines file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_golden_set_csv(rows: list[dict[str, Any]], path: str) -> None:
    """Write the golden question set to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "user_input", "reference", "category", "level",
            "expected_doc_ids", "source_types", "supporting_facts",
        ])
        for row in rows:
            sf_str = "; ".join(f"{sf['title']} (sent {sf['sent_id']})" for sf in row["supporting_facts"])
            writer.writerow([
                row["question_id"],
                row["user_input"],
                row["reference"],
                row["category"],
                row["level"],
                ";".join(row["expected_doc_ids"]),
                ";".join(row["source_types"]),
                sf_str,
            ])


def _build_pool_from_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a flat, deduplicated list of every candidate document (paragraph)
    referenced across the provided documents list, with its full content — i.e. the
    same set of documents that build_documents() will turn into CAIPE
    Documents and ingest.
    """
    pool = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "")
        page_content = doc.get("page_content", "")
        prefix = f"{title}\n\n"
        if page_content.startswith(prefix):
            content = page_content[len(prefix):]
        else:
            content = page_content
        pool.append({
            "document_id": metadata.get("document_id"),
            "title": title,
            "content": content,
        })
    return pool


def _build_pool_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a flat, deduplicated list of every candidate document (paragraph)
    referenced across all fetched rows, with its full content — i.e. the
    same set of documents that build_documents() will turn into CAIPE
    Documents and ingest.
    """
    pool: dict[str, dict[str, Any]] = {}
    for row in rows:
        context = row.get("context", {})
        if "title" not in context or "sentences" not in context:
            continue
        for title, sentences in zip(context["title"], context["sentences"]):
            document_id = compute_document_id(title)
            if document_id in pool:
                continue
            pool[document_id] = {
                "document_id": document_id,
                "title": title,
                "content": "".join(sentences),
            }
    return list(pool.values())


def build_document_pool(rows: list[dict[str, Any]], documents: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """
    Build a flat, deduplicated list of every candidate document (paragraph)
    referenced across all fetched rows, with its full content — i.e. the
    same set of documents that build_documents() will turn into CAIPE
    Documents and ingest.
    """
    if documents:
        return _build_pool_from_documents(documents)
    return _build_pool_from_rows(rows)


def write_document_pool_jsonl(pool: list[dict[str, Any]], path: str) -> None:
    """Write the candidate document pool to a JSON Lines file."""
    with open(path, "w", encoding="utf-8") as f:
        for doc in pool:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def write_document_pool_csv(pool: list[dict[str, Any]], path: str) -> None:
    """Write the candidate document pool to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document_id", "title", "content"])
        for doc in pool:
            writer.writerow([doc["document_id"], doc["title"], doc["content"]])


def register_ingestor(session: requests.Session, rag_url: str) -> tuple[str, int]:
    """Register the hotpotqa ingestor in CAIPE and return its ID and document capacity."""
    resp = _check(
        session.post(
            f"{rag_url}/v1/ingestor/heartbeat",
            json={
                "ingestor_type": INGESTOR_TYPE,
                "ingestor_name": INGESTOR_NAME,
                "description": "Local dev script: HotpotQA sample ingestion for RAG evaluation",
            },
        )
    )
    body = resp.json()
    return body["ingestor_id"], body["max_documents_per_ingest"]


def delete_datasource(session: requests.Session, rag_url: str, datasource_id: str) -> None:
    """Delete a datasource by ID in CAIPE."""
    resp = session.delete(f"{rag_url}/v1/datasource", params={"datasource_id": datasource_id})
    if resp.status_code == 404:
        return
    _check(resp)
    logger.info(f"Deleted existing datasource {datasource_id!r}")


def upsert_datasource(session: requests.Session, rag_url: str, datasource_id: str, name: str, ingestor_id: str) -> None:
    """Create or update a datasource in CAIPE."""
    _check(
        session.post(
            f"{rag_url}/v1/datasource",
            json={
                "datasource_id": datasource_id,
                "name": name,
                "ingestor_id": ingestor_id,
                "description": "HotpotQA distractor-set context paragraphs (local dev RAG evaluation sample)",
                "source_type": INGESTOR_TYPE,
                "last_updated": int(time.time()),
            },
        )
    )


def create_job(session: requests.Session, rag_url: str, datasource_id: str, total: int) -> str:
    """Create a new document ingestion job in CAIPE."""
    resp = _check(
        session.post(
            f"{rag_url}/v1/job",
            params={
                "datasource_id": datasource_id,
                "job_status": "in_progress",
                "message": "HotpotQA ingestion (local dev script)",
                "total": total,
            },
        )
    )
    return resp.json()["job_id"]


def ingest_documents(
    session: requests.Session,
    rag_url: str,
    ingestor_id: str,
    datasource_id: str,
    job_id: str,
    documents: list[dict[str, Any]],
    batch_size: int,
    start_index: int = 0,
) -> None:
    """Upload documents to CAIPE in batches, updating the job's progress on the server."""
    for i in range(start_index, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        resp = _check(
            session.post(
                f"{rag_url}/v1/ingest",
                json={
                    "documents": batch,
                    "ingestor_id": ingestor_id,
                    "datasource_id": datasource_id,
                    "job_id": job_id,
                },
            )
        )
        # Update progress and document count metrics on the server
        _check(
            session.post(
                f"{rag_url}/v1/job/{job_id}/increment-document-count",
                params={"increment": len(batch)},
            )
        )
        _check(
            session.post(
                f"{rag_url}/v1/job/{job_id}/increment-progress",
                params={"increment": len(batch)},
            )
        )
        logger.info(f"  ingested batch {i // batch_size + 1} ({len(batch)} documents): {resp.json()}")


def complete_job(session: requests.Session, rag_url: str, job_id: str) -> None:
    """Mark an ingestion job as completed in CAIPE."""
    _check(
        session.patch(
            f"{rag_url}/v1/job/{job_id}",
            params={"job_status": "completed", "message": "HotpotQA ingestion complete"},
        )
    )


def run_sample_queries(
    session: requests.Session,
    rag_url: str,
    datasource_id: str,
    rows: list[dict[str, Any]],
    num_queries: int,
    query_limit: int,
) -> None:
    recalls: list[float] = []
    for row in rows[:num_queries]:
        question = row["question"]
        resp = _check(
            session.post(
                f"{rag_url}/v1/query",
                json={"query": question, "limit": query_limit, "filters": {"datasource_id": datasource_id}},
            )
        )
        results = resp.json()
        logger.info(f"\nQ: {question}")
        logger.info(f"   expected answer: {row['answer']}")
        if not results:
            logger.info("   (no results)")

        retrieved_titles: set[str] = set()
        for r in results:
            doc = r["document"]
            title = doc.get("metadata", {}).get("title", "?")
            retrieved_titles.add(title)
            snippet = doc["page_content"][:160].replace("\n", " ")
            logger.info(f"   [{r['score']:.3f}] {title}: {snippet}...")

        # HotpotQA ground truth: titles of the paragraphs actually needed to
        # answer the question. Used as a recall@k retrieval-quality signal.
        expected_titles = set(row["supporting_facts"]["title"])
        hit_titles = expected_titles & retrieved_titles
        recall = len(hit_titles) / len(expected_titles) if expected_titles else 0.0
        recalls.append(recall)
        logger.info(f"   supporting facts: {sorted(expected_titles)}")
        logger.info(f"   recall@{query_limit}: {len(hit_titles)}/{len(expected_titles)} ({recall:.2f})")

    if recalls:
        logger.info(f"\n=== Mean recall@{query_limit} over {len(recalls)} questions: {sum(recalls) / len(recalls):.3f} ===")



def _adjust_limits(args: argparse.Namespace) -> None:
    if args.limit_per_category is not None:
        # Since HotpotQA has 2 categories ("bridge", "comparison") and 3 difficulty levels ("easy", "medium", "hard"),
        # we have up to 6 distinct category-difficulty combinations.
        required_limit = 6 * args.limit_per_category
        if args.limit < required_limit:
            logger.warning(
                f"Capping limit of {args.limit} is too low for --limit-per-category {args.limit_per_category}. "
                f"Adjusting --limit to {required_limit} to satisfy all 6 category-difficulty combinations."
            )
            args.limit = required_limit


def _load_qa_rows(args: argparse.Namespace) -> list[dict]:
    if args.input_questions_file:
        logger.info(f"Loading questions from local file {args.input_questions_file}...")
        return load_questions_from_file(args.input_questions_file, limit=None)
    if args.input_file and os.path.exists(DEFAULT_QUESTIONS_FILE):
        logger.info(f"Automatically loading questions from default local path {DEFAULT_QUESTIONS_FILE}...")
        return load_questions_from_file(DEFAULT_QUESTIONS_FILE, limit=None)
    logger.info(f"Fetching all HotpotQA rows ({args.config}/{args.split}) starting at offset {args.resume_offset}...")
    return fetch_hotpotqa_rows(args.config, args.split, limit=None, start_offset=args.resume_offset)


def _get_row_expected_ids(row: dict) -> list[str]:
    expected = row.get("expected_doc_ids")
    if expected:
        return expected
    if "supporting_facts" in row:
        sf = row["supporting_facts"]
        return [compute_document_id(t) for t in sf.get("title", [])]
    return []


def _extract_reference_ids(args: argparse.Namespace, rows: list[dict]) -> set[str]:
    ref_ids = set()
    if not args.prioritize_reference:
        return ref_ids
    q_file = args.input_questions_file or DEFAULT_QUESTIONS_FILE
    if os.path.exists(q_file):
        logger.info(f"Extracting reference document IDs from all questions in {q_file}...")
        all_questions = load_questions_from_file(q_file, limit=None)
        for q_row in all_questions:
            ref_ids.update(_get_row_expected_ids(q_row))
    else:
        for row in rows:
            ref_ids.update(_get_row_expected_ids(row))
    return ref_ids


def _build_stratified_documents(
    args: argparse.Namespace, rows: list[dict], local_docs: list[dict], ingestor_id: str
) -> list[dict]:
    # Group questions by category to find their reference IDs
    category_ref_ids: dict[str, set[str]] = {}
    for row in rows:
        cat_type = row.get("type") or "basic"
        cat_level = row.get("level") or "basic"
        cat = f"{cat_type}_{cat_level}"
        if cat not in category_ref_ids:
            category_ref_ids[cat] = set()
        
        if args.prioritize_reference:
            category_ref_ids[cat].update(_get_row_expected_ids(row))

    # Build stratified document set: limit_per_category docs for each category
    ALL_HOTPOTQA_CATEGORIES = [
        "bridge_easy", "bridge_medium", "bridge_hard",
        "comparison_easy", "comparison_medium", "comparison_hard"
    ]
    documents = []
    selected_ids = set()
    for cat in ALL_HOTPOTQA_CATEGORIES:
        ref_ids = category_ref_ids.get(cat, set())
        cat_docs = build_documents_from_local_pool(
            local_docs, args.datasource_id, ingestor_id,
            reference_doc_ids=ref_ids, exclude_doc_ids=selected_ids
        )
        # Slice to the limit per category
        cat_docs = cat_docs[:args.limit_per_category]
        documents.extend(cat_docs)
        selected_ids.update(d["metadata"]["document_id"] for d in cat_docs)
        logger.info(f"  Category {cat}: collected {len(cat_docs)} docs")
    
    # Deduplicate across categories
    seen_ids = set()
    dedup_docs = []
    for doc in documents:
        doc_id = doc["metadata"]["document_id"]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            dedup_docs.append(doc)
    return dedup_docs


def _load_or_build_documents(
    args: argparse.Namespace, rows: list[dict], ingestor_id: str
) -> list[dict]:
    if not args.input_file:
        return build_documents(rows, args.datasource_id, ingestor_id)

    logger.info(f"Loading documents from local file {args.input_file}...")
    local_docs = load_documents_from_file(args.input_file)

    if args.limit_per_category is not None:
        logger.info(f"Building stratified document pool: limit {args.limit_per_category} docs per category...")
        return _build_stratified_documents(args, rows, local_docs, ingestor_id)

    ref_ids = _extract_reference_ids(args, rows)
    documents = build_documents_from_local_pool(
        local_docs, args.datasource_id, ingestor_id, reference_doc_ids=ref_ids
    )
    return documents[:args.limit]


def _filter_questions_per_category(
    args: argparse.Namespace, rows: list[dict], documents: list[dict]
) -> list[dict]:
    if args.limit_per_category is None:
        return rows

    ingested_ids = {d["metadata"]["document_id"] for d in documents}
    fully_covered = []
    partially_covered = []
    uncovered = []
    
    for row in rows:
        cat_type = row.get("type") or "basic"
        cat_level = row.get("level") or "basic"
        cat = f"{cat_type}_{cat_level}"
        row["category"] = cat
        
        expected = set(_get_row_expected_ids(row))
            
        if not expected:
            uncovered.append(row)
        elif expected <= ingested_ids:
            fully_covered.append(row)
        elif expected & ingested_ids:
            partially_covered.append(row)
        else:
            uncovered.append(row)
            
    category_counts: dict[str, int] = {}
    filtered_rows = []
    for row in fully_covered + partially_covered + uncovered:
        cat = row["category"]
        count = category_counts.get(cat, 0)
        if count < args.limit_per_category:
            filtered_rows.append(row)
            category_counts[cat] = count + 1
    logger.info(f"Filtered to {len(filtered_rows)} QA examples ({args.limit_per_category} max per category-difficulty combination)")
    return filtered_rows


def _run_ingestion_job(
    session: requests.Session, args: argparse.Namespace, documents: list[dict]
) -> None:
    if args.skip_ingest:
        return

    if args.reset:
        delete_datasource(session, args.rag_url, args.datasource_id)

    registered_ingestor_id, max_docs = register_ingestor(session, args.rag_url)
    logger.info(f"Ingestor registered: {registered_ingestor_id} (max_documents_per_ingest={max_docs})")

    upsert_datasource(session, args.rag_url, args.datasource_id, args.datasource_name, registered_ingestor_id)
    logger.info(f"Datasource ready: {args.datasource_id}")

    # Deduplication check
    if not args.reset:
        logger.info("Checking for already ingested documents on the server...")
        existing_doc_ids = get_existing_doc_ids(session, args.rag_url, args.datasource_id)
        logger.info(f"Found {len(existing_doc_ids)} existing document IDs on the server.")
        to_ingest = [d for d in documents if d["metadata"]["document_id"] not in existing_doc_ids]
        logger.info(f"Filtered out already ingested documents. {len(to_ingest)} new documents to ingest.")
    else:
        to_ingest = documents

    if to_ingest:
        job_id = create_job(session, args.rag_url, args.datasource_id, total=len(to_ingest))
        logger.info(f"Job created: {job_id}")

        batch_size = min(args.batch_size, max_docs)
        start_index = (args.start_batch - 1) * batch_size

        if start_index > 0:
            logger.info(f"Resuming ingestion from batch {args.start_batch} (document offset: {start_index})...")
            _check(
                session.post(
                    f"{args.rag_url}/v1/job/{job_id}/increment-document-count",
                    params={"increment": start_index},
                )
            )
            _check(
                session.post(
                    f"{args.rag_url}/v1/job/{job_id}/increment-progress",
                    params={"increment": start_index},
                )
            )

        ingest_documents(
            session,
            args.rag_url,
            registered_ingestor_id,
            args.datasource_id,
            job_id,
            to_ingest,
            batch_size,
            start_index=start_index,
        )

        complete_job(session, args.rag_url, job_id)
        logger.info("Job marked completed")
    else:
        logger.info("\nAll targeted documents are already present on the server. Skipping ingestion job.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rag-url", default="http://localhost:9446", help="RAG server base URL")
    parser.add_argument("--config", default="distractor", choices=["distractor", "fullwiki"], help="HotpotQA config")
    parser.add_argument("--split", default="validation", help="HotpotQA split (validation/train)")
    parser.add_argument("--limit", type=int, default=300000, help="Max documents to ingest")
    parser.add_argument(
        "--resume-offset", type=int, default=0,
        help="Resume fetching from this offset (use the offset printed if a previous run hit repeated 429s)",
    )
    parser.add_argument("--datasource-id", default="hotpotqa_sample", help="CAIPE datasource_id to ingest into")
    parser.add_argument("--datasource-name", default="HotpotQA sample", help="Display name for the datasource")
    parser.add_argument("--batch-size", type=int, default=200, help="Documents per /v1/ingest request")
    parser.add_argument("--num-queries", type=int, default=5, help="How many HotpotQA questions to test retrieval with")
    parser.add_argument("--query-limit", type=int, default=3, help="Results per query")
    parser.add_argument("--reset", action="store_true", help="Delete the datasource before ingesting (full re-run)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion, only run sample queries")
    parser.add_argument(
        "--output-jsonl", default="hotpotqa_questions.jsonl",
        help="Where to write the golden question set (RAGAS/DeepEval-ready JSONL)",
    )
    parser.add_argument(
        "--output-csv", default="hotpotqa_questions.csv",
        help="Where to write the golden question set (CSV)",
    )
    parser.add_argument(
        "--pool-output-jsonl", default="hotpotqa_document_pool.jsonl",
        help="Where to write the full candidate document pool (all ingested paragraphs + content)",
    )
    parser.add_argument(
        "--pool-output-csv", default="hotpotqa_document_pool.csv",
        help="Where to write the full candidate document pool (CSV)",
    )

    # Local file inputs
    parser.add_argument(
        "--input-file",
        help="Path to a local document pool JSONL file to ingest documents from instead of fetching from Hugging Face",
    )
    parser.add_argument(
        "--input-questions-file",
        help="Path to a local questions JSONL file to load questions from instead of fetching from Hugging Face",
    )

    # OIDC configurations
    parser.add_argument(
        "--disable-oidc",
        dest="use_oidc",
        action="store_false",
        default=True,
        help="Disable OIDC token authentication",
    )
    parser.add_argument(
        "--oidc-token-url",
        default=os.environ.get("CAIPE_OIDC_TOKEN_URL", "http://localhost:8080/realms/caipe/protocol/openid-connect/token"),
        help="OIDC Token Endpoint URL",
    )
    parser.add_argument(
        "--oidc-token",
        default=os.environ.get("CAIPE_OIDC_TOKEN"),
        help="OIDC Access Token for CAIPE RAG server authentication",
    )
    parser.add_argument(
        "--oidc-client-id",
        default=os.environ.get("CAIPE_OIDC_CLIENT_ID"),
        help="OIDC Client ID",
    )
    parser.add_argument(
        "--oidc-client-secret",
        default=os.environ.get("CAIPE_OIDC_CLIENT_SECRET"),
        help="OIDC Client Secret",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=1,
        help="Batch number (1-based) to resume ingestion from",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        help="Limit number of questions to fetch/load per category",
    )
    parser.add_argument(
        "--prioritize-reference",
        nargs="?",
        const=DEFAULT_QUESTIONS_FILE,
        help=f"Prioritize reference documents covered by the specified questions file (defaults to {DEFAULT_QUESTIONS_FILE})",
    )

    args = parser.parse_args()

    _adjust_limits(args)

    # No Authorization header is sent on this session - on a local dev stack with
    # CAIPE_UNSAFE_RBAC_BYPASS=true this is what makes rag-server treat the
    # caller as an admin.
    session = _setup_session(args)

    # Load questions / rows
    rows = _load_qa_rows(args)
    logger.info(f"Loaded {len(rows)} QA examples")

    # Load/build documents
    ingestor_id = f"{INGESTOR_TYPE}:{INGESTOR_NAME}"
    documents = _load_or_build_documents(args, rows, ingestor_id)
    logger.info(f"Built {len(documents)} unique context documents")

    # Filter questions per category, prioritizing fully/partially covered ones
    rows = _filter_questions_per_category(args, rows, documents)

    # Ingest documents if needed
    _run_ingestion_job(session, args, documents)

    # Golden question set — built from the same `rows` that were (or already
    # were) ingested, so expected_doc_ids always line up with what's in CAIPE.
    golden_set = build_golden_question_set(rows)
    write_golden_set_jsonl(golden_set, args.output_jsonl)
    write_golden_set_csv(golden_set, args.output_csv)
    logger.info(f"\nWrote golden question set: {args.output_jsonl}")
    logger.info(f"Wrote golden question set: {args.output_csv}")

    # Candidate document pool — every unique paragraph across all fetched
    # rows, with full content.
    document_pool = build_document_pool(rows, documents)
    write_document_pool_jsonl(document_pool, args.pool_output_jsonl)
    write_document_pool_csv(document_pool, args.pool_output_csv)
    logger.info(f"Wrote document pool ({len(document_pool)} unique docs): {args.pool_output_jsonl}")
    logger.info(f"Wrote document pool ({len(document_pool)} unique docs): {args.pool_output_csv}")

    logger.info("\n=== Sample queries ===")
    run_sample_queries(session, args.rag_url, args.datasource_id, rows, args.num_queries, args.query_limit)

    logger.info("\n=== Ingestion Summary ===")

    logger.info(f"Datasource: {args.datasource_id}")
    logger.info(f"Documents Ingested/Present: {len(documents)}")
    logger.info(f"Golden QA Examples Generated: {len(rows)}")
    logger.info(f"Golden Question Set File: {args.output_jsonl}")
    logger.info(f"Document Pool File: {args.pool_output_jsonl}")
    logger.info("=========================")


if __name__ == "__main__":
    main()


