"""
Ingest a sample of the EnterpriseRAG-Bench dataset into a local CAIPE RAG
server via its public REST API, then run sample queries to verify retrieval(optional).

Flow: download source-type slice .zip(s) from GitHub Releases (EnterpriseRAG-Bench) -> extract
      -> ingestor heartbeat -> create/replace datasource -> create job
      -> POST /v1/ingest (batched) -> mark job completed -> /v1/query.

Requires a local dev rag-server with CAIPE_UNSAFE_RBAC_BYPASS=true (and no
Authorization header on these requests)

"""

from __future__ import annotations

import argparse
import logging
import hashlib
import io
import os
import json
import time
import zipfile
from typing import Any

import requests

RELEASE_BASE_URL = (
    "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0"
)
QUESTIONS_JSONL_URL = (
    "https://raw.githubusercontent.com/onyx-dot-app/"
    "EnterpriseRAG-Bench/main/questions.jsonl"
)

INGESTOR_TYPE = "enterprise_rag_bench"
INGESTOR_NAME = "enterprise-rag-bench-eval-script"

logger = logging.getLogger(__name__)

TEN_YEARS_IN_SECONDS = 10 * 365 * 24 * 60 * 60  # 315360000
YEAR_2033_EPOCH = 2000000000

# source_type -> number of release slice zips for that source

SOURCE_SLICE_COUNTS = {
    "confluence": 2,
    "jira": 2,
    "github": 2,
    "hubspot": 4,
    "fireflies": 3,
    "linear": 8,
    "google_drive": 6,
    "gmail": 25,
    "slack": 58,
}
ALL_SOURCE_TYPES = list(SOURCE_SLICE_COUNTS.keys())


def _get_oidc_token(
    oidc_token_url: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    insecure: bool = False,
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
            verify=not insecure,
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception:
        logger.exception("Error fetching token")
        raise


def _check(resp: requests.Response) -> requests.Response:
    """Raise with the response body printed, for easier debugging of 4xx/5xx."""
    if not resp.ok:
        logger.error(
            f"HTTP {resp.status_code} for {resp.request.method} {resp.request.url}"
        )
        logger.error(resp.text)
        resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# EnterpriseRAG-Bench dataset loading (GitHub Release zips + GitHub raw)
# ---------------------------------------------------------------------------


def _parse_doc_filename(name: str) -> tuple[str, str] | None:
    """
    Parse "<source_type>/dsid_<hex>__<title-slug>.txt" -> (doc_id, title_slug).
    Returns None for non-matching entries (directories etc.).
    """
    base = name.rsplit("/", 1)[-1]
    if not base.startswith("dsid_") or not base.endswith(".txt"):
        return None
    stem = base[:-4]  # strip ".txt"
    if "__" not in stem:
        return None
    doc_id, slug = stem.split("__", 1)
    return doc_id, slug


def _load_zip_content(zip_name: str, zip_path: str) -> bytes | None:
    """Helper to check cache or download and save zip file content."""
    if os.path.exists(zip_path):
        logger.info(f"    using cached {zip_name}...")
        with open(zip_path, "rb") as f:
            return f.read()

    logger.info(f"    downloading {zip_name}...")
    url = f"{RELEASE_BASE_URL}/{zip_name}"
    resp = requests.get(url, timeout=120)
    if not resp.ok:
        logger.warning(
            f"    Warning: failed to download {zip_name}: HTTP {resp.status_code}"
        )
        return None

    zip_content = resp.content
    with open(zip_path, "wb") as f:
        f.write(zip_content)
    return zip_content


def _process_zip_entry(
    zf: zipfile.ZipFile,
    name: str,
    source_type: str,
    reference_doc_ids: set[str],
    seen_hashes: set[str],
    reference_docs: list[dict[str, Any]],
    other_docs: list[dict[str, Any]],
) -> None:
    """Helper to parse a single document file from a zip file and classify it."""
    parsed = _parse_doc_filename(name)
    if not parsed:
        return
    doc_id, slug = parsed

    raw = zf.read(name).decode("utf-8", errors="replace")
    lines = raw.split("\n", 1)
    title = lines[0].strip() if lines else slug.replace("-", " ")
    text = raw.strip()
    if not text:
        return

    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen_hashes:
        return
    seen_hashes.add(h)

    doc_entry = {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "source_type": source_type,
    }

    if doc_id in reference_doc_ids:
        reference_docs.append(doc_entry)
    else:
        other_docs.append(doc_entry)


def _process_zip_file(
    zip_content: bytes,
    zip_name: str,
    source_type: str,
    reference_doc_ids: set[str],
    seen_hashes: set[str],
    reference_docs: list[dict[str, Any]],
    other_docs: list[dict[str, Any]],
) -> None:
    """Helper to process all entries in the zip content."""
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".txt")]
        logger.info(f"    {zip_name}: {len(names)} files")
        for name in names:
            _process_zip_entry(
                zf,
                name,
                source_type,
                reference_doc_ids,
                seen_hashes,
                reference_docs,
                other_docs,
            )


def fetch_documents(
    source_types: list[str],
    limit_per_source: int,
    cache_dir: str = "cache",
    reference_doc_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Download slice zip(s) for each requested source type from GitHub Releases,
    extract in-memory, and parse up to `limit_per_source` documents per source.
    Prioritizes documents whose IDs are in `reference_doc_ids`.
    """
    if reference_doc_ids is None:
        reference_doc_ids = set()

    docs: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    os.makedirs(cache_dir, exist_ok=True)

    for source_type in source_types:
        n_slices = SOURCE_SLICE_COUNTS.get(source_type, 1)
        print(
            f"  {source_type}: up to {n_slices} slice(s), limit {limit_per_source}..."
        )

        reference_docs: list[dict[str, Any]] = []
        other_docs: list[dict[str, Any]] = []

        for slice_num in range(1, n_slices + 1):
            zip_name = f"{source_type}_slice_{slice_num:04d}.zip"
            zip_path = os.path.join(cache_dir, zip_name)
            zip_content = _load_zip_content(zip_name, zip_path)
            if zip_content is None:
                continue

            _process_zip_file(
                zip_content,
                zip_name,
                source_type,
                reference_doc_ids,
                seen_hashes,
                reference_docs,
                other_docs,
            )

        # Merge reference docs first, then fill with other docs
        source_docs = reference_docs + other_docs
        selected = source_docs[:limit_per_source]
        docs.extend(selected)

        print(
            f"    -> collected {len(selected)} docs from {source_type} "
            f"({len(reference_docs)} reference docs, {len(selected) - len(reference_docs)} other docs)"
        )

    return docs


def fetch_all_questions() -> list[dict[str, Any]]:
    """
    Fetch the full questions.jsonl so we can filter for questions whose reference documents are actually
    present in whatever sample we ingested.

    Schema per row: question_id, category, source_types, user_input,
                    expected_doc_ids, reference, answer_facts
    """
    resp = requests.get(QUESTIONS_JSONL_URL, timeout=30)
    resp.raise_for_status()

    questions = []
    for line in resp.text.strip().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            {
                "question_id": row.get("question_id", ""),
                "question": row.get("user_input", ""),
                "answer": row.get("reference", ""),
                "category": row.get("category", ""),
                "source_types": row.get("source_types", []),
                "expected_doc_ids": row.get("expected_doc_ids", []),
            }
        )
    return questions


def select_questions(
    all_questions: list[dict[str, Any]],
    source_types: list[str],
    ingested_doc_ids: set[str],
    num_questions: int,
) -> list[dict[str, Any]]:
    """
    Pick questions for the sample-query check, prioritizing ones whose reference
    documents are actually present in the ingested sample — otherwise recall
    is meaningless.

    Priority: 1) fully covered (all expected_doc_ids ingested)
    2) partially covered (at least one expected_doc_id ingested)
    3) fallback: source_types match only, no guarantee of coverage
    """
    wanted_sources = set(source_types)

    def sources_match(q: dict[str, Any]) -> bool:
        qs = set(q.get("source_types", []))
        return not qs or bool(qs & wanted_sources)

    candidates = [q for q in all_questions if sources_match(q)]

    fully_covered, partially_covered, uncovered = [], [], []
    for q in candidates:
        expected = set(q.get("expected_doc_ids", []))
        if not expected:
            uncovered.append(
                q
            )  # e.g. high_level / info_not_found — no reference docs by design
        elif expected <= ingested_doc_ids:
            fully_covered.append(q)
        elif expected & ingested_doc_ids:
            partially_covered.append(q)
        else:
            uncovered.append(q)

    selected = (fully_covered + partially_covered + uncovered)[:num_questions]
    print(
        f"  Question selection: {len(fully_covered)} fully covered, "
        f"{len(partially_covered)} partially covered, "
        f"{len(uncovered)} uncovered/no-reference-doc (out of {len(candidates)} candidates) "
        f"-> picked {len(selected)}"
    )
    return selected


def build_documents(
    raw_docs: list[dict[str, Any]], datasource_id: str, ingestor_id: str
) -> list[dict[str, Any]]:
    """Convert raw docs into CAIPE Document dicts."""
    documents = []
    for doc in raw_docs:
        documents.append(
            {
                "page_content": doc["text"],
                "type": "Document",
                "metadata": {
                    "document_id": doc["doc_id"],
                    "datasource_id": datasource_id,
                    "ingestor_id": ingestor_id,
                    "title": doc["title"],
                    "description": f"Enterprise RAG Bench — {doc['source_type']}",
                    "is_structured_entity": False,
                    "document_type": "text",
                    "document_ingested_at": None,
                    "fresh_until": None,
                    "metadata": {
                        "source": "enterprise_rag_bench",
                        "source_type": doc["source_type"],
                    },
                },
            }
        )
    return documents


def get_existing_doc_ids(
    session: requests.Session, rag_url: str, datasource_id: str
) -> set[str]:
    """Fetch unique document IDs already stored in the datasource to avoid re-embedding them."""
    doc_ids = set()
    offset = 0
    limit = 1000
    while True:
        if offset + limit >= 16384:
            print(
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
            print(
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


# ---------------------------------------------------------------------------
# CAIPE RAG server API calls (verified schema, no Authorization header)
# ---------------------------------------------------------------------------


def register_ingestor(session: requests.Session, rag_url: str) -> tuple[str, int]:
    """Register the ingestor with the RAG server heartbeat endpoint and return ingestor ID and limit."""
    resp = _check(
        session.post(
            f"{rag_url}/v1/ingestor/heartbeat",
            json={
                "ingestor_type": INGESTOR_TYPE,
                "ingestor_name": INGESTOR_NAME,
                "description": "Local dev script: Enterprise RAG Bench ingestion for RAG evaluation",
            },
        )
    )
    body = resp.json()
    return body["ingestor_id"], body["max_documents_per_ingest"]


def delete_datasource(
    session: requests.Session, rag_url: str, datasource_id: str
) -> None:
    """Deletes the specified datasource from the RAG server if it exists."""
    resp = session.delete(
        f"{rag_url}/v1/datasource", params={"datasource_id": datasource_id}
    )
    if resp.status_code == 404:
        return
    _check(resp)
    logger.info(f"Deleted existing datasource {datasource_id!r}")


def upsert_datasource(
    session: requests.Session,
    rag_url: str,
    datasource_id: str,
    name: str,
    ingestor_id: str,
) -> None:
    """Creates or updates a datasource definition on the RAG server."""
    _check(
        session.post(
            f"{rag_url}/v1/datasource",
            json={
                "datasource_id": datasource_id,
                "name": name,
                "ingestor_id": ingestor_id,
                "description": (
                    "Enterprise RAG Bench — synthetic enterprise corpus simulating a tech "
                    "company: Confluence, Jira, GitHub, Slack, Gmail, "
                    "Fireflies, HubSpot, Linear, Google Drive."
                ),
                "source_type": INGESTOR_TYPE,
                "last_updated": int(time.time()),
                "reload_interval": TEN_YEARS_IN_SECONDS,
            },
        )
    )


def create_job(
    session: requests.Session, rag_url: str, datasource_id: str, total: int
) -> str:
    """Creates an ingestion job on the RAG server and returns the job ID."""
    resp = _check(
        session.post(
            f"{rag_url}/v1/job",
            params={
                "datasource_id": datasource_id,
                "job_status": "in_progress",
                "message": "Enterprise RAG Bench ingestion",
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
    """Ingests documents in batches to the RAG server and updates the job progress."""
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
                    "fresh_until": YEAR_2033_EPOCH,
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
        print(
            f"  ingested batch {i // batch_size + 1} ({len(batch)} documents): {resp.json()}"
        )


def complete_job(session: requests.Session, rag_url: str, job_id: str) -> None:
    """Marks the specified ingestion job as completed on the RAG server."""
    _check(
        session.patch(
            f"{rag_url}/v1/job/{job_id}",
            params={
                "job_status": "completed",
                "message": "Enterprise RAG Bench ingestion complete",
            },
        )
    )


def run_sample_queries(
    session: requests.Session,
    rag_url: str,
    datasource_id: str,
    questions: list[dict[str, Any]],
    query_limit: int,
) -> None:
    """Runs a set of sample queries against the RAG server to measure and print retrieval recall."""
    recalls: list[float] = []
    for q in questions:
        resp = _check(
            session.post(
                f"{rag_url}/v1/query",
                json={
                    "query": q["question"],
                    "limit": query_limit,
                    "filters": {"datasource_id": datasource_id},
                },
            )
        )
        results = resp.json()
        logger.info(f"Q [{q.get('category', '')}]: {q['question']}")
        if q.get("answer"):
            logger.info(f"   expected: {q['answer'][:150]}")
        if not results:
            logger.info("   (no results)")

        retrieved_ids: set[str] = set()
        for r in results:
            doc = r["document"]
            doc_id = doc.get("metadata", {}).get("document_id", "?")
            retrieved_ids.add(doc_id)
            title = doc.get("metadata", {}).get("title", "?")
            snippet = doc["page_content"][:140].replace("\n", " ")
            logger.info(f"   [{r['score']:.3f}] {title}: {snippet}...")

        expected_ids = set(q.get("expected_doc_ids", []))
        if expected_ids:
            hit_ids = expected_ids & retrieved_ids
            recall = len(hit_ids) / len(expected_ids)
            recalls.append(recall)
            logger.info(f"   expected_doc_ids: {sorted(expected_ids)}")
            print(
                f"   recall@{query_limit}: {len(hit_ids)}/{len(expected_ids)} ({recall:.2f})"
            )
        else:
            logger.info(
                "   (no expected_doc_ids for this category — recall not applicable)"
            )

    if recalls:
        print(
            f"\n=== Mean recall@{query_limit} over {len(recalls)} questions: {sum(recalls) / len(recalls):.3f} ==="
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rag-url", default="http://localhost:9446")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        choices=ALL_SOURCE_TYPES,
        help="Source types to ingest (default: all sources if not specified)",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=300000,
        help="Max documents per source type",
    )
    parser.add_argument("--datasource-id", default="enterprise_rag_bench")
    parser.add_argument("--datasource-name", default="Enterprise RAG Bench")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Documents per /v1/ingest request"
    )
    parser.add_argument("--num-queries", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=3)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument(
        "--start-batch",
        type=int,
        default=1,
        help="Batch index to start/resume ingestion from (1-based)",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Local directory to cache downloaded zip files",
    )
    parser.add_argument(
        "--disable-oidc",
        action="store_false",
        dest="use_oidc",
        default=True,
        help="Disable OIDC token authentication",
    )
    parser.add_argument(
        "--oidc-token-url",
        default="http://localhost:7080/realms/caipe/protocol/openid-connect/token",
        help="OIDC Token Endpoint URL",
    )
    parser.add_argument(
        "--oidc-token",
        default=os.environ.get("CAIPE_OIDC_TOKEN"),
        help="OIDC Access Token for CAIPE RAG server authentication",
    )
    parser.add_argument(
        "--oidc-client-id",
        default=None,
        help="OIDC Client ID",
    )
    parser.add_argument(
        "--oidc-client-secret",
        default=None,
        help="OIDC Client Secret",
    )
    parser.add_argument(
        "--prioritize-reference",
        action="store_true",
        help="Prioritize reference documents covered by questions first",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=False,
        help="Disable SSL certificate verification (for self-signed certs)",
    )
    return parser.parse_args()


def _setup_session(args: argparse.Namespace) -> requests.Session:
    """Set up requests.Session with optional OIDC auto-refresh authentication."""
    session = requests.Session()
    if getattr(args, "insecure", False):
        session.verify = False
    if not args.use_oidc:
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
                    insecure=getattr(args, "insecure", False),
                )
                session.headers.update({"Authorization": f"Bearer {new_token}"})
                if "headers" in req_kwargs and "Authorization" in req_kwargs["headers"]:
                    req_kwargs["headers"]["Authorization"] = f"Bearer {new_token}"
                req_kwargs["_is_retry"] = True
                resp = session.request(*req_args, **req_kwargs)
            except Exception:
                logger.exception("Failed to refresh OIDC token")
        return resp

    session.request = auto_refresh_request
    return session


def _get_prioritized_doc_ids(
    args: argparse.Namespace, all_questions: list[dict], sources: list[str]
) -> set[str]:
    """Get expected_doc_ids to prioritize if prioritization is enabled."""
    reference_doc_ids = set()
    if args.prioritize_reference:
        for q in all_questions:
            # Check if the question's source types overlap with our target sources
            q_sources = q.get("source_types", [])
            if not q_sources or any(src in sources for src in q_sources):
                reference_doc_ids.update(q.get("expected_doc_ids", []))
        logger.info(
            f"Identified {len(reference_doc_ids)} reference doc IDs to prioritize."
        )
    else:
        logger.info("Reference document prioritization is disabled.")
    return reference_doc_ids


def _run_ingestion_job(
    session: requests.Session,
    args: argparse.Namespace,
    sources: list[str],
    reference_doc_ids: set[str],
) -> set[str]:
    """Perform the ingestion pipeline and return a set of ingested document IDs."""
    if args.reset:
        delete_datasource(session, args.rag_url, args.datasource_id)

    ingestor_id, max_docs = register_ingestor(session, args.rag_url)
    print(f"Ingestor registered: {ingestor_id} (max_documents_per_ingest={max_docs})")

    logger.info(f"Fetching documents from EnterpriseRAG-Bench ({sources})...")
    raw_docs = fetch_documents(
        sources,
        args.limit_per_source,
        cache_dir=args.cache_dir,
        reference_doc_ids=reference_doc_ids,
    )
    if not raw_docs:
        print(
            "No documents fetched. Check network access to github.com release assets."
        )
        return set()

    documents = build_documents(raw_docs, args.datasource_id, ingestor_id)
    logger.info(f"Built {len(documents)} unique documents")

    upsert_datasource(
        session, args.rag_url, args.datasource_id, args.datasource_name, ingestor_id
    )
    logger.info(f"Datasource ready: {args.datasource_id}")

    if not args.reset:
        logger.info("Checking for already ingested documents on the server...")
        existing_doc_ids = get_existing_doc_ids(
            session, args.rag_url, args.datasource_id
        )
        logger.info(
            f"Found {len(existing_doc_ids)} existing document IDs on the server."
        )
        documents = [
            d for d in documents if d["metadata"]["document_id"] not in existing_doc_ids
        ]
        print(
            f"Filtered out already ingested documents. {len(documents)} new documents to ingest."
        )

    if documents:
        job_id = create_job(
            session, args.rag_url, args.datasource_id, total=len(documents)
        )
        logger.info(f"Job created: {job_id}")

        batch_size = min(args.batch_size, max_docs)
        start_index = (args.start_batch - 1) * batch_size

        if start_index > 0:
            print(
                f"Resuming ingestion from batch {args.start_batch} (document offset: {start_index})..."
            )
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
            ingestor_id,
            args.datasource_id,
            job_id,
            documents,
            batch_size,
            start_index=start_index,
        )

        complete_job(session, args.rag_url, job_id)
        logger.info("Job marked completed")

        logger.info("Waiting 5s for vectors to be indexed in Milvus...")
        time.sleep(5)
    else:
        print(
            "\nAll targeted documents (including reference documents) are already present on the server. Skipping ingestion job."
        )

    return {d["doc_id"] for d in raw_docs}


def _run_skip_ingestion_path(
    args: argparse.Namespace,
    sources: list[str],
    reference_doc_ids: set[str],
) -> set[str]:
    """Re-fetch document IDs for the skip-ingest flow."""
    print(
        f"\n--skip-ingest: re-fetching document IDs for {sources} to select covered questions..."
    )
    raw_docs = fetch_documents(
        sources,
        args.limit_per_source,
        cache_dir=args.cache_dir,
        reference_doc_ids=reference_doc_ids,
    )
    return {d["doc_id"] for d in raw_docs}


def main() -> None:
    """Main function to orchestrate the download, parser, and ingestion flow of EnterpriseRAG-Bench data."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = _parse_args()
    sources = args.sources if args.sources is not None else ALL_SOURCE_TYPES

    session = _setup_session(args)

    logger.info(f"RAG server  : {args.rag_url}")
    logger.info(f"Datasource  : {args.datasource_id}")
    logger.info(f"Sources     : {sources}")
    logger.info(f"Limit/source: {args.limit_per_source}")

    logger.info("Loading benchmark questions...")
    all_questions = fetch_all_questions()
    reference_doc_ids = _get_prioritized_doc_ids(args, all_questions, sources)

    if not args.skip_ingest:
        ingested_doc_ids = _run_ingestion_job(session, args, sources, reference_doc_ids)
        if not ingested_doc_ids:
            return
    else:
        # --skip-ingest: we don't know what's already in the datasource, so
        # we can't guarantee reference-doc coverage. Re-fetch the doc set just to
        # compute coverage for question selection.
        ingested_doc_ids = _run_skip_ingestion_path(args, sources, reference_doc_ids)

    logger.info(
        f"Selecting covered benchmark questions (source-filtered to {sources})..."
    )
    questions = select_questions(
        all_questions, sources, ingested_doc_ids, args.num_queries
    )

    logger.info("=== Sample queries ===")
    run_sample_queries(
        session, args.rag_url, args.datasource_id, questions, args.query_limit
    )


if __name__ == "__main__":
    main()
