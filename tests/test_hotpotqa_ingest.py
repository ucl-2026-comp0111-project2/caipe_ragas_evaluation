import json
import argparse
import unittest.mock as mock
import pytest
import requests

from ragas_eval import hotpotqa_rag_ingest as ingest


# 1. _get_oidc_token
def test_get_oidc_token_positive():
    """Test get_oidc_token success path."""
    with mock.patch("requests.post") as mock_post:
        mock_resp = mock.Mock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"access_token": "token123"}
        mock_post.return_value = mock_resp
        token = ingest._get_oidc_token("http://url", "client", "secret")
        assert token == "token123"


def test_get_oidc_token_missing_credentials():
    """Test get_oidc_token missing credentials raises ValueError."""
    with pytest.raises(ValueError, match="Both client_id and client_secret must be provided"):
        ingest._get_oidc_token("http://url", None, "secret")
    with pytest.raises(ValueError, match="Both client_id and client_secret must be provided"):
        ingest._get_oidc_token("http://url", "client", None)


def test_get_oidc_token_http_failure():
    """Test get_oidc_token propagates requests errors."""
    with mock.patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection lost")
        with pytest.raises(requests.exceptions.ConnectionError):
            ingest._get_oidc_token("http://url", "client", "secret")


# 2. _check
def test_check_positive():
    """Test _check with OK response."""
    mock_resp = mock.Mock()
    mock_resp.ok = True
    assert ingest._check(mock_resp) == mock_resp


def test_check_negative():
    """Test _check raises HTTPError on non-OK status."""
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 500
    mock_resp.request = mock.Mock(method="GET", url="http://test")
    mock_resp.text = "Error Details"
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    with pytest.raises(requests.exceptions.HTTPError):
        ingest._check(mock_resp)


# 3. _setup_session
def test_setup_session_no_oidc():
    """Test setup_session when OIDC is disabled."""
    args = argparse.Namespace(use_oidc=False)
    session = ingest._setup_session(args)
    assert isinstance(session, requests.Session)
    assert "Authorization" not in session.headers


def test_setup_session_explicit_token():
    """Test setup_session when explicit token is provided."""
    args = argparse.Namespace(
        use_oidc=True,
        oidc_token="my_token",
        oidc_token_url="http://token",
        oidc_client_id="client",
        oidc_client_secret="secret"
    )
    session = ingest._setup_session(args)
    assert session.headers["Authorization"] == "Bearer my_token"


def test_setup_session_auto_refresh_on_401():
    """Test setup_session auto-refreshes token on 401 response."""
    args = argparse.Namespace(
        use_oidc=True,
        oidc_token=None,
        oidc_token_url="http://token",
        oidc_client_id="client",
        oidc_client_secret="secret"
    )
    with mock.patch("ragas_eval.hotpotqa_rag_ingest._get_oidc_token", return_value="refreshed_token") as mock_get_token:
        # Mock requests.Session.request before calling _setup_session to keep test offline
        mock_resp_401 = mock.Mock(status_code=401)
        mock_resp_200 = mock.Mock(status_code=200)
        with mock.patch("requests.Session.request", side_effect=[mock_resp_401, mock_resp_200]):
            session = ingest._setup_session(args)
            resp = session.request("GET", "http://test")
            assert resp.status_code == 200
            mock_get_token.assert_called_once()
            assert session.headers["Authorization"] == "Bearer refreshed_token"


# 4. _wait_seconds_from_headers
def test_wait_seconds_retry_after():
    """Test wait_seconds_from_headers parses Retry-After header."""
    resp = mock.Mock()
    resp.headers = {"Retry-After": "4.5"}
    assert ingest._wait_seconds_from_headers(resp, 1) == 4.5
    
    # Invalid Retry-After fallback
    resp.headers = {"Retry-After": "abc"}
    assert ingest._wait_seconds_from_headers(resp, 1) == 2.0


def test_wait_seconds_ratelimit():
    """Test wait_seconds_from_headers parses RateLimit header."""
    resp = mock.Mock()
    resp.headers = {"RateLimit": "api;r=0;t=123"}
    assert ingest._wait_seconds_from_headers(resp, 1) == 123.0

    # Invalid RateLimit format
    resp.headers = {"RateLimit": "api;r=0;t=abc"}
    assert ingest._wait_seconds_from_headers(resp, 1) == 2.0


def test_wait_seconds_fallback():
    """Test wait_seconds_from_headers exponential backoff fallback."""
    resp = mock.Mock()
    resp.headers = {}
    assert ingest._wait_seconds_from_headers(resp, 3) == 8.0


# 5. fetch_hotpotqa_rows
def test_fetch_hotpotqa_rows_success():
    """Test fetching HotpotQA rows successfully."""
    with mock.patch("requests.get") as mock_get:
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"rows": [{"row": {"question": "Q1"}}, {"row": {"question": "Q2"}}]}
        mock_get.return_value = mock_resp
        
        rows = ingest.fetch_hotpotqa_rows("config", "split", 2)
        assert len(rows) == 2
        assert rows[0]["question"] == "Q1"


def test_fetch_hotpotqa_rows_retry_success():
    """Test fetch_hotpotqa_rows retries on 429 and then succeeds."""
    with mock.patch("requests.get") as mock_get, mock.patch("time.sleep"):
        mock_resp_429 = mock.Mock(status_code=429)
        mock_resp_429.headers = {"Retry-After": "0.1"}
        mock_resp_200 = mock.Mock(status_code=200)
        mock_resp_200.json.return_value = {"rows": [{"row": {"question": "Q1"}}]}
        
        mock_get.side_effect = [mock_resp_429, mock_resp_200]
        
        rows = ingest.fetch_hotpotqa_rows("config", "split", 1, max_retries=3)
        assert len(rows) == 1
        assert mock_get.call_count == 2


def test_fetch_hotpotqa_rows_rate_limit_failure():
    """Test fetching HotpotQA rows raises RuntimeError on repeated 429s."""
    with mock.patch("requests.get") as mock_get, mock.patch("time.sleep"):
        mock_resp = mock.Mock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "0.1"}
        mock_get.return_value = mock_resp
        
        with pytest.raises(RuntimeError, match="Gave up after repeated 429s"):
            ingest.fetch_hotpotqa_rows("config", "split", 1, max_retries=2)


# 6. load_documents_from_file
def test_load_documents_from_file_positive(tmp_path):
    """Test load_documents_from_file successfully parses file."""
    doc_file = tmp_path / "docs.jsonl"
    with open(doc_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"document_id": "d1", "title": "T1", "content": "C1"}) + "\n")
        f.write("\n")  # Empty line test
    docs = ingest.load_documents_from_file(str(doc_file))
    assert len(docs) == 1
    assert docs[0]["document_id"] == "d1"


def test_load_documents_from_file_empty(tmp_path):
    """Test load_documents_from_file with empty file."""
    doc_file = tmp_path / "empty.jsonl"
    with open(doc_file, "w") as f:
        f.write("")
    docs = ingest.load_documents_from_file(str(doc_file))
    assert docs == []


# 7. load_questions_from_file
def test_load_questions_from_file_limit(tmp_path):
    """Test load_questions_from_file respects limit parameter."""
    q_file = tmp_path / "questions.jsonl"
    with open(q_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": "q1", "user_input": "Q1", "reference": "A1"}) + "\n")
        f.write(json.dumps({"question_id": "q2", "user_input": "Q2", "reference": "A2"}) + "\n")
    rows = ingest.load_questions_from_file(str(q_file), limit=1)
    assert len(rows) == 1
    assert rows[0]["id"] == "q1"


def test_load_questions_from_file_sf_formats(tmp_path):
    """Test load_questions_from_file parses questions with dictionary supporting facts."""
    q_file = tmp_path / "questions.jsonl"
    q_data_1 = {
        "question_id": "q1",
        "user_input": "Q1",
        "reference": "A1",
        "supporting_facts": {"title": ["T1"], "sent_id": [0]}
    }
    q_data_2 = {
        "question_id": "q2",
        "user_input": "Q2",
        "reference": "A2"
    }
    with open(q_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(q_data_1) + "\n")
        f.write(json.dumps(q_data_2) + "\n")
    rows = ingest.load_questions_from_file(str(q_file))
    assert len(rows) == 2
    assert rows[0]["supporting_facts"]["title"] == ["T1"]
    assert rows[1]["supporting_facts"]["title"] == []


# 8. get_existing_doc_ids
def test_get_existing_doc_ids_paginated():
    """Test get_existing_doc_ids returns deduplicated set and paginates."""
    mock_session = mock.Mock()
    mock_resp_1 = mock.Mock(status_code=200, ok=True)
    mock_resp_1.json.return_value = {"documents": [{"document_id": "doc1"}], "has_more": True}
    mock_resp_2 = mock.Mock(status_code=200, ok=True)
    mock_resp_2.json.return_value = {"documents": [{"document_id": "doc2"}], "has_more": False}
    
    mock_session.get.side_effect = [mock_resp_1, mock_resp_2]
    
    ids = ingest.get_existing_doc_ids(mock_session, "http://url", "ds_id")
    assert ids == {"doc1", "doc2"}


def test_get_existing_doc_ids_milvus_limit():
    """Test get_existing_doc_ids breaks at Milvus pagination limit."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_resp.json.return_value = {"documents": [{"document_id": "doc"}], "has_more": True}
    mock_session.get.return_value = mock_resp
    
    with mock.patch("logging.Logger.warning") as mock_warn:
        ids = ingest.get_existing_doc_ids(mock_session, "http://url", "ds_id")
        mock_warn.assert_called_once()
        assert len(ids) == 1


def test_get_existing_doc_ids_non_ok():
    """Test get_existing_doc_ids returns empty set if request fails."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=500, ok=False)
    mock_session.get.return_value = mock_resp
    assert ingest.get_existing_doc_ids(mock_session, "http://url", "ds_id") == set()


# 9. compute_document_id
def test_compute_document_id_ascii():
    """Test compute_document_id formats correctly."""
    assert ingest.compute_document_id("Hello") == ingest.compute_document_id("Hello")
    assert ingest.compute_document_id("Hello").startswith("hotpotqa_")


def test_compute_document_id_unicode():
    """Test compute_document_id handles unicode titles."""
    title = "äöü"
    assert ingest.compute_document_id(title) == ingest.compute_document_id(title)


# 10. build_documents
def test_build_documents_valid():
    """Test build_documents builds document list."""
    rows = [{"context": {"title": ["T1"], "sentences": [["S1"]]}}]
    docs = ingest.build_documents(rows, "ds_id", "ingestor_1")
    assert len(docs) == 1
    assert docs[0]["metadata"]["title"] == "T1"


def test_build_documents_invalid_context():
    """Test build_documents ignores rows with missing context keys."""
    rows = [{"context": {"title": ["T1"]}}]
    assert ingest.build_documents(rows, "ds_id", "ingestor_1") == []


# 11. build_documents_from_local_pool
def test_build_documents_from_local_pool_empty():
    """Test build_documents_from_local_pool with empty list."""
    assert ingest.build_documents_from_local_pool([], "ds", "ing") == []


def test_build_documents_from_local_pool_exclude():
    """Test build_documents_from_local_pool with exclude_doc_ids parameter."""
    local = [
        {"title": "T1", "content": "C1", "document_id": "id1"},
        {"title": "T2", "content": "C2", "document_id": "id2"}
    ]
    docs = ingest.build_documents_from_local_pool(local, "ds", "ing", exclude_doc_ids={"id1"})
    assert len(docs) == 1
    assert docs[0]["metadata"]["document_id"] == "id2"


def test_build_documents_from_local_pool_prioritise():
    """Test build_documents_from_local_pool puts referenced docs first."""
    local = [
        {"title": "T1", "content": "C1", "document_id": "id1"},
        {"title": "T2", "content": "C2", "document_id": "id2"}
    ]
    docs = ingest.build_documents_from_local_pool(local, "ds", "ing", reference_doc_ids={"id2"})
    assert docs[0]["metadata"]["document_id"] == "id2"


# 12. build_golden_question_set
def test_build_golden_question_set_valid():
    """Test build_golden_question_set mappings."""
    rows = [{
        "id": "q1",
        "question": "Q1",
        "answer": "A1",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": {"title": ["T1"], "sent_id": [0]}
    }]
    golden = ingest.build_golden_question_set(rows)
    assert len(golden) == 1
    assert golden[0]["question_id"] == "q1"


def test_build_golden_question_set_empty():
    """Test build_golden_question_set returns empty list for empty rows."""
    assert ingest.build_golden_question_set([]) == []


# 13. write_golden_set_jsonl / write_golden_set_csv
def test_write_golden_set_files_empty(tmp_path):
    """Test writing empty golden set."""
    jsonl = tmp_path / "empty.jsonl"
    csv_file = tmp_path / "empty.csv"
    ingest.write_golden_set_jsonl([], str(jsonl))
    ingest.write_golden_set_csv([], str(csv_file))
    assert jsonl.exists()
    assert csv_file.exists()


# 14. build_document_pool
def test_build_document_pool_empty():
    """Test build_document_pool returns empty list when no data is provided."""
    assert ingest.build_document_pool([]) == []


# 15. write_document_pool_jsonl / write_document_pool_csv
def test_write_document_pool_files(tmp_path):
    """Test writing document pool files."""
    pool = [{"document_id": "d1", "title": "T1", "content": "C1"}]
    jsonl = tmp_path / "pool.jsonl"
    csv_file = tmp_path / "pool.csv"
    
    ingest.write_document_pool_jsonl(pool, str(jsonl))
    ingest.write_document_pool_csv(pool, str(csv_file))
    
    assert jsonl.exists()
    assert csv_file.exists()
    
    with open(csv_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2


def test_write_document_pool_files_empty(tmp_path):
    """Test writing empty document pool files."""
    jsonl = tmp_path / "empty_pool.jsonl"
    csv_file = tmp_path / "empty_pool.csv"
    ingest.write_document_pool_jsonl([], str(jsonl))
    ingest.write_document_pool_csv([], str(csv_file))
    assert jsonl.exists()
    assert csv_file.exists()


# 16. register_ingestor
def test_register_ingestor_success():
    """Test register_ingestor returns ingestor credentials."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_resp.json.return_value = {"ingestor_id": "ing_1", "max_documents_per_ingest": 100}
    mock_session.post.return_value = mock_resp
    ing_id, limit = ingest.register_ingestor(mock_session, "http://url")
    assert ing_id == "ing_1"
    assert limit == 100


def test_register_ingestor_failure():
    """Test register_ingestor propagates error on request failure."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=500, ok=False)
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_session.post.return_value = mock_resp
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.register_ingestor(mock_session, "http://url")


# 17. delete_datasource
def test_delete_datasource_success():
    """Test delete_datasource post request is triggered."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_session.delete.return_value = mock_resp
    ingest.delete_datasource(mock_session, "http://url", "ds_1")
    mock_session.delete.assert_called_once()


def test_delete_datasource_404_ignored():
    """Test delete_datasource ignores 404 response."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=404, ok=False)
    mock_session.delete.return_value = mock_resp
    # Should not raise exception
    ingest.delete_datasource(mock_session, "http://url", "ds_1")


# 18. upsert_datasource
def test_upsert_datasource_success():
    """Test upsert_datasource sends correct payload."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_session.post.return_value = mock_resp
    ingest.upsert_datasource(mock_session, "http://url", "ds_1", "name_1", "ing_1")
    mock_session.post.assert_called_once()
    payload = mock_session.post.call_args[1]["json"]
    assert payload["datasource_id"] == "ds_1"
    assert payload["reload_interval"] == 315360000


def test_upsert_datasource_failure():
    """Test upsert_datasource failure raises exception."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=500, ok=False)
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_session.post.return_value = mock_resp
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.upsert_datasource(mock_session, "http://url", "ds_1", "name", "ing")


# 19. create_job
def test_create_job_success():
    """Test create_job returns job ID."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_resp.json.return_value = {"job_id": "job_99"}
    mock_session.post.return_value = mock_resp
    assert ingest.create_job(mock_session, "http://url", "ds_1", 10) == "job_99"


def test_create_job_failure():
    """Test create_job failure raises exception."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=500, ok=False)
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_session.post.return_value = mock_resp
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.create_job(mock_session, "http://url", "ds_1", 10)


# 20. ingest_documents
def test_ingest_documents_success():
    """Test ingest_documents calls post endpoints and updates progress."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_resp.json.return_value = {"status": "ok"}
    mock_session.post.return_value = mock_resp
    
    docs = [{"doc_id": "d1"}]
    ingest.ingest_documents(mock_session, "http://url", "ing_1", "ds_1", "job_1", docs, batch_size=1)
    # /ingest, /increment-document-count, /increment-progress
    assert mock_session.post.call_count == 3
    payload = mock_session.post.call_args_list[0][1]["json"]
    assert payload["fresh_until"] == 2000000000


def test_ingest_documents_failure():
    """Test ingest_documents failure bubbles up exception."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=500, ok=False)
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_session.post.return_value = mock_resp
    
    docs = [{"doc_id": "d1"}]
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.ingest_documents(mock_session, "http://url", "ing_1", "ds_1", "job_1", docs, batch_size=1)


# 21. complete_job
def test_complete_job_success():
    """Test complete_job patches status correctly."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_session.patch.return_value = mock_resp
    ingest.complete_job(mock_session, "http://url", "job_1")
    mock_session.patch.assert_called_once()


def test_complete_job_failure():
    """Test complete_job raises exception on failure."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=500, ok=False)
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_session.patch.return_value = mock_resp
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.complete_job(mock_session, "http://url", "job_1")


# 22. run_sample_queries
def test_run_sample_queries_success():
    """Test run_sample_queries executes search query and parses scores."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_resp.json.return_value = [{"score": 0.85, "document": {"page_content": "Doc Text", "metadata": {"title": "T1"}}}]
    mock_session.post.return_value = mock_resp
    
    rows = [{"question": "Q?", "answer": "A1", "supporting_facts": {"title": ["T1"]}}]
    ingest.run_sample_queries(mock_session, "http://url", "ds_1", rows, num_queries=1, query_limit=1)
    mock_session.post.assert_called_once()


def test_run_sample_queries_empty_results():
    """Test run_sample_queries with empty search result."""
    mock_session = mock.Mock()
    mock_resp = mock.Mock(status_code=200, ok=True)
    mock_resp.json.return_value = []
    mock_session.post.return_value = mock_resp
    
    rows = [{"question": "Q?", "answer": "A1", "supporting_facts": {"title": ["T1"]}}]
    ingest.run_sample_queries(mock_session, "http://url", "ds_1", rows, num_queries=1, query_limit=1)
    mock_session.post.assert_called_once()


# 23. main workflow
def test_main_workflow_skip_ingest(tmp_path):
    """Test main workflow with --skip-ingest flag enabled."""
    q_file = tmp_path / "questions.jsonl"
    with open(q_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": "q1", "user_input": "Q1", "reference": "A1", "category": "comparison", "level": "easy"}) + "\n")
        
    mock_args = argparse.Namespace(
        rag_url="http://localhost:8000",
        config="distractor",
        split="validation",
        limit=5,
        resume_offset=0,
        datasource_id="test_ds",
        datasource_name="test_ds_name",
        batch_size=10,
        num_queries=0,
        query_limit=3,
        reset=False,
        skip_ingest=True,
        output_jsonl=str(tmp_path / "out_q.jsonl"),
        output_csv=str(tmp_path / "out_q.csv"),
        pool_output_jsonl=str(tmp_path / "out_pool.jsonl"),
        pool_output_csv=str(tmp_path / "out_pool.csv"),
        input_file=None,
        input_questions_file=str(q_file),
        use_oidc=False,
        start_batch=1,
        limit_per_category=None,
        prioritize_reference=False
    )
    
    with mock.patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.fetch_hotpotqa_rows", return_value=[]), \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.run_sample_queries") as mock_queries:
             
        ingest.main()
        mock_queries.assert_called_once()


def test_main_workflow_reset_flow(tmp_path):
    """Test main workflow with --reset flag enabled."""
    q_file = tmp_path / "questions.jsonl"
    with open(q_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": "q1",
            "user_input": "Q1",
            "reference": "A1",
            "category": "comparison",
            "level": "easy",
            "context": {"title": ["Title A"], "sentences": [["Content A."]]}
        }) + "\n")

    mock_args = argparse.Namespace(
        rag_url="http://localhost:8000",
        config="distractor",
        split="validation",
        limit=5,
        resume_offset=0,
        datasource_id="test_ds",
        datasource_name="test_ds_name",
        batch_size=10,
        num_queries=0,
        query_limit=3,
        reset=True,
        skip_ingest=False,
        output_jsonl=str(tmp_path / "out_q.jsonl"),
        output_csv=str(tmp_path / "out_q.csv"),
        pool_output_jsonl=str(tmp_path / "out_pool.jsonl"),
        pool_output_csv=str(tmp_path / "out_pool.csv"),
        input_file=None,
        input_questions_file=str(q_file),
        use_oidc=False,
        start_batch=1,
        limit_per_category=None,
        prioritize_reference=False
    )

    with mock.patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.register_ingestor", return_value=("ing_1", 100)), \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.delete_datasource") as mock_del, \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.upsert_datasource") as mock_upsert, \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.create_job", return_value="job_1") as mock_job, \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.ingest_documents") as mock_ingest, \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.complete_job") as mock_complete, \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.get_existing_doc_ids", return_value=set()):

        ingest.main()
        mock_del.assert_called_once()
        mock_upsert.assert_called_once()
        mock_job.assert_called_once()
        mock_ingest.assert_called_once()
        mock_complete.assert_called_once()


def test_main_workflow_local_input(tmp_path):
    """Test main workflow with local document and questions file inputs."""
    q_file = tmp_path / "questions.jsonl"
    doc_file = tmp_path / "docs.jsonl"

    q_data = {
        "question_id": "q1",
        "user_input": "Is this a question?",
        "reference": "Yes",
        "category": "comparison",
        "level": "hard",
        "supporting_facts": [{"title": "Title A", "sent_id": 0}],
        "expected_doc_ids": ["docA"]
    }
    doc_data = {
        "document_id": "docA",
        "title": "Title A",
        "content": "Content A"
    }

    with open(q_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(q_data) + "\n")
    with open(doc_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc_data) + "\n")

    mock_args = argparse.Namespace(
        rag_url="http://localhost:8000",
        config="distractor",
        split="validation",
        limit=5,
        resume_offset=0,
        datasource_id="test_ds",
        datasource_name="test_ds_name",
        batch_size=10,
        num_queries=1,
        query_limit=3,
        reset=False,
        skip_ingest=False,
        output_jsonl=str(tmp_path / "out_q.jsonl"),
        output_csv=str(tmp_path / "out_q.csv"),
        pool_output_jsonl=str(tmp_path / "out_pool.jsonl"),
        pool_output_csv=str(tmp_path / "out_pool.csv"),
        input_file=str(doc_file),
        input_questions_file=str(q_file),
        use_oidc=False,
        start_batch=2,
        limit_per_category=1,
        prioritize_reference=True
    )

    mock_resp_ok = mock.Mock(status_code=200, ok=True)
    mock_resp_ok.json.return_value = {
        "status": "ok",
        "job_id": "job_1",
        "ingestor_id": "ing_1",
        "max_documents_per_ingest": 100,
        "documents": [{"document_id": "docB"}],
        "has_more": False
    }

    with mock.patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         mock.patch("requests.Session.post", return_value=mock_resp_ok) as mock_post, \
         mock.patch("requests.Session.get", return_value=mock_resp_ok) as mock_get, \
         mock.patch("requests.Session.patch", return_value=mock_resp_ok) as mock_patch_method, \
         mock.patch("ragas_eval.hotpotqa_rag_ingest.run_sample_queries") as mock_queries:

        ingest.main()

        assert mock_post.call_count >= 1
        assert mock_get.call_count >= 1
        assert mock_patch_method.call_count >= 1
        mock_queries.assert_called_once()


def test_main_workflow_document_limit(tmp_path):
    """Test that main workflow limits loaded questions to num_questions and slices documents to limit."""
    q_file = tmp_path / "questions.jsonl"
    doc_file = tmp_path / "docs.jsonl"
    
    # Write 2 questions
    with open(q_file, "w") as f:
        for i in range(2):
            f.write(json.dumps({
                "id": f"id_{i}",
                "question": f"Q{i}",
                "answer": f"A{i}",
                "supporting_facts": {"title": [f"T{i}"], "sent_id": [0]},
                "type": "bridge",
                "level": "hard"
            }) + "\n")
            
    # Write 10 docs
    with open(doc_file, "w") as f:
        for i in range(10):
            f.write(json.dumps({"title": f"T{i}", "content": f"Content {i}"}) + "\n")
            
    # Limit to 5 documents
    mock_args = argparse.Namespace(
        rag_url="http://localhost:8000",
        config="distractor",
        split="validation",
        limit=5,
        resume_offset=0,
        datasource_id="test_ds",
        datasource_name="test_ds_name",
        batch_size=10,
        num_queries=0,
        query_limit=3,
        reset=False,
        skip_ingest=True,
        output_jsonl=str(tmp_path / "out_q.jsonl"),
        output_csv=str(tmp_path / "out_q.csv"),
        pool_output_jsonl=str(tmp_path / "out_pool.jsonl"),
        pool_output_csv=str(tmp_path / "out_pool.csv"),
        input_file=str(doc_file),
        input_questions_file=str(q_file),
        use_oidc=False,
        start_batch=1,
        limit_per_category=None,
        prioritize_reference=False
    )
    
    with mock.patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        ingest.main()
        
    # Verify generated document pool contains exactly 5 documents (limit=5)
    with open(tmp_path / "out_pool.jsonl", "r") as f:
        pool_lines = f.readlines()
    assert len(pool_lines) == 5
    
    # Verify generated question set contains exactly 2 questions (num_questions=2)
    with open(tmp_path / "out_q.jsonl", "r") as f:
        q_lines = f.readlines()
    assert len(q_lines) == 2
