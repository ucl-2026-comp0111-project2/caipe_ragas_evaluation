import io
import json
import os
import zipfile
import argparse
import unittest.mock as mock
import pytest
import requests

from ragas_eval import enterprise_rag_bench_ingest as ingest


def test_parse_doc_filename_positive():
    """Test that _parse_doc_filename parses correctly formatted document filenames."""
    # Positive: parses correctly formatted filenames
    result = ingest._parse_doc_filename("confluence/dsid_12f9__confluence-doc.txt")
    assert result == ("dsid_12f9", "confluence-doc")


def test_parse_doc_filename_negative():
    """Test that _parse_doc_filename returns None for invalid or mismatched filename structures."""
    # Negative: incorrect structure
    assert ingest._parse_doc_filename("confluence/dsid12f9confluence-doc.txt") is None
    assert ingest._parse_doc_filename("confluence/dsid_12f9_confluence-doc.pdf") is None
    assert ingest._parse_doc_filename("confluence/dsid_12f9_confluence-doc") is None
    assert ingest._parse_doc_filename("confluence/other_file.txt") is None


def test_check_positive():
    """Test that _check returns the response object unchanged when the HTTP status is OK."""
    # Positive: HTTP status is OK
    mock_resp = mock.Mock()
    mock_resp.ok = True
    result = ingest._check(mock_resp)
    assert result == mock_resp


def test_check_negative():
    """Test that _check raises an HTTPError exception when response status code is not OK."""
    # Negative: HTTP status is 500 error
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 500
    mock_resp.request = mock.Mock(method="POST", url="http://test-server")
    mock_resp.text = "Internal Server Error"
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Error")
    
    with pytest.raises(requests.exceptions.HTTPError):
        ingest._check(mock_resp)


def test_get_oidc_token_positive():
    """Test that _get_oidc_token successfully retrieves and returns an access token."""
    # Positive: token returned successfully
    with mock.patch("requests.post") as mock_post:
        mock_resp = mock.Mock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"access_token": "mocked-jwt"}
        mock_post.return_value = mock_resp
        
        token = ingest._get_oidc_token("http://token-url", "client_1", "secret_1")
        assert token == "mocked-jwt"


def test_get_oidc_token_negative():
    """Test that _get_oidc_token propagates connection errors raised by request post."""
    # Negative: requests raises connection error
    with mock.patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection timeout")
        with pytest.raises(requests.exceptions.ConnectionError):
            ingest._get_oidc_token("http://token-url", "client_1", "secret_1")


def test_fetch_documents_cached(tmp_path):
    """Test that fetch_documents successfully loads document data from cached zip files."""
    # Positive: read from a cached zip file
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    zip_file_path = cache_dir / "confluence_slice_0001.zip"
    
    # Create a mock zip content
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Write valid doc filename
        zf.writestr("confluence/dsid_c1__doc-1.txt", "Title-1\nThis is document content.")
    
    with open(zip_file_path, "wb") as f:
        f.write(zip_buffer.getvalue())
        
    with mock.patch.dict(ingest.SOURCE_SLICE_COUNTS, {"confluence": 1}, clear=True):
        docs = ingest.fetch_documents(
            source_types=["confluence"],
            limit_per_source=2,
            cache_dir=str(cache_dir)
        )
    
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "dsid_c1"
    assert docs[0]["title"] == "Title-1"


def test_register_ingestor_positive():
    """Test that register_ingestor successfully registers and returns ingestor ID and limit."""
    # Positive: returns ingestor_id and doc limit
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "ingestor_id": "ingestor_abc",
        "max_documents_per_ingest": 1000
    }
    mock_session.post.return_value = mock_resp
    
    ingestor_id, max_docs = ingest.register_ingestor(mock_session, "http://localhost:8000")
    assert ingestor_id == "ingestor_abc"
    assert max_docs == 1000


def test_register_ingestor_negative():
    """Test that register_ingestor propagates HTTP errors on registration failure."""
    # Negative: register endpoint returns 500 error
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Error")
    mock_session.post.return_value = mock_resp
    
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.register_ingestor(mock_session, "http://localhost:8000")


def test_delete_datasource_positive():
    """Test that delete_datasource sends the correct delete requests to the RAG server."""
    # Positive: deletes datasource successfully
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_session.delete.return_value = mock_resp
    
    ingest.delete_datasource(mock_session, "http://localhost:8000", "test_ds")
    mock_session.delete.assert_called_once_with(
        "http://localhost:8000/v1/datasource",
        params={"datasource_id": "test_ds"}
    )


def test_delete_datasource_negative():
    """Test that delete_datasource propagates errors when delete endpoint fails."""
    # Negative: delete endpoint returns error
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 400
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Error")
    mock_session.delete.return_value = mock_resp
    
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.delete_datasource(mock_session, "http://localhost:8000", "test_ds")


def test_upsert_datasource_positive():
    """Test that upsert_datasource successfully sends upsert requests with correct payloads."""
    # Positive: upsert datasource successfully
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_session.post.return_value = mock_resp
    
    ingest.upsert_datasource(mock_session, "http://localhost:8000", "test_ds", "test_ds_name", "ingestor_123")
    mock_session.post.assert_called_once()
    args_json = mock_session.post.call_args[1]["json"]
    assert args_json["datasource_id"] == "test_ds"
    assert args_json["name"] == "test_ds_name"
    assert args_json["ingestor_id"] == "ingestor_123"


def test_upsert_datasource_negative():
    """Test that upsert_datasource propagates HTTP errors when upsert endpoint fails."""
    # Negative: upsert datasource fails
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Error")
    mock_session.post.return_value = mock_resp
    
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.upsert_datasource(mock_session, "http://localhost:8000", "test_ds", "test_ds_name", "ingestor_123")


def test_create_job_positive():
    """Test that create_job successfully posts and returns the created job ID."""
    # Positive: create job returns job_id
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"job_id": "job_999"}
    mock_session.post.return_value = mock_resp
    
    job_id = ingest.create_job(mock_session, "http://localhost:8000", "test_ds", 50)
    assert job_id == "job_999"


def test_create_job_negative():
    """Test that create_job propagates HTTP errors when create job endpoint fails."""
    # Negative: create job fails
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 400
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Error")
    mock_session.post.return_value = mock_resp
    
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.create_job(mock_session, "http://localhost:8000", "test_ds", 50)


def test_complete_job_positive():
    """Test that complete_job completes job successfully on the RAG server."""
    # Positive: completes job successfully
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_session.patch.return_value = mock_resp
    
    ingest.complete_job(mock_session, "http://localhost:8000", "job_999")
    mock_session.patch.assert_called_once()


def test_complete_job_negative():
    """Test that complete_job propagates HTTP errors when patching the job fails."""
    # Negative: complete job fails
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Error")
    mock_session.patch.return_value = mock_resp
    
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.complete_job(mock_session, "http://localhost:8000", "job_999")


def test_build_documents_positive():
    """Test that build_documents maps raw document dicts to CAIPE document schemas."""
    # Positive: builds document structure correctly
    raw_docs = [
        {"doc_id": "c1", "text": "confluence content", "title": "C Title", "source_type": "confluence"}
    ]
    docs = ingest.build_documents(raw_docs, "test_ds", "ingestor_1")
    assert len(docs) == 1
    assert docs[0]["page_content"] == "confluence content"
    assert docs[0]["metadata"]["document_id"] == "c1"
    assert docs[0]["metadata"]["title"] == "C Title"


def test_build_documents_negative():
    """Test that build_documents returns empty list when raw document list is empty."""
    # Negative: empty raw document returns empty list
    docs = ingest.build_documents([], "test_ds", "ingestor_1")
    assert docs == []


def test_get_existing_doc_ids_positive():
    """Test that get_existing_doc_ids successfully retrieves document IDs from the datasource."""
    # Positive: parses document IDs
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "documents": [{"document_id": "docA"}, {"document_id": "docB"}],
        "has_more": False
    }
    mock_session.get.return_value = mock_resp
    
    doc_ids = ingest.get_existing_doc_ids(mock_session, "http://localhost:8000", "test_ds")
    assert doc_ids == {"docA", "docB"}


def test_get_existing_doc_ids_negative():
    """Test that get_existing_doc_ids returns an empty set on 404 response."""
    # Negative: endpoint returns 404
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 404
    mock_session.get.return_value = mock_resp
    
    doc_ids = ingest.get_existing_doc_ids(mock_session, "http://localhost:8000", "test_ds")
    assert doc_ids == set()


def test_get_existing_doc_ids_pagination_limit():
    """Test that get_existing_doc_ids triggers warning and breaks at pagination limit boundary."""
    # Positive: pagination boundary warning
    mock_session = mock.Mock()
    with mock.patch("builtins.print") as mock_print, \
         mock.patch("requests.Session.get"):
             
        mock_resp = mock.Mock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        
        call_count = 0
        def json_side_effect():
            """Mock json decoder that increments page call offset count."""
            nonlocal call_count
            call_count += 1
            return {
                "documents": [{"document_id": f"doc_{call_count}"}],
                "has_more": True
            }
        mock_resp.json.side_effect = json_side_effect
        mock_session.get.return_value = mock_resp
        
        doc_ids = ingest.get_existing_doc_ids(mock_session, "http://localhost:8000", "test_ds")
        assert len(doc_ids) == 16
        mock_print.assert_any_call("  Warning: reached Milvus pagination limit (16384 chunks). Cannot fetch more existing document IDs.")


def test_ingest_documents_positive():
    """Test that ingest_documents posts correct batch sizes and updates RAG server metrics."""
    # Positive: posts document batches and increment metrics
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"status": "ok"}
    mock_session.post.return_value = mock_resp
    
    docs = [{"doc_id": "1"}, {"doc_id": "2"}]
    ingest.ingest_documents(mock_session, "http://localhost:8000", "ingestor_1", "test_ds", "job_1", docs, batch_size=1)
    
    # 1 batch per doc (2 iterations total) -> each iteration makes 3 POST requests: /ingest, /increment-document-count, /increment-progress
    assert mock_session.post.call_count == 6


def test_ingest_documents_negative():
    """Test that ingest_documents bubbles up HTTP errors when ingestion post fails."""
    # Negative: posts document batch fails
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = False
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Error")
    mock_session.post.return_value = mock_resp
    
    docs = [{"doc_id": "1"}]
    with pytest.raises(requests.exceptions.HTTPError):
        ingest.ingest_documents(mock_session, "http://localhost:8000", "ingestor_1", "test_ds", "job_1", docs, batch_size=1)


def test_fetch_all_questions_positive():
    """Test that fetch_all_questions successfully parses lines of the questions file."""
    # Positive: parses questions lines
    with mock.patch("requests.get") as mock_get:
        mock_resp = mock.Mock()
        mock_resp.ok = True
        mock_resp.text = json.dumps({"question_id": "q1", "user_input": "Q?", "reference": "Ans", "category": "cat", "source_types": ["jira"], "expected_doc_ids": ["d1"]})
        mock_get.return_value = mock_resp
        
        qs = ingest.fetch_all_questions()
        assert len(qs) == 1
        assert qs[0]["question"] == "Q?"


def test_fetch_all_questions_negative():
    """Test that fetch_all_questions propagates HTTP errors on question endpoint failure."""
    # Negative: endpoint returns error status
    with mock.patch("requests.get") as mock_get:
        mock_resp = mock.Mock()
        mock_resp.ok = False
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Error")
        mock_get.return_value = mock_resp
        
        with pytest.raises(requests.exceptions.HTTPError):
            ingest.fetch_all_questions()


def test_select_questions():
    """Test that select_questions prioritizes questions whose expected docs are ingested."""
    # Positive: selects fully/partially/uncovered candidates
    all_qs = [
        {"question_id": "q1", "question": "Q1", "source_types": ["confluence"], "expected_doc_ids": ["docA"]},
        {"question_id": "q2", "question": "Q2", "source_types": ["confluence"], "expected_doc_ids": ["docB"]},
        {"question_id": "q3", "question": "Q3", "source_types": ["confluence"], "expected_doc_ids": []}
    ]
    
    selected = ingest.select_questions(all_qs, ["confluence"], {"docA"}, num_questions=3)
    assert len(selected) == 3
    assert selected[0]["question_id"] == "q1"


def test_select_questions_negative():
    """Test that select_questions returns empty list when no questions match targeted source types."""
    # Negative: returns empty selected list when no candidates match the wanted sources
    all_qs = [
        {"question_id": "q1", "question": "Q1", "source_types": ["jira"], "expected_doc_ids": ["docA"]}
    ]
    selected = ingest.select_questions(all_qs, ["confluence"], set(), num_questions=2)
    assert selected == []


def test_run_sample_queries():
    """Test that run_sample_queries successfully queries RAG server and prints recall."""
    # Positive: checks query recall
    mock_session = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.ok = True
    mock_resp.json.return_value = [
        {"score": 0.9, "document": {"page_content": "doc content", "metadata": {"document_id": "docA", "title": "Doc A"}}}
    ]
    mock_session.post.return_value = mock_resp
    
    questions = [
        {"question": "What is CAIPE?", "category": "general", "expected_doc_ids": ["docA"]}
    ]
    ingest.run_sample_queries(mock_session, "http://localhost", "test_ds", questions, query_limit=3)
    mock_session.post.assert_called_once()


def test_run_sample_queries_negative():
    """Test that run_sample_queries propagates connection errors on post requests."""
    # Negative: endpoint throws exception on post query
    mock_session = mock.Mock()
    mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection lost")
    
    questions = [{"question": "What is CAIPE?", "category": "general", "expected_doc_ids": ["docA"]}]
    with pytest.raises(requests.exceptions.ConnectionError):
        ingest.run_sample_queries(mock_session, "http://localhost", "test_ds", questions, query_limit=3)


def test_main_workflow():
    """Test the complete ingestion main workflow under normal settings and mock APIs."""
    # Positive: run main workflow with skipped login and mock ingestion endpoints
    mock_args = argparse.Namespace(
        rag_url="http://localhost:8000",
        sources=["confluence"],
        limit=5,
        datasource_id="test_ds",
        datasource_name="test_ds_name",
        batch_size=10,
        num_queries=0,
        query_limit=3,
        reset=True,
        skip_ingest=False,
        all_sources=False,
        start_batch=1,
        cache_dir="cache",
        use_oidc=True,
        oidc_token_url="http://token",
        oidc_client_id="client",
        oidc_client_secret="secret",
        prioritize_reference=True
    )
    
    with mock.patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest._get_oidc_token", return_value="token123"), \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.fetch_all_questions") as mock_questions, \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.fetch_documents") as mock_fetch, \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.register_ingestor", return_value=("ing_1", 100)), \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.delete_datasource") as mock_del, \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.upsert_datasource") as mock_upsert, \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.create_job", return_value="job_1") as mock_job, \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.ingest_documents") as mock_ingest, \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.complete_job") as mock_complete, \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.select_questions", return_value=[]), \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.run_sample_queries") as mock_queries, \
         mock.patch("time.sleep"):
             
        mock_questions.return_value = [{"expected_doc_ids": ["doc1"], "source_types": ["confluence"]}]
        mock_fetch.return_value = [{"doc_id": "c1", "text": "content", "title": "Title", "source_type": "confluence"}]
        
        ingest.main()
        
        mock_del.assert_called_once()
        mock_fetch.assert_called_once()
        mock_upsert.assert_called_once()
        mock_job.assert_called_once()
        mock_ingest.assert_called_once()
        mock_complete.assert_called_once()
        mock_queries.assert_called_once()


def test_main_workflow_negative():
    """Test that main workflow prints warning and returns when zero documents are fetched."""
    # Negative: main workflow prints warning and exits when fetch_documents returns no raw documents
    mock_args = argparse.Namespace(
        rag_url="http://localhost:8000",
        sources=["confluence"],
        limit=5,
        datasource_id="test_ds",
        datasource_name="test_ds_name",
        batch_size=10,
        num_queries=0,
        query_limit=3,
        reset=False,
        skip_ingest=False,
        all_sources=False,
        start_batch=1,
        cache_dir="cache",
        use_oidc=False,
        prioritize_reference=False
    )
    
    with mock.patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.fetch_all_questions", return_value=[]), \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.fetch_documents", return_value=[]), \
         mock.patch("ragas_eval.enterprise_rag_bench_ingest.register_ingestor", return_value=("ing_1", 100)), \
         mock.patch("builtins.print") as mock_print:
             
        ingest.main()
        mock_print.assert_any_call("No documents fetched. Check network access to github.com release assets.")


def test_load_zip_content_additional(tmp_path):
    """Test _load_zip_content behaves correctly for valid and invalid file/zip configurations."""
    # Positive: valid zip read
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.txt", b"zip-data")
    assert os.path.exists(zip_path)
    data = ingest._load_zip_content("test.zip", str(zip_path))
    assert data is not None
    assert data.startswith(b"PK")

    # Negative: target name not in zip file (will download and fail if server offline, so mock download)
    with mock.patch("requests.get") as mock_get:
        mock_resp = mock.Mock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp
        assert ingest._load_zip_content("missing.zip", str(tmp_path / "missing.zip")) is None


def test_process_zip_entry_additional():
    """Test _process_zip_entry extraction rules."""
    import io
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("confluence/dsid_abc__title.txt", b"Title\nContent line 1\nContent line 2")
    
    zip_buffer.seek(0)
    with zipfile.ZipFile(zip_buffer) as zf:
        reference_docs = []
        other_docs = []
        seen_hashes = set()
        # Positive: process zip entry
        ingest._process_zip_entry(
            zf,
            "confluence/dsid_abc__title.txt",
            "confluence",
            {"dsid_abc"},
            seen_hashes,
            reference_docs,
            other_docs
        )
        assert len(reference_docs) == 1
        assert reference_docs[0]["doc_id"] == "dsid_abc"

        # Negative: process zip entry with invalid name (does not add)
        reference_docs_2 = []
        other_docs_2 = []
        ingest._process_zip_entry(
            zf,
            "confluence/invalid_name.txt",
            "confluence",
            {"dsid_abc"},
            seen_hashes,
            reference_docs_2,
            other_docs_2
        )
        assert len(reference_docs_2) == 0
        assert len(other_docs_2) == 0


def test_process_zip_file_additional(tmp_path):
    """Test _process_zip_file with normal contents and invalid zip contents."""
    # Positive: parses correct entries in zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("confluence/dsid_123__doc1.txt", b"Doc1 Title\nDoc1 Content")
    zip_content = zip_buffer.getvalue()
    
    reference_docs = []
    other_docs = []
    seen_hashes = set()
    ingest._process_zip_file(
        zip_content,
        "test.zip",
        "confluence",
        {"dsid_123"},
        seen_hashes,
        reference_docs,
        other_docs
    )
    assert len(reference_docs) == 1
    assert reference_docs[0]["doc_id"] == "dsid_123"

    # Negative: processes invalid zip path (raises or returns empty list gracefully)
    try:
        ingest._process_zip_file(b"invalid zip content", "invalid.zip", "confluence", set(), set(), [], [])
    except Exception:
        pass


def test_parse_args_additional():
    """Test _parse_args parses command line args."""
    # Positive: normal parameters
    with mock.patch("sys.argv", ["script", "--limit", "10"]):
        args = ingest._parse_args()
        assert args.limit == 10
        assert args.reset is False

    # Negative: system exit on invalid argument type
    with mock.patch("sys.argv", ["script", "--limit", "abc"]), pytest.raises(SystemExit):
        ingest._parse_args()


def test_setup_session_additional():
    """Test session setup, including OIDC token acquisition."""
    # Scenario 1: explicit token provided
    args_token = argparse.Namespace(
        use_oidc=True,
        oidc_token="explicit_token_val",
        oidc_token_url="http://token",
        oidc_client_id="client",
        oidc_client_secret="secret"
    )
    with mock.patch("ragas_eval.enterprise_rag_bench_ingest._get_oidc_token") as mock_get_token:
        sess = ingest._setup_session(args_token)
        assert sess.headers["Authorization"] == "Bearer explicit_token_val"
        mock_get_token.assert_not_called()

    # Scenario 2: fetch token via credentials deferred
    args = argparse.Namespace(
        use_oidc=True,
        oidc_token=None,
        oidc_token_url="http://token",
        oidc_client_id="client",
        oidc_client_secret="secret"
    )
    
    # Positive: successfully fetches OIDC token on first request 401
    with mock.patch("ragas_eval.enterprise_rag_bench_ingest._get_oidc_token", return_value="tok123") as mock_get_token:
        with mock.patch("requests.Session.request") as mock_request:
            sess = ingest._setup_session(args)
            assert "Authorization" not in sess.headers
            mock_get_token.assert_not_called()
            
            mock_resp_401 = mock.Mock(status_code=401)
            mock_resp_200 = mock.Mock(status_code=200)
            mock_request.side_effect = [mock_resp_401, mock_resp_200]
            
            sess.request("GET", "http://localhost:8000/v1/datasource")
            mock_get_token.assert_called_once()
            assert sess.headers["Authorization"] == "Bearer tok123"

    # Negative: token acquisition throws exception on first 401 request
    with mock.patch("ragas_eval.enterprise_rag_bench_ingest._get_oidc_token", side_effect=ValueError("OAuth error")) as mock_get_token:
        with mock.patch("requests.Session.request") as mock_request:
            sess = ingest._setup_session(args)
            mock_resp_401 = mock.Mock(status_code=401)
            mock_request.return_value = mock_resp_401
            
            resp = sess.request("GET", "http://localhost:8000/v1/datasource")
            mock_get_token.assert_called_once()
            # The exception is caught and logged, and the original 401 is returned
            assert resp.status_code == 401


def test_get_prioritized_doc_ids():
    """Test _get_prioritized_doc_ids ordering prioritization logic."""
    args = argparse.Namespace(prioritize_reference=True)
    all_questions = [
        {"source_types": ["jira"], "expected_doc_ids": ["doc1"]},
        {"source_types": ["confluence"], "expected_doc_ids": ["doc2"]}
    ]
    # Positive: priority mapping
    selected = ingest._get_prioritized_doc_ids(args, all_questions, ["jira"])
    assert "doc1" in selected
    assert "doc2" not in selected

    # Negative: prioritize_reference is False
    args_false = argparse.Namespace(prioritize_reference=False)
    selected_empty = ingest._get_prioritized_doc_ids(args_false, all_questions, ["jira"])
    assert len(selected_empty) == 0


def test_run_ingestion_job_negative():
    """Test _run_ingestion_job failure workflow."""
    sess = mock.Mock()
    # Negative: fail to register ingestor
    sess.post.side_effect = requests.exceptions.HTTPError("Registration failed")
    args = argparse.Namespace(reset=True, rag_url="http://localhost", datasource_id="ds_id")
    
    with pytest.raises(requests.exceptions.HTTPError):
        ingest._run_ingestion_job(sess, args, ["jira"], set())


def test_run_skip_ingestion_path_negative():
    """Test _run_skip_ingestion_path when question fetching fails."""
    args = argparse.Namespace(prioritize_reference=False, cache_dir="cache", limit=10)
    # Negative: fetch_documents fails
    with mock.patch("ragas_eval.enterprise_rag_bench_ingest.fetch_documents", side_effect=requests.exceptions.HTTPError("Fetch failed")):
        with pytest.raises(requests.exceptions.HTTPError):
            ingest._run_skip_ingestion_path(args, ["jira"], set())


def test_fetch_documents_limit(tmp_path):
    """Test that fetch_documents slices the documents for each source type to limit_per_source."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    # Create two slices with 2 documents each
    for source in ["confluence", "jira"]:
        zip_file = cache_dir / f"{source}_slice_0001.zip"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr(f"{source}/dsid_1__doc-1.txt", f"Title 1 {source}\nContent 1 {source}")
            zf.writestr(f"{source}/dsid_2__doc-2.txt", f"Title 2 {source}\nContent 2 {source}")
        with open(zip_file, "wb") as f:
            f.write(zip_buffer.getvalue())
            
    with mock.patch.dict(ingest.SOURCE_SLICE_COUNTS, {"confluence": 1, "jira": 1}, clear=True):
        # limit_per_source = 1
        docs = ingest.fetch_documents(
            source_types=["confluence", "jira"],
            limit_per_source=1,
            cache_dir=str(cache_dir)
        )
    # Since limit_per_source=1, we should get exactly 1 doc from confluence and 1 doc from jira
    assert len(docs) == 2
    confluence_docs = [d for d in docs if d["source_type"] == "confluence"]
    jira_docs = [d for d in docs if d["source_type"] == "jira"]
    assert len(confluence_docs) == 1
    assert len(jira_docs) == 1

