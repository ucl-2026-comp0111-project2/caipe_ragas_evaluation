import unittest.mock as mock
import pytest
import requests

from ragas_eval.rag import SimpleKeywordRetriever, CaipeRetriever


def test_simple_keyword_retriever_positive():
    """Test that SimpleKeywordRetriever correctly fits documents and retrieves correct matched elements."""
    # Positive: successful fit and keyword count retrieval
    retriever = SimpleKeywordRetriever()
    docs = [
        "Python is a popular programming language.",
        "Java is class-based and object-oriented.",
        "Python code is clean and readable."
    ]
    retriever.fit(docs)
    
    results = retriever.get_top_k("python", k=2)
    assert len(results) == 2
    doc_indices = [idx for idx, score in results]
    assert 0 in doc_indices
    assert 2 in doc_indices


def test_simple_keyword_retriever_negative():
    """Test that SimpleKeywordRetriever handles empty queries and non-matching search words gracefully."""
    # Negative: query matching no words returns empty scores
    retriever = SimpleKeywordRetriever()
    retriever.fit(["Document about Ragas"])
    results = retriever.get_top_k("unrelated keyword", k=3)
    # The keyword match count score should be 0
    assert results[0][1] == 0

    # Negative: fit with empty documents list
    retriever.fit([])
    results = retriever.get_top_k("any query", k=3)
    assert len(results) == 0


def test_caipe_retriever_oidc_token_positive():
    """Test that CaipeRetriever correctly prioritizes explicitly provided OIDC tokens."""
    # Positive: token from initialization parameter takes priority
    retriever = CaipeRetriever(token="explicit-token")
    assert retriever.token == "explicit-token"


def test_caipe_retriever_oidc_token_from_settings(monkeypatch):
    """Test that CaipeRetriever loads the OIDC token fallback from settings when not explicitly provided."""
    # Positive: token loaded from settings if not passed explicitly
    from ragas_eval.config import settings
    monkeypatch.setattr(settings, "caipe_oidc_token", "settings-token")
    
    retriever = CaipeRetriever()
    assert retriever.token == "settings-token"


def test_caipe_retriever_oidc_token_missing():
    """Test that retrieving token raises a ValueError if no OIDC token is specified anywhere."""
    # Negative: raises ValueError if no token is provided
    from ragas_eval.config import settings
    retriever = CaipeRetriever(token=None)
    # temporarily force settings token to None if it's set
    with mock.patch.object(settings, "caipe_oidc_token", None):
        with pytest.raises(ValueError, match="OIDC Token is required"):
            _ = retriever.token


def test_caipe_retriever_get_top_k_positive():
    """Test that get_top_k successfully queries remote CAIPE endpoint and extracts results and metadata."""
    # Positive: query vector search successfully parses documents and metadata
    retriever = CaipeRetriever("http://localhost:9999", "ds_1")
    
    with mock.patch.object(CaipeRetriever, "token", new_callable=mock.PropertyMock) as mock_token, \
         mock.patch("requests.post") as mock_post:
        
        mock_token.return_value = "token-123"
        mock_resp = mock.Mock()
        mock_resp.json.return_value = [
            {
                "score": 0.88,
                "document": {
                    "page_content": "Found doc text.",
                    "document_id": "doc_id_9"
                }
            }
        ]
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp
        
        results = retriever.get_top_k("query test", k=1)
        assert len(results) == 1
        assert results[0][1] == 0.88
        assert retriever.documents[results[0][0]] == "Found doc text."
        assert retriever.documents_metadata[results[0][0]] == {"doc_id": "doc_id_9"}


def test_caipe_retriever_get_top_k_negative():
    """Test that get_top_k handles server HTTP errors gracefully by returning an empty results list."""
    # Negative: REST query connection error catches exception and returns empty list
    retriever = CaipeRetriever("http://localhost:9999", "ds_1")
    
    with mock.patch.object(CaipeRetriever, "token", new_callable=mock.PropertyMock) as mock_token, \
         mock.patch("requests.post") as mock_post:
        
        mock_token.return_value = "token-123"
        mock_post.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        results = retriever.get_top_k("query test", k=3)
        assert len(results) == 0


def test_base_retriever_methods():
    """Test BaseRetriever fit and get_top_k directly."""
    from ragas_eval.rag import BaseRetriever
    retriever = BaseRetriever()
    
    # Positive: fit stores documents
    retriever.fit(["doc1", "doc2"])
    assert retriever.documents == ["doc1", "doc2"]
    
    # Negative: BaseRetriever get_top_k raises NotImplementedError
    with pytest.raises(NotImplementedError):
        retriever.get_top_k("query")


def test_caipe_retriever_fit():
    """Test CaipeRetriever.fit works and initializes documents/metadata lists."""
    # Positive: fit documents
    retriever = CaipeRetriever()
    retriever.fit(["docA", "docB"])
    assert retriever.documents == ["docA", "docB"]
    assert retriever.documents_metadata == [{}, {}]


def test_caipe_retriever_parse_items():
    """Test CaipeRetriever._parse_items parsing logic."""
    retriever = CaipeRetriever()
    
    # Positive: results is a list
    assert retriever._parse_items([{"item": 1}]) == [{"item": 1}]
    
    # Positive: results is a dict
    assert retriever._parse_items({"results": [{"item": 2}]}) == [{"item": 2}]
    
    # Negative: results is invalid type (e.g. None or str)
    assert retriever._parse_items(None) == []
    assert retriever._parse_items("invalid") == []


def test_caipe_retriever_extract_doc_content():
    """Test CaipeRetriever._extract_doc_content with different structures."""
    retriever = CaipeRetriever()
    
    # Positive: page_content in item
    assert retriever._extract_doc_content({"page_content": "hello"}, {}) == "hello"
    
    # Positive: page_content in doc_obj
    assert retriever._extract_doc_content({}, {"page_content": "world"}) == "world"
    
    # Positive: content in metadata
    assert retriever._extract_doc_content({"metadata": {"content": "meta-content"}}, {}) == "meta-content"
    
    # Negative: content completely missing
    assert retriever._extract_doc_content({}, {}) == ""


def test_caipe_retriever_extract_doc_id():
    """Test CaipeRetriever._extract_doc_id handles diverse metadata configurations."""
    retriever = CaipeRetriever()
    
    # Positive: document_id in doc_obj
    assert retriever._extract_doc_id({}, {"document_id": "id1"}) == "id1"
    
    # Positive: doc_id in item metadata
    assert retriever._extract_doc_id({"metadata": {"doc_id": "id2"}}, {}) == "id2"
    
    # Negative: document ID not present
    assert retriever._extract_doc_id({}, {}) is None

