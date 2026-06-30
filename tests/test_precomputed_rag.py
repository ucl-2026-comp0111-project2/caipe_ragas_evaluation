import json
import unittest.mock as mock
import pytest

from ragas_eval.precomputed_rag import PrecomputedRAG, normalize_string


def test_normalize_string():
    """Test that normalize_string normalizes whitespace, casing, and handles invalid non-string types."""
    # Positive
    assert normalize_string("  Test  String!  ") == "test string!"
    # Negative / empty types
    assert normalize_string(None) == ""
    assert normalize_string(123) == ""


def test_precomputed_rag_file_not_found():
    """Test that PrecomputedRAG raises a FileNotFoundError when initialized with a missing file path."""
    # Negative: non-existent file path
    with pytest.raises(FileNotFoundError):
        PrecomputedRAG("non_existent_dataset.jsonl")


def test_precomputed_rag_preloaded():
    """Test that PrecomputedRAG correctly initializes and queries using preloaded dictionary list."""
    preloaded = [
        {"user_input": "What is CAIPE RAG?", "reference": "CAIPE RAG is an enterprise evaluation system.", "category": "basic"},
    ]
    mock_retriever = mock.Mock()
    mock_retriever.documents = ["CAIPE is a platform."]
    mock_retriever.documents_metadata = [{"doc_id": "caipe-doc"}]
    mock_retriever.get_top_k.return_value = [(0, 0.9)]

    with mock.patch("ragas_eval.precomputed_rag.CaipeRetriever") as mock_caipe_class:
        mock_caipe_class.return_value = mock_retriever
        pr = PrecomputedRAG(preloaded_samples=preloaded)
        res = pr.query("What is CAIPE RAG?")
        assert res["answer"] == "CAIPE RAG is an enterprise evaluation system."
        assert len(res["retrieved_docs"]) == 1


def test_precomputed_rag_lifecycle(tmp_path):
    """Test the complete PrecomputedRAG lifecycle including loading data and querying with exact/fuzzy matches."""
    # Positive: valid loading and query processing
    jsonl_file = tmp_path / "questions.jsonl"
    data = [
        {"user_input": "What is CAIPE RAG?", "reference": "CAIPE RAG is an enterprise evaluation system.", "category": "basic"},
        {"question": "How to deploy?", "reference": "Run scripts/deploy.sh.", "category": "infra"}
    ]
    with open(jsonl_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Mock CaipeRetriever retrieval
    # Mocking retriever docs list and metadata
    mock_retriever = mock.Mock()
    mock_retriever.documents = ["CAIPE is a platform."]
    mock_retriever.documents_metadata = [{"doc_id": "caipe-doc"}]
    mock_retriever.get_top_k.return_value = [(0, 0.9)]
    
    # Instantiate PrecomputedRAG
    with mock.patch("ragas_eval.precomputed_rag.CaipeRetriever") as mock_caipe_class:
        mock_caipe_class.return_value = mock_retriever
        pr = PrecomputedRAG(str(jsonl_file))
        
        # Positive Query: Exact lookup
        res = pr.query("What is CAIPE RAG?")
        assert res["answer"] == "CAIPE RAG is an enterprise evaluation system."
        assert len(res["retrieved_docs"]) == 1
        assert res["retrieved_docs"][0]["content"] == "CAIPE is a platform."
        assert res["retrieved_docs"][0]["metadata"] == {"doc_id": "caipe-doc"}
        
        # Positive Query: Fuzzy/substring lookup
        res_fuzzy = pr.query("caipe rag")
        assert res_fuzzy["answer"] == "CAIPE RAG is an enterprise evaluation system."
        
        # Negative Query: Missing/unrelated question
        res_missing = pr.query("totally unrelated question")
        assert res_missing["answer"] == "No precomputed answer available."
        assert len(res_missing["retrieved_docs"]) == 0


def test_precomputed_rag_load_from_samples_additional():
    """Test load_from_samples with various inputs (SingleTurnSample-like objects and invalid inputs)."""
    # Helper mock class for SingleTurnSample
    class MockSample:
        def __init__(self, user_input, reference, category):
            self.user_input = user_input
            self.reference = reference
            self.category = category

    # Positive: load from SingleTurnSample-like objects
    samples = [
        MockSample("What is Ragas?", "Ragas is an evaluation framework.", "basic"),
        {"user_input": "Mock dict question", "reference": "Mock dict answer", "category": "custom"},
    ]
    
    mock_retriever = mock.Mock()
    with mock.patch("ragas_eval.precomputed_rag.CaipeRetriever") as mock_caipe_class:
        mock_caipe_class.return_value = mock_retriever
        pr = PrecomputedRAG(preloaded_samples=samples)
        
        assert "what is ragas?" in pr.data_by_question
        assert pr.data_by_question["what is ragas?"]["answer"] == "Ragas is an evaluation framework."
        assert "mock dict question" in pr.data_by_question
        
        # Negative: sample without question/user_input (should be skipped)
        invalid_samples = [
            {"reference": "No question here"}
        ]
        pr.load_from_samples(invalid_samples)
        # Should not raise exception and list remains size 2
        assert len(pr.data_by_question) == 2


def test_precomputed_rag_load_dataset_additional(tmp_path):
    """Test load_dataset with different file formats and invalid dataset path."""
    mock_retriever = mock.Mock()
    with mock.patch("ragas_eval.precomputed_rag.CaipeRetriever") as mock_caipe_class:
        mock_caipe_class.return_value = mock_retriever
        
        # Negative: missing dataset path
        pr_no_path = PrecomputedRAG.__new__(PrecomputedRAG)
        pr_no_path.dataset_path = None
        with pytest.raises(ValueError, match="dataset_path must be provided"):
            pr_no_path.load_dataset()

        # Positive: load from json file
        json_file = tmp_path / "data.json"
        with open(json_file, "w") as f:
            json.dump([{"question": "JSON question", "reference": "JSON answer"}], f)
        pr_json = PrecomputedRAG.__new__(PrecomputedRAG)
        pr_json.dataset_path = str(json_file)
        pr_json.data_by_question = {}
        pr_json.load_dataset()
        assert "json question" in pr_json.data_by_question

        # Positive: load from csv file
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("question,reference\nCSV question,CSV answer\n,No question here\n")
        pr_csv = PrecomputedRAG.__new__(PrecomputedRAG)
        pr_csv.dataset_path = str(csv_file)
        pr_csv.data_by_question = {}
        pr_csv.load_dataset()
        assert "csv question" in pr_csv.data_by_question
        assert len(pr_csv.data_by_question) == 1  # empty question row was skipped


def test_precomputed_rag_retrieve_documents_negative():
    """Test retrieve_documents handles query not found in index."""
    mock_retriever = mock.Mock()
    with mock.patch("ragas_eval.precomputed_rag.CaipeRetriever") as mock_caipe_class:
        mock_caipe_class.return_value = mock_retriever
        pr = PrecomputedRAG(preloaded_samples=[])
        
        # Negative: query not in data_by_question
        docs = pr.retrieve_documents("not-found-query")
        assert docs == []

