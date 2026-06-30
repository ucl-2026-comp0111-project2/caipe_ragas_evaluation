from pathlib import Path
import unittest.mock as mock
import pytest

from ragas_eval.rag import BaseRAG, SimpleKeywordRetriever, CaipeRetriever


def test_base_rag_document_handling():
    """Test that BaseRAG correctly adds, sets, and tracks documents in the retriever."""
    mock_llm = mock.Mock()
    retriever = SimpleKeywordRetriever()
    rag = BaseRAG(llm_client=mock_llm, model_name="mock-model", retriever=retriever, logdir="test_logs")
    
    assert len(rag.documents) == 0
    rag.add_documents(["Doc A", "Doc B"])
    assert len(rag.documents) == 2
    assert "Doc A" in rag.documents
    
    rag.set_documents(["Doc C"])
    assert len(rag.documents) == 1
    assert "Doc C" in rag.documents


def test_base_rag_retrieve_not_fitted_error():
    """Test that retrieving documents raises a ValueError when retriever has not been fitted, except for CaipeRetriever."""
    mock_llm = mock.Mock()
    retriever = SimpleKeywordRetriever()
    rag = BaseRAG(llm_client=mock_llm, model_name="mock-model", retriever=retriever, logdir="test_logs")
    # Negative: retrieve when not fitted raises ValueError
    with pytest.raises(ValueError, match="No documents have been added"):
        rag.retrieve_documents("test query")

    # Positive: CaipeRetriever allows retrieval even if not fitted locally
    caipe_retriever = CaipeRetriever()
    rag_caipe = BaseRAG(llm_client=mock_llm, model_name="mock-model", retriever=caipe_retriever, logdir="test_logs")
    with mock.patch.object(CaipeRetriever, "get_top_k", return_value=[]):
        docs = rag_caipe.retrieve_documents("test query")
        assert len(docs) == 0


def test_base_rag_generate_response_not_fitted_error():
    """Test that generate_response raises a ValueError when the RAG system is not yet fitted with documents."""
    mock_llm = mock.Mock()
    retriever = SimpleKeywordRetriever()
    rag = BaseRAG(llm_client=mock_llm, model_name="mock-model", retriever=retriever, logdir="test_logs")
    # Negative: generate_response when not fitted raises ValueError
    with pytest.raises(ValueError, match="No documents have been added"):
        rag.generate_response("test query")


def test_base_rag_generate_response_positive():
    """Test that generate_response successfully formats prompt and queries LLM to generate response text."""
    mock_llm = mock.Mock()
    mock_choice = mock.Mock()
    mock_choice.message.content = "This is a generated answer."
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_llm.chat.completions.create.return_value = mock_response

    retriever = SimpleKeywordRetriever()
    retriever.fit(["Doc X: Ragas focuses on evaluation metrics."])
    rag = BaseRAG(llm_client=mock_llm, model_name="mock-model", retriever=retriever, logdir="test_logs")
    rag.is_fitted = True
    
    answer = rag.generate_response("what does ragas focus on?", top_k=1)
    assert answer == "This is a generated answer."


def test_base_rag_query_workflow_positive(tmp_path):
    """Test successful query workflow including retrieval, generation, and log exporting steps."""
    # Positive RAG query workflow
    mock_llm = mock.Mock()
    mock_choice = mock.Mock()
    mock_choice.message.content = "Generated answer."
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_llm.chat.completions.create.return_value = mock_response

    retriever = SimpleKeywordRetriever()
    retriever.fit(["Doc Y"])
    
    log_dir = tmp_path / "rag_logs"
    rag = BaseRAG(llm_client=mock_llm, model_name="mock-model", retriever=retriever, logdir=str(log_dir))
    rag.is_fitted = True
    
    result = rag.query("Doc Y", top_k=1, run_id="run_123")
    assert result["answer"] == "Generated answer."
    assert len(result["retrieved_docs"]) == 1
    
    # Verify trace log file export
    log_file_path = Path(result["logs"])
    assert log_file_path.exists()


def test_base_rag_query_workflow_negative(tmp_path):
    """Test that query workflow handles LLM exceptions by logging the error and returning error description."""
    # Negative RAG query workflow: generation fails with LLM exception
    mock_llm = mock.Mock()
    mock_llm.chat.completions.create.side_effect = Exception("LLM connection timed out")

    retriever = SimpleKeywordRetriever()
    retriever.fit(["Doc Y"])
    
    log_dir = tmp_path / "rag_logs"
    rag = BaseRAG(llm_client=mock_llm, model_name="mock-model", retriever=retriever, logdir=str(log_dir))
    rag.is_fitted = True
    
    result = rag.query("Doc Y", top_k=1, run_id="run_123")
    assert "Error generating response" in result["answer"]
    
    # Verify trace logs logged the error
    log_file_path = Path(result["logs"])
    assert log_file_path.exists()


def test_default_rag_client_positive():
    """Test default_rag_client correctly instantiates and configures BaseRAG instances."""
    from ragas_eval.rag import default_rag_client, CaipeRetriever
    mock_llm = mock.Mock()
    
    # Positive: configure with use_caipe=True
    client_caipe = default_rag_client(mock_llm, logdir="test_logs", use_caipe=True, model_name="gpt-custom")
    assert isinstance(client_caipe.retriever, CaipeRetriever)
    assert client_caipe.model_name == "gpt-custom"
    
    # Positive: configure with use_caipe=False (instantiates local retriever and loads default docs)
    client_local = default_rag_client(mock_llm, logdir="test_logs", use_caipe=False)
    assert isinstance(client_local.retriever, SimpleKeywordRetriever)
    assert len(client_local.documents) > 0



def test_default_rag_client_negative():
    """Test default_rag_client fallback to settings model when no model name is specified."""
    # Negative: fallback verification is verified by calling without model name and checking it is set
    mock_llm = mock.Mock()
    from ragas_eval.rag import default_rag_client
    from ragas_eval.config import settings
    
    with mock.patch.object(settings, "openai_model_name", "qwen3.5-35b"):
        client = default_rag_client(mock_llm, use_caipe=True)
        assert client.model_name == "qwen3.5-35b"

