import pytest
from pydantic import ValidationError

from ragas_eval.config import Settings


def test_config_defaults():
    """Test that default Settings values are correctly loaded when no env overrides exist."""
    # Positive test: check baseline default config loading
    settings = Settings()
    assert settings.openai_api_key == "mock-openai-key"
    assert settings.openai_endpoint == "http://localhost:4000/v1"
    assert settings.openai_model_name == "qwen3.5-35b"
    assert settings.embeddings_model == "bge-m3"
    assert settings.rag_eval_top_k == 10
    assert settings.rag_eval_retrieval_only is False
    assert settings.rag_eval_generation_only is False


def test_config_env_overrides(monkeypatch):
    """Test that environment variables successfully override the baseline default configurations."""
    # Positive test: check that environment variables successfully override defaults
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
    monkeypatch.setenv("RAG_EVAL_TOP_K", "10")
    monkeypatch.setenv("RAG_EVAL_RETRIEVAL_ONLY", "true")

    settings = Settings()
    assert settings.openai_model_name == "gpt-4o"
    assert settings.openai_endpoint == "https://api.openai.com/v1"
    assert settings.rag_eval_top_k == 10
    assert settings.rag_eval_retrieval_only is True


def test_config_type_casting(monkeypatch):
    """Test that configuration settings automatically cast environment strings to correct datatypes."""
    # Positive test: check automatic casting from env strings to correct types
    monkeypatch.setenv("RAG_EVAL_TOP_K", "5")
    monkeypatch.setenv("RAG_EVAL_RETRIEVAL_ONLY", "false")
    settings = Settings()
    assert settings.rag_eval_top_k == 5
    assert settings.rag_eval_retrieval_only is False


def test_config_invalid_type_error(monkeypatch):
    """Test that validation error is correctly raised when invalid types are supplied to Settings."""
    # Negative test: verify validation error is raised for invalid type mappings
    monkeypatch.setenv("RAG_EVAL_TOP_K", "not-an-integer")
    with pytest.raises(ValidationError):
        Settings()
