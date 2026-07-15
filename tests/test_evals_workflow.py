import os
import json
from pathlib import Path
import unittest.mock as mock
import pytest
import pandas as pd

from ragas_eval import evals
from ragas_eval.config import settings


class MockExperimentResults(list):
    def __init__(self, data=None, name="mock_exp"):
        super().__init__(data or [])
        self.name = name



@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    evals.cleanup_evaluator()


def test_init_evaluator_positive():
    """Test that init_evaluator correctly applies CLI arguments to configure Settings overrides."""
    args = mock.Mock()
    args.openai_api_key = "api-override-key"
    args.openai_endpoint = "http://override-endpoint/v1"
    args.openai_model_name = "override-model"
    args.embeddings_model = "override-embeddings"
    args.retrieval_only = True
    args.generation_only = False
    args.datasource = "mock"
    args.top_k = 8
    args.limit_per_category = 2
    args.compute_model_eval = False
    args.questions_path = None

    with mock.patch("ragas_eval.evals.OpenAI"), mock.patch(
        "ragas_eval.evals.llm_factory"
    ), mock.patch(
        "ragas_eval.evals.embedding_factory"
    ), mock.patch.dict(
        os.environ, {}, clear=True
    ):

        evals.init_evaluator(args)

        assert os.environ["OPENAI_API_KEY"] == "api-override-key"
        assert os.environ["OPENAI_ENDPOINT"] == "http://override-endpoint/v1"
        assert os.environ["OPENAI_MODEL_NAME"] == "override-model"
        assert os.environ["EMBEDDINGS_MODEL"] == "override-embeddings"
        assert evals.rag_eval_top_k == 8


def test_init_evaluator_precomputed():
    """Test that init_evaluator configures PrecomputedRAG when compute_model_eval is enabled."""
    args = mock.Mock()
    args.openai_api_key = "api-override-key"
    args.openai_endpoint = "http://override-endpoint/v1"
    args.openai_model_name = "override-model"
    args.embeddings_model = "override-embeddings"
    args.retrieval_only = False
    args.generation_only = False
    args.datasource = "mock"
    args.top_k = 3
    args.limit_per_category = None
    args.compute_model_eval = True
    args.questions_path = None
    args.agentic = False

    with mock.patch("ragas_eval.evals.OpenAI"), mock.patch(
        "ragas_eval.evals.llm_factory"
    ), mock.patch("ragas_eval.evals.embedding_factory"), mock.patch(
        "ragas_eval.precomputed_rag.PrecomputedRAG"
    ) as mock_precomputed_class:

        mock_dataset = mock.Mock()
        mock_dataset.samples = [mock.Mock()]
        evals.init_evaluator(args, dataset=mock_dataset)
        mock_precomputed_class.assert_called_once_with(
            preloaded_samples=mock_dataset.samples
        )


def test_init_evaluator_with_real_ragas_dataset():
    """Test that init_evaluator accepts a real ragas.Dataset and doesn't raise AttributeError."""
    from ragas import Dataset
    args = mock.Mock()
    args.openai_api_key = "api-override-key"
    args.openai_endpoint = "http://override-endpoint/v1"
    args.openai_model_name = "override-model"
    args.embeddings_model = "override-embeddings"
    args.retrieval_only = False
    args.generation_only = False
    args.datasource = "mock"
    args.top_k = 3
    args.limit_per_category = None
    args.compute_model_eval = True
    args.questions_path = None
    args.agentic = False

    # Instantiate a real ragas.Dataset and add a mock sample
    real_dataset = Dataset(name="test_dataset", backend="local/csv", root_dir="evals")
    real_dataset.append({"question": "Q1", "user_input": "Q1", "reference": "Ref1"})

    with mock.patch("ragas_eval.evals.OpenAI"), \
         mock.patch("ragas_eval.evals.llm_factory"), \
         mock.patch("ragas_eval.evals.embedding_factory"), \
         mock.patch("ragas_eval.precomputed_rag.PrecomputedRAG") as mock_precomputed_class, \
         mock.patch.dict(os.environ, {}, clear=True):

        evals.init_evaluator(args, dataset=real_dataset)
        # Check that PrecomputedRAG is called with list(real_dataset)
        mock_precomputed_class.assert_called_once_with(
            preloaded_samples=[{"question": "Q1", "user_input": "Q1", "reference": "Ref1"}]
        )


def test_init_evaluator_llm_factory_parameters():
    """Test that init_evaluator calls llm_factory with base_url instead of the unsupported client parameter."""
    args = mock.Mock()
    args.openai_api_key = "api-override-key"
    args.openai_endpoint = "http://override-endpoint/v1"
    args.openai_model_name = "override-model"
    args.embeddings_model = "override-embeddings"
    args.retrieval_only = False
    args.generation_only = False
    args.datasource = "mock"
    args.top_k = 3
    args.limit_per_category = None
    args.compute_model_eval = False
    args.questions_path = None

    with mock.patch("ragas_eval.evals.OpenAI"), \
         mock.patch("ragas_eval.evals.llm_factory") as mock_llm_factory, \
         mock.patch("ragas_eval.evals.embedding_factory"), \
         mock.patch.dict(os.environ, {}, clear=True):

        evals.init_evaluator(args)
        
        # Verify that client parameter is NOT passed to llm_factory, but base_url is
        called_args, called_kwargs = mock_llm_factory.call_args
        assert "client" not in called_kwargs
        assert called_kwargs["base_url"] == "http://override-endpoint/v1"
        assert called_kwargs["model"] == "override-model"


def test_init_evaluator_llm_wrapper_attributes():
    """Test that init_evaluator correctly configures max_tokens and temperature on LangchainLLMWrapper."""
    args = mock.Mock()
    args.openai_api_key = "api-override-key"
    args.openai_endpoint = "http://override-endpoint/v1"
    args.openai_model_name = "qwen3.5-35b"
    args.embeddings_model = "override-embeddings"
    args.retrieval_only = False
    args.generation_only = False
    args.datasource = "mock"
    args.top_k = 3
    args.limit_per_category = None
    args.compute_model_eval = False
    args.questions_path = None

    mock_wrapper = mock.Mock()
    del mock_wrapper.model_args  # ensure it has no model_args
    mock_inner_llm = mock.Mock()
    mock_inner_llm.max_tokens = 0
    mock_inner_llm.temperature = 0.0
    mock_inner_llm.model_kwargs = {}
    mock_wrapper.langchain_llm = mock_inner_llm

    with mock.patch("ragas_eval.evals.OpenAI"), \
         mock.patch("ragas_eval.evals.llm_factory", return_value=mock_wrapper), \
         mock.patch("ragas_eval.evals.embedding_factory"), \
         mock.patch.dict(os.environ, {}, clear=True):

        evals.init_evaluator(args)
        
        assert mock_inner_llm.max_tokens == 8192
        assert mock_inner_llm.temperature == 0.2
        assert "options" in mock_inner_llm.model_kwargs


def test_init_evaluator_negative():
    """Test that init_evaluator handles empty/missing overrides gracefully by preserving current config settings."""
    # Negative: initialization with empty strings preserves fallback configuration values
    args = mock.Mock()
    args.openai_api_key = ""
    args.openai_endpoint = ""
    args.openai_model_name = ""
    args.embeddings_model = ""
    args.retrieval_only = False
    args.generation_only = False
    args.datasource = "mock"
    args.top_k = 3
    args.limit_per_category = None
    args.compute_model_eval = False
    args.questions_path = None

    with mock.patch("ragas_eval.evals.OpenAI"), mock.patch(
        "ragas_eval.evals.llm_factory"
    ), mock.patch("ragas_eval.evals.embedding_factory"), mock.patch.dict(
        os.environ, {}, clear=True
    ):

        evals.init_evaluator(args)
        assert os.environ["OPENAI_API_KEY"] == settings.openai_api_key
        assert os.environ["OPENAI_ENDPOINT"] == settings.openai_endpoint
        assert os.environ["OPENAI_MODEL_NAME"] == settings.openai_model_name


def test_load_enterprise_rag_bench_positive(tmp_path):
    """Test that load_enterprise_rag_bench successfully reads benchmark questions from JSONL data."""
    # Positive: reading from jsonl file
    jsonl_file = tmp_path / "questions.jsonl"
    data = [
        {
            "user_input": "Q1",
            "reference": "Ref1",
            "expected_doc_ids": ["doc1"],
            "category": "cat1",
        },
        {
            "user_input": "Q2",
            "reference": "Ref2",
            "expected_doc_ids": ["doc2"],
            "category": "cat2",
        },
    ]
    with open(jsonl_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    results = evals.load_enterprise_rag_bench(jsonl_file)
    assert len(results) == 2
    assert results[0]["question"] == "Q1"
    assert results[0]["expected_doc_ids"] == ["doc1"]


def test_load_enterprise_rag_bench_negative():
    """Test that load_enterprise_rag_bench returns an empty list for a non-existent file path."""
    # Negative: non-existent file path returns empty list
    results = evals.load_enterprise_rag_bench(Path("non_existent_file.jsonl"))
    assert results == []


def test_load_dataset_mock():
    """Test that load_dataset correctly loads mock datasource questions into Ragas Dataset."""
    # Positive: loading from "mock" datasource
    with mock.patch("ragas_eval.evals.Dataset") as mock_dataset_class:
        mock_dataset = mock.Mock()
        mock_dataset_class.return_value = mock_dataset

        settings.ragas_datasource = "mock"
        settings.limit_per_category = None

        evals.load_dataset(limit=2, datasource="mock")

        # Verify dataset save is called
        mock_dataset.save.assert_called_once()
        # Verify appends
        assert mock_dataset.append.call_count == 2


def test_load_dataset_category_limits(tmp_path):
    """Test that load_dataset respects category-specific limits when loading questions."""
    # Positive: testing category limit slice filters
    jsonl_file = tmp_path / "questions.jsonl"
    data = [
        {
            "user_input": "Q1",
            "reference": "Ref1",
            "expected_doc_ids": ["doc1"],
            "category": "cat1",
        },
        {
            "user_input": "Q2",
            "reference": "Ref2",
            "expected_doc_ids": ["doc2"],
            "category": "cat1",
        },
        {
            "user_input": "Q3",
            "reference": "Ref3",
            "expected_doc_ids": ["doc3"],
            "category": "cat1",
        },
        {
            "user_input": "Q4",
            "reference": "Ref4",
            "expected_doc_ids": ["doc4"],
            "category": "cat2",
        },
    ]
    with open(jsonl_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    with mock.patch("ragas_eval.evals.Dataset") as mock_dataset_class, mock.patch(
        "ragas_eval.evals.load_enterprise_rag_bench", return_value=data
    ):
        mock_dataset = mock.Mock()
        mock_dataset_class.return_value = mock_dataset

        settings.ragas_datasource = "enterprise_rag_bench"
        settings.limit_per_category = 2
        settings.questions_path = str(jsonl_file)

        evals.load_dataset(
            limit=None,
            datasource="enterprise_rag_bench",
            limit_per_category=2,
            questions_path=str(jsonl_file),
        )

        assert mock_dataset.append.call_count == 3


def test_load_dataset_invalid():
    """Test that load_dataset raises a ValueError when an invalid/unsupported datasource name is provided."""
    # Negative: invalid datasource name throws ValueError
    with pytest.raises(ValueError, match="Unsupported datasource"):
        evals.load_dataset(datasource="invalid_source")


def test_load_dataset_hotpotqa(tmp_path):
    """Test that load_dataset correctly loads hotpotqa datasource questions."""
    jsonl_file = tmp_path / "hotpotqa_questions.jsonl"
    data = [
        {
            "user_input": "Q1",
            "reference": "Ref1",
            "expected_doc_ids": ["doc1"],
            "category": "cat1",
        }
    ]
    with open(jsonl_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    with mock.patch("ragas_eval.evals.Dataset") as mock_dataset_class:
        mock_dataset = mock.Mock()
        mock_dataset_class.return_value = mock_dataset

        evals.load_dataset(
            limit=None,
            datasource="hotpotqa",
            limit_per_category=None,
            questions_path=str(jsonl_file),
        )

        mock_dataset.save.assert_called_once()
        assert mock_dataset.append.call_count == 1


@pytest.mark.asyncio
async def test_run_experiment_retrieval_only():
    """Test that run_experiment yields 'N/A' response and returns retrieved contexts when retrieval_only is enabled."""
    # Positive: retrieval only skips LLM query generations
    evals.rag_eval_retrieval_only = True
    evals.rag_eval_top_k = 2
    settings.rag_eval_retrieval_only = True
    settings.rag_eval_top_k = 2

    mock_rag = mock.Mock()
    mock_rag.retrieve_documents.return_value = [
        {"content": "Ctx A", "metadata": {"doc_id": "docA"}},
        {"content": "Ctx B", "metadata": {"doc_id": "docB"}},
    ]
    evals.rag_client = mock_rag

    row = {"question": "Test query", "reference": "Ref A"}
    result = await evals.run_experiment(row)

    assert result["response"] == "N/A"
    assert result["retrieved_contexts"] == ["Ctx A", "Ctx B"]
    assert result["retrieved_doc_ids"] == ["docA", "docB"]
    mock_rag.retrieve_documents.assert_called_once_with("Test query", top_k=2)


@pytest.mark.asyncio
async def test_run_experiment_generation():
    """Test that run_experiment queries the RAG system and successfully returns generated answer, context, and usage."""
    # Positive: full query generation returns LLM answer and token usage
    evals.rag_eval_retrieval_only = False
    evals.rag_eval_top_k = 3
    settings.rag_eval_retrieval_only = False
    settings.rag_eval_top_k = 3

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "This is generated answer",
        "retrieved_docs": [{"content": "Ctx X", "metadata": {"doc_id": "docX"}}],
        "usage": {"total_tokens": 150},
        "logs": "log_path.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Test query", "reference": "Ref X"}
    result = await evals.run_experiment(row)

    assert result["response"] == "This is generated answer"
    assert result["retrieved_contexts"] == ["Ctx X"]
    assert result["retrieved_doc_ids"] == ["docX"]
    assert result["total_tokens"] == 150
    mock_rag.query.assert_called_once_with("Test query", top_k=3)


@pytest.mark.asyncio
async def test_run_experiment_negative():
    """Test that run_experiment correctly propagates exceptions raised by the underlying RAG system."""
    # Negative: pipeline failure bubbles up the exception
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False
    mock_rag = mock.Mock()
    mock_rag.query.side_effect = RuntimeError("Failed pipeline")
    evals.rag_client = mock_rag

    row = {"question": "Test query", "reference": "Ref X"}
    with pytest.raises(RuntimeError, match="Failed pipeline"):
        await evals.run_experiment(row)


@pytest.mark.asyncio
async def test_evals_main_workflow(tmp_path):
    """Test the complete main evaluation workflow including generation, batch evaluation, and result saving."""
    import argparse

    mock_args = argparse.Namespace(
        limit=1,
        datasource="mock",
        openai_api_key="test_key",
        openai_endpoint="http://localhost:4000/v1",
        openai_model_name="test_model",
        embeddings_model="test_embeddings",
        retrieval_only=False,
        generation_only=False,
        limit_per_category=None,
        top_k=3,
        compute_model_eval=False,
        questions_path=None,
    )

    with mock.patch(
        "argparse.ArgumentParser.parse_args", return_value=mock_args
    ), mock.patch("ragas_eval.evals.init_evaluator"), mock.patch(
        "ragas_eval.evals.load_dataset"
    ) as mock_load_ds, mock.patch(
        "ragas_eval.evals.run_experiment.arun"
    ) as mock_arun, mock.patch(
        "ragas_eval.evals.evaluate"
    ) as mock_evaluate:

        # Mock dataset loaded
        mock_ds = mock.Mock()
        mock_load_ds.return_value = mock_ds

        # Mock experiment generation results
        class MockExperimentResults(list):
            def __init__(self, data, name):
                """Initializes the MockExperimentResults list subclass with a given name."""
                super().__init__(data)
                self.name = name

        mock_arun.return_value = MockExperimentResults(
            [
                {
                    "question": "test question",
                    "response": "test answer",
                    "retrieved_contexts": ["context text"],
                    "retrieved_doc_ids": ["doc_1"],
                    "reference": "test reference",
                    "expected_doc_ids": ["doc_1"],
                    "latency": 1.5,
                    "total_tokens": 100,
                    "log_file": "log.json",
                }
            ],
            name="experiment_test_run",
        )

        # Mock Ragas evaluate output
        mock_scores_df = pd.DataFrame(
            [
                {
                    "factual_correctness": 0.9,
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.85,
                    "context_precision": 0.95,
                    "context_recall": 1.0,
                }
            ]
        )
        mock_results = mock.Mock()
        mock_results.to_pandas.return_value = mock_scores_df
        mock_evaluate.return_value = mock_results

        # Mock output file write operations
        with mock.patch("pandas.DataFrame.to_csv") as mock_to_csv, mock.patch(
            "builtins.open", mock.mock_open()
        ) as mock_file_open:

            # Run the main orchestrator
            await evals.main()

            # Assert evaluate was invoked on constructed dataset
            mock_evaluate.assert_called_once()
            mock_to_csv.assert_called_once()
            assert mock_file_open.call_count == 2


@pytest.mark.asyncio
async def test_evals_main_workflow_negative():
    """Test that main workflow raises an exception if the dataset loading step encounters an error."""
    # Negative: main fails when dataset loading raises exception
    import argparse

    mock_args = argparse.Namespace(
        limit=1,
        datasource="mock",
        openai_api_key="test_key",
        openai_endpoint="http://localhost:4000/v1",
        openai_model_name="test_model",
        embeddings_model="test_embeddings",
        retrieval_only=False,
        generation_only=False,
        limit_per_category=None,
        top_k=3,
        compute_model_eval=False,
        questions_path=None,
    )

    with mock.patch(
        "argparse.ArgumentParser.parse_args", return_value=mock_args
    ), mock.patch("ragas_eval.evals.init_evaluator"), mock.patch(
        "ragas_eval.evals.load_dataset", side_effect=RuntimeError("Dataset error")
    ):

        with pytest.raises(RuntimeError, match="Dataset error"):
            await evals.main()


def test_patched_ragas_evaluator_llm_create_drop_params():
    """Test that patched_ragas_evaluator_llm_create strips 'n' parameter and adds 'drop_params' to extra_headers."""
    kwargs = {
        "temperature": 0.5,
        "top_p": 0.9,
        "n": 3,
        "messages": [{"role": "user", "content": "hello"}],
    }
    
    with mock.patch("ragas_eval.evals.ragas_prompt_tokens", 0), \
         mock.patch("ragas_eval.evals.ragas_completion_tokens", 0), \
         mock.patch("ragas_eval.evals.original_create") as mock_original_create:
        
        evals.patched_ragas_evaluator_llm_create(**kwargs)
        
        called_kwargs = mock_original_create.call_args[1]
        assert "top_p" not in called_kwargs  # both temperature and top_p were set
        assert "n" not in called_kwargs  # n parameter should be stripped
        assert called_kwargs["extra_headers"]["drop_params"] == "true"


@pytest.mark.asyncio
async def test_cleanup_evaluator_positive():
    """Test that cleanup_evaluator correctly sets rag_client to None."""
    # Positive: set rag_client to a mock then cleanup
    evals.rag_client = mock.Mock()
    assert evals.rag_client is not None
    evals.cleanup_evaluator()
    assert evals.rag_client is None


@pytest.mark.asyncio
async def test_cleanup_evaluator_negative():
    """Test that cleanup_evaluator handles already None rag_client gracefully."""
    # Negative: rag_client is already None
    evals.rag_client = None
    evals.cleanup_evaluator()
    assert evals.rag_client is None


@pytest.mark.asyncio
async def test_load_dataset_with_questions_path(tmp_path):
    """Test that load_dataset correctly loads from questions_path with limit_per_category."""
    jsonl_file = tmp_path / "questions.jsonl"
    data = [
        {"user_input": "Q1", "reference": "Ref1", "category": "cat1"},
        {"user_input": "Q2", "reference": "Ref2", "category": "cat1"},
        {"user_input": "Q3", "reference": "Ref3", "category": "cat2"},
    ]
    with open(jsonl_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    with mock.patch("ragas_eval.evals.Dataset") as mock_dataset_class, \
         mock.patch("ragas_eval.evals.load_enterprise_rag_bench") as mock_load_bench:
        mock_load_bench.return_value = data
        mock_dataset = mock.Mock()
        mock_dataset_class.return_value = mock_dataset

        evals.load_dataset(
            limit=2,
            datasource="enterprise_rag_bench",
            limit_per_category=1,
            questions_path=str(jsonl_file),
        )

        mock_dataset.save.assert_called_once()


@pytest.mark.asyncio
async def test_load_dataset_limit_applied():
    """Test that load_dataset correctly applies the limit parameter."""
    with mock.patch("ragas_eval.evals.Dataset") as mock_dataset_class:
        mock_dataset = mock.Mock()
        mock_dataset_class.return_value = mock_dataset

        settings.ragas_datasource = "mock"
        settings.limit_per_category = None

        evals.load_dataset(limit=5, datasource="mock")

        mock_dataset.save.assert_called_once()


@pytest.mark.asyncio
async def test_run_experiment_retrieval_only_no_docs():
    """Test run_experiment with retrieval_only and empty retrieved documents."""
    evals.rag_eval_retrieval_only = True
    evals.rag_eval_top_k = 2
    settings.rag_eval_retrieval_only = True
    settings.rag_eval_top_k = 2

    mock_rag = mock.Mock()
    mock_rag.retrieve_documents.return_value = []
    evals.rag_client = mock_rag

    row = {"question": "Test query", "reference": "Ref A"}
    result = await evals.run_experiment(row)

    assert result["response"] == "N/A"
    assert result["retrieved_contexts"] == []
    assert result["retrieved_doc_ids"] == []


@pytest.mark.asyncio
async def test_run_experiment_retrieve_only_true():
    """Test run_experiment when ragas_eval_retrieval_only is True in evals module."""
    evals.rag_eval_retrieval_only = True
    evals.rag_eval_top_k = 1
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.retrieve_documents.return_value = [
        {"content": "Ctx Single", "metadata": {"doc_id": "docS"}}
    ]
    evals.rag_client = mock_rag

    row = {"question": "Single doc query", "reference": "Ref S"}
    result = await evals.run_experiment(row)

    assert result["response"] == "N/A"
    assert result["retrieved_contexts"] == ["Ctx Single"]


@pytest.mark.asyncio
async def test_run_experiment_with_expected_doc_ids():
    """Test that run_experiment correctly extracts expected_doc_ids from the row."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Answer with expected",
        "retrieved_docs": [{"content": "Ctx E", "metadata": {"doc_id": "docE"}}],
        "usage": {"total_tokens": 50},
        "logs": "log_expected.json",
    }
    evals.rag_client = mock_rag

    row = {
        "question": "Query with expected docs",
        "reference": "Ref E",
        "expected_doc_ids": ["expected1", "expected2"],
    }
    result = await evals.run_experiment(row)

    assert result["expected_doc_ids"] == ["expected1", "expected2"]


@pytest.mark.asyncio
async def test_run_experiment_latency_tracking():
    """Test that run_experiment correctly tracks latency."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Fast answer",
        "retrieved_docs": [],
        "usage": {"total_tokens": 10},
        "logs": "log_latency.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Latency query", "reference": "Ref Lat"}
    result = await evals.run_experiment(row)

    assert result["latency"] > 0
    assert isinstance(result["latency"], float)


@pytest.mark.asyncio
async def test_patched_ragas_evaluator_llm_create_negative():
    """Test patched_ragas_evaluator_llm_create when tokens are non-zero."""
    kwargs = {
        "temperature": 0.5,
        "top_p": 0.9,
        "n": 3,
        "messages": [{"role": "user", "content": "hello"}],
    }

    with mock.patch("ragas_eval.evals.ragas_prompt_tokens", 100), \
         mock.patch("ragas_eval.evals.ragas_completion_tokens", 50), \
         mock.patch("ragas_eval.evals.original_create") as mock_original_create:

        evals.patched_ragas_evaluator_llm_create(**kwargs)

        mock_original_create.assert_called_once()


@pytest.mark.asyncio
async def test_main_workflow_compute_model_eval():
    """Test main workflow when compute_model_eval is enabled."""
    import argparse
    from ragas_eval.config import settings

    settings.rag_eval_retrieval_only = False
    settings.rag_eval_generation_only = False

    mock_args = argparse.Namespace(
        limit=1,
        datasource="mock",
        openai_api_key="test_key",
        openai_endpoint="http://localhost:4000/v1",
        openai_model_name="test_model",
        embeddings_model="test_embeddings",
        retrieval_only=False,
        generation_only=False,
        limit_per_category=None,
        top_k=3,
        compute_model_eval=True,
        questions_path="dummy_path.jsonl",
    )

    with mock.patch(
        "argparse.ArgumentParser.parse_args", return_value=mock_args
    ), mock.patch("ragas_eval.evals.init_evaluator") as mock_init_eval, mock.patch(
        "ragas_eval.evals.load_dataset"
    ) as mock_load_ds, mock.patch(
        "ragas_eval.evals.run_experiment.arun"
    ) as mock_arun, mock.patch(
        "ragas_eval.evals.evaluate"
    ) as mock_evaluate, mock.patch(
        "ragas_eval.precomputed_rag.PrecomputedRAG"
    ) as mock_precomputed, mock.patch(
        "pathlib.Path.rename"
    ), mock.patch(
        "pathlib.Path.exists", return_value=True
    ):

        mock_ds = mock.MagicMock()
        mock_ds.__iter__.return_value = [
            {
                "question": "Q1",
                "user_input": "Q1",
                "answer": "A1",
                "reference": "R1",
                "retrieved_contexts": ["C1"],
                "retrieved_doc_ids": ["d1"],
                "total_tokens": 10,
                "latency": 1.0,
                "log_file": " ",
            }
        ]
        mock_load_ds.return_value = mock_ds

        mock_arun.return_value = MockExperimentResults()
        mock_evaluate.return_value = mock.Mock()
        mock_precomputed.return_value = mock.Mock()

        await evals.main()

        mock_init_eval.assert_called_once()


@pytest.mark.asyncio
async def test_main_workflow_generation_only():
    """Test main workflow when generation_only is enabled."""
    import argparse
    from ragas_eval.config import settings

    settings.rag_eval_retrieval_only = False
    settings.rag_eval_generation_only = False

    mock_args = argparse.Namespace(
        limit=1,
        datasource="mock",
        openai_api_key="test_key",
        openai_endpoint="http://localhost:4000/v1",
        openai_model_name="test_model",
        embeddings_model="test_embeddings",
        retrieval_only=False,
        generation_only=True,
        limit_per_category=None,
        top_k=3,
        compute_model_eval=False,
        questions_path=None,
    )

    with mock.patch(
        "argparse.ArgumentParser.parse_args", return_value=mock_args
    ), mock.patch("ragas_eval.evals.init_evaluator"), mock.patch(
        "ragas_eval.evals.load_dataset"
    ) as mock_load_ds, mock.patch(
        "ragas_eval.evals.run_experiment.arun"
    ) as mock_arun, mock.patch(
        "ragas_eval.evals.evaluate"
    ) as mock_evaluate:

        mock_ds = mock.Mock()
        mock_load_ds.return_value = mock_ds

        mock_arun.return_value = MockExperimentResults()
        mock_evaluate.return_value = mock.Mock()

        await evals.main()

        mock_evaluate.assert_called_once()


@pytest.mark.asyncio
async def test_init_evaluator_with_agentic():
    """Test that init_evaluator configures AgenticRAG when agentic is enabled."""
    args = mock.Mock()
    args.openai_api_key = "api-override-key"
    args.openai_endpoint = "http://override-endpoint/v1"
    args.openai_model_name = "override-model"
    args.embeddings_model = "override-embeddings"
    args.retrieval_only = False
    args.generation_only = False
    args.datasource = "mock"
    args.top_k = 3
    args.limit_per_category = None
    args.compute_model_eval = False
    args.questions_path = None
    args.agentic = True
    args.agent_api_url = "http://agent-api-url"
    args.agent_api_timeout = 100.0
    args.enable_trace_log = False

    with mock.patch("ragas_eval.evals.OpenAI"), \
         mock.patch("ragas_eval.evals.llm_factory"), \
         mock.patch("ragas_eval.evals.embedding_factory"), \
         mock.patch("ragas_eval.evals.default_agentic_rag_client") as mock_agentic:

        evals.init_evaluator(args)

        mock_agentic.assert_called_once_with(
            logdir="evals/logs",
            agent_api_url="http://agent-api-url",
            timeout=100.0,
            insecure=False,
            trace_log=False,
        )



@pytest.mark.asyncio
async def test_load_dataset_empty_data():
    """Test that load_dataset handles empty data gracefully."""
    with mock.patch("ragas_eval.evals.Dataset") as mock_dataset_class:
        mock_dataset = mock.Mock()
        mock_dataset_class.return_value = mock_dataset

        settings.ragas_datasource = "mock"
        settings.limit_per_category = None

        evals.load_dataset(limit=0, datasource="mock")

        mock_dataset.save.assert_called_once()


@pytest.mark.asyncio
async def test_load_enterprise_rag_bench_valid_file(tmp_path):
    """Test load_enterprise_rag_bench with valid JSONL file containing various fields."""
    jsonl_file = tmp_path / "questions_full.jsonl"
    data = [
        {
            "user_input": "Q1",
            "reference": "Ref1",
            "expected_doc_ids": ["doc1"],
            "category": "cat1",
            "question_type": "type1",
        },
    ]
    with open(jsonl_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    results = evals.load_enterprise_rag_bench(jsonl_file)
    assert len(results) == 1
    assert "category" in results[0]


@pytest.mark.asyncio
async def test_load_dataset_no_limit():
    """Test load_dataset with no limit applied."""
    with mock.patch("ragas_eval.evals.Dataset") as mock_dataset_class:
        mock_dataset = mock.Mock()
        mock_dataset_class.return_value = mock_dataset

        settings.ragas_datasource = "mock"
        settings.limit_per_category = None

        evals.load_dataset(limit=None, datasource="mock")

        mock_dataset.save.assert_called_once()


@pytest.mark.asyncio
async def test_run_experiment_missing_fields():
    """Test run_experiment handles rows with missing fields gracefully."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Partial answer",
        "retrieved_docs": [],
        "usage": {"total_tokens": 0},
        "logs": "log_missing.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Test query"}
    result = await evals.run_experiment(row)

    assert "response" in result


@pytest.mark.asyncio
async def test_run_experiment_with_all_fields():
    """Test run_experiment with all expected fields in the row."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Full answer",
        "retrieved_docs": [{"content": "Ctx Full", "metadata": {"doc_id": "docF"}}],
        "usage": {"total_tokens": 200},
        "logs": "log_full.json",
    }
    evals.rag_client = mock_rag

    row = {
        "question": "Full query",
        "reference": "Ref Full",
        "expected_doc_ids": ["docF"],
        "category": "cat_full",
        "question_type": "type_full",
    }
    result = await evals.run_experiment(row)

    assert result["response"] == "Full answer"
    assert result["expected_doc_ids"] == ["docF"]
    assert result["category"] == "cat_full"


@pytest.mark.asyncio
async def test_run_experiment_retrieved_docs_none():
    """Test run_experiment when retrieved_docs is None."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Answer with None docs",
        "retrieved_docs": None,
        "usage": {"total_tokens": 0},
        "logs": "log_none.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "None docs query", "reference": "Ref None"}
    result = await evals.run_experiment(row)

    assert result["retrieved_contexts"] == []
    assert result["retrieved_doc_ids"] == []


@pytest.mark.asyncio
async def test_evals_main_workflow_empty_dataset():
    """Test evals main workflow with empty dataset."""
    import argparse
    from ragas_eval.config import settings

    settings.rag_eval_retrieval_only = False
    settings.rag_eval_generation_only = False

    mock_args = argparse.Namespace(
        limit=1,
        datasource="mock",
        openai_api_key="test_key",
        openai_endpoint="http://localhost:4000/v1",
        openai_model_name="test_model",
        embeddings_model="test_embeddings",
        retrieval_only=False,
        generation_only=False,
        limit_per_category=None,
        top_k=3,
        compute_model_eval=False,
        questions_path=None,
    )

    with mock.patch(
        "argparse.ArgumentParser.parse_args", return_value=mock_args
    ), mock.patch("ragas_eval.evals.init_evaluator"), mock.patch(
        "ragas_eval.evals.load_dataset"
    ) as mock_load_ds, mock.patch(
        "ragas_eval.evals.run_experiment.arun"
    ) as mock_arun, mock.patch(
        "ragas_eval.evals.evaluate"
    ) as mock_evaluate:

        mock_ds = mock.Mock()
        mock_ds.samples = []
        mock_load_ds.return_value = mock_ds

        mock_arun.return_value = MockExperimentResults()
        mock_evaluate.return_value = mock.Mock()

        await evals.main()


@pytest.mark.asyncio
async def test_run_experiment_with_latency():
    """Test that run_experiment captures accurate latency measurements."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Latency test answer",
        "retrieved_docs": [{"content": "Ctx Lat", "metadata": {"doc_id": "docLat"}}],
        "usage": {"total_tokens": 25},
        "logs": "log_latency_test.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Latency test", "reference": "Ref Lat Test"}
    result = await evals.run_experiment(row)

    assert result["latency"] > 0
    assert isinstance(result["latency"], (int, float))


@pytest.mark.asyncio
async def test_run_experiment_with_empty_answer():
    """Test run_experiment when the RAG returns an empty answer."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "",
        "retrieved_docs": [{"content": "Ctx Empty", "metadata": {"doc_id": "docE"}}],
        "usage": {"total_tokens": 0},
        "logs": "log_empty.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Empty answer query", "reference": "Ref Empty"}
    result = await evals.run_experiment(row)

    assert result["response"] == ""


@pytest.mark.asyncio
async def test_run_experiment_with_multiple_retrieved_docs():
    """Test run_experiment with multiple retrieved documents."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Multi doc answer",
        "retrieved_docs": [
            {"content": "Doc 1", "metadata": {"doc_id": "d1"}},
            {"content": "Doc 2", "metadata": {"doc_id": "d2"}},
            {"content": "Doc 3", "metadata": {"doc_id": "d3"}},
        ],
        "usage": {"total_tokens": 300},
        "logs": "log_multi.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Multi doc query", "reference": "Ref Multi"}
    result = await evals.run_experiment(row)

    assert len(result["retrieved_contexts"]) == 3
    assert result["retrieved_doc_ids"] == ["d1", "d2", "d3"]


@pytest.mark.asyncio
async def test_load_enterprise_rag_bench_with_empty_lines(tmp_path):
    """Test load_enterprise_rag_bench handles empty lines in JSONL file."""
    jsonl_file = tmp_path / "questions_empty_lines.jsonl"
    with open(jsonl_file, "w") as f:
        f.write('{"user_input": "Q1", "reference": "Ref1"}\n')
        f.write('\n')
        f.write('{"user_input": "Q2", "reference": "Ref2"}\n')

    results = evals.load_enterprise_rag_bench(jsonl_file)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_load_dataset_with_limit_per_category_zero():
    """Test load_dataset with limit_per_category set to 0."""
    with mock.patch("ragas_eval.evals.Dataset") as mock_dataset_class:
        mock_dataset = mock.Mock()
        mock_dataset_class.return_value = mock_dataset

        settings.ragas_datasource = "mock"
        settings.limit_per_category = 0

        evals.load_dataset(limit=None, datasource="mock")

        mock_dataset.save.assert_called_once()


@pytest.mark.asyncio
async def test_run_experiment_with_unicode_answer():
    """Test run_experiment handles unicode characters in answers."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Réponse avec des caractères unicode: 你好 🎉",
        "retrieved_docs": [{"content": "Ctx Unicode", "metadata": {"doc_id": "docU"}}],
        "usage": {"total_tokens": 50},
        "logs": "log_unicode.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Unicode query", "reference": "Ref Unicode"}
    result = await evals.run_experiment(row)

    assert "unicode" in result["response"].lower() or "réponse" in result["response"]


@pytest.mark.asyncio
async def test_run_experiment_with_large_output():
    """Test run_experiment handles very large response text."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    large_answer = "A " * 10000

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": large_answer,
        "retrieved_docs": [{"content": "Ctx Large", "metadata": {"doc_id": "docL"}}],
        "usage": {"total_tokens": 10000},
        "logs": "log_large.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Large output query", "reference": "Ref Large"}
    result = await evals.run_experiment(row)

    assert len(result["response"]) == 20000


@pytest.mark.asyncio
async def test_run_experiment_with_special_characters_in_question():
    """Test run_experiment handles special characters in question text."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Special chars answer",
        "retrieved_docs": [],
        "usage": {"total_tokens": 10},
        "logs": "log_special.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Question with $pecial chars: @#$%^&*()", "reference": "Ref Special"}
    result = await evals.run_experiment(row)

    assert result["response"] == "Special chars answer"


@pytest.mark.asyncio
async def test_run_experiment_with_log_file():
    """Test that run_experiment includes log_file in the result."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Log file answer",
        "retrieved_docs": [],
        "usage": {"total_tokens": 10},
        "logs": "/path/to/log_file.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Log file query", "reference": "Ref Log"}
    result = await evals.run_experiment(row)

    assert result["log_file"] == "/path/to/log_file.json"


@pytest.mark.asyncio
async def test_run_experiment_with_none_usage():
    """Test run_experiment when usage is None."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "None usage answer",
        "retrieved_docs": [],
        "usage": None,
        "logs": "log_none_usage.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "None usage query", "reference": "Ref NoneUsage"}
    result = await evals.run_experiment(row)

    assert result["total_tokens"] == 0


@pytest.mark.asyncio
async def test_run_experiment_with_empty_retrieved_docs():
    """Test run_experiment when retrieved_docs is an empty list."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Empty docs answer",
        "retrieved_docs": [],
        "usage": {"total_tokens": 0},
        "logs": "log_empty_docs.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Empty docs query", "reference": "Ref EmptyDocs"}
    result = await evals.run_experiment(row)

    assert result["retrieved_contexts"] == []
    assert result["retrieved_doc_ids"] == []


@pytest.mark.asyncio
async def test_run_experiment_with_missing_answer_key():
    """Test run_experiment handles missing answer key in response."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "retrieved_docs": [{"content": "Ctx", "metadata": {"doc_id": "docM"}}],
        "usage": {"total_tokens": 0},
        "logs": "log_missing_ans.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Missing answer query", "reference": "Ref MissingAns"}
    result = await evals.run_experiment(row)

    assert "response" in result


@pytest.mark.asyncio
async def test_run_experiment_with_metadata_parsing():
    """Test run_experiment handles metadata with doc_id correctly."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Metadata parsing answer",
        "retrieved_docs": [
            {"content": "Doc A", "metadata": {"doc_id": "id_a", "title": "Title A"}},
            {"content": "Doc B", "metadata": {"doc_id": "id_b"}},
        ],
        "usage": {"total_tokens": 20},
        "logs": "log_metadata.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Metadata query", "reference": "Ref Metadata"}
    result = await evals.run_experiment(row)

    assert result["retrieved_doc_ids"] == ["id_a", "id_b"]


@pytest.mark.asyncio
async def test_run_experiment_with_empty_metadata():
    """Test run_experiment when metadata dict is empty."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Empty metadata answer",
        "retrieved_docs": [{"content": "Doc", "metadata": {}}],
        "usage": {"total_tokens": 5},
        "logs": "log_empty_meta.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Empty metadata query", "reference": "Ref EmptyMeta"}
    result = await evals.run_experiment(row)

    assert result["retrieved_doc_ids"] == []


@pytest.mark.asyncio
async def test_run_experiment_with_complex_metadata():
    """Test run_experiment with complex metadata structures."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Complex metadata answer",
        "retrieved_docs": [
            {
                "content": "Doc",
                "metadata": {
                    "doc_id": "complex_id",
                    "title": "Complex Title",
                    "author": "Author Name",
                    "date": "2024-01-01",
                    "tags": ["tag1", "tag2"],
                },
            }
        ],
        "usage": {"total_tokens": 15},
        "logs": "log_complex_meta.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Complex metadata query", "reference": "Ref ComplexMeta"}
    result = await evals.run_experiment(row)

    assert result["retrieved_doc_ids"] == ["complex_id"]


@pytest.mark.asyncio
async def test_run_experiment_with_single_char_document():
    """Test run_experiment with a single character document."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Single char answer",
        "retrieved_docs": [{"content": "x", "metadata": {"doc_id": "single"}}],
        "usage": {"total_tokens": 2},
        "logs": "log_single_char.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Single char query", "reference": "Ref SingleChar"}
    result = await evals.run_experiment(row)

    assert result["retrieved_doc_ids"] == ["single"]


@pytest.mark.asyncio
async def test_run_experiment_with_duplicate_doc_ids():
    """Test run_experiment handles duplicate doc_ids in retrieved_docs."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Duplicate docs answer",
        "retrieved_docs": [
            {"content": "Doc A", "metadata": {"doc_id": "dup"}},
            {"content": "Doc A again", "metadata": {"doc_id": "dup"}},
        ],
        "usage": {"total_tokens": 10},
        "logs": "log_dup.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Duplicate docs query", "reference": "Ref Dup"}
    result = await evals.run_experiment(row)

    assert result["retrieved_doc_ids"] == ["dup", "dup"]


@pytest.mark.asyncio
async def test_run_experiment_with_none_metadata():
    """Test run_experiment when metadata is None."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "None metadata answer",
        "retrieved_docs": [{"content": "Doc", "metadata": None}],
        "usage": {"total_tokens": 5},
        "logs": "log_none_meta.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "None metadata query", "reference": "Ref NoneMeta"}
    result = await evals.run_experiment(row)

    assert result["retrieved_doc_ids"] == []


@pytest.mark.asyncio
async def test_run_experiment_with_large_top_k():
    """Test run_experiment with a very large top_k value."""
    evals.rag_eval_retrieval_only = False
    settings.rag_eval_retrieval_only = False

    mock_rag = mock.Mock()
    mock_rag.query.return_value = {
        "answer": "Large top_k answer",
        "retrieved_docs": [
            {"content": f"Doc {i}", "metadata": {"doc_id": f"doc_{i}"}}
            for i in range(10)
        ],
        "usage": {"total_tokens": 100},
        "logs": "log_large_top_k.json",
    }
    evals.rag_client = mock_rag

    row = {"question": "Large top_k query", "reference": "Ref LargeTopK"}
    result = await evals.run_experiment(row)

    assert len(result["retrieved_doc_ids"]) == 10


@pytest.mark.asyncio
async def test_init_evaluator_llm_generation_wrapping():
    """Test that init_evaluator wraps/patches agenerate_text and generate_text for bedrock/anthropic models."""
    from langchain_core.outputs import LLMResult, Generation
    
    args = mock.Mock()
    args.openai_api_key = "api-override-key"
    args.openai_endpoint = "http://override-endpoint/v1"
    args.openai_model_name = "bedrock/anthropic.claude-v2"
    args.embeddings_model = "override-embeddings"
    args.retrieval_only = False
    args.generation_only = False
    args.datasource = "mock"
    args.top_k = 3
    args.limit_per_category = None
    args.compute_model_eval = False
    args.questions_path = None

    mock_wrapper = mock.Mock()
    del mock_wrapper.model_args  # ensure it has no model_args
    mock_inner_llm = mock.Mock()
    mock_inner_llm.max_tokens = 0
    mock_inner_llm.temperature = 0.0
    mock_inner_llm.model_kwargs = {}
    mock_wrapper.langchain_llm = mock_inner_llm

    async def mock_agenerate_text(prompt, n=1, **kwargs):
        return LLMResult(generations=[[Generation(text="gen_1")]])

    def mock_generate_text(prompt, n=1, **kwargs):
        return LLMResult(generations=[[Generation(text="gen_1")]])

    mock_wrapper.agenerate_text = mock_agenerate_text
    mock_wrapper.generate_text = mock_generate_text

    with mock.patch("ragas_eval.evals.OpenAI"), \
         mock.patch("ragas_eval.evals.llm_factory", return_value=mock_wrapper), \
         mock.patch("ragas_eval.evals.embedding_factory"), \
         mock.patch.dict(os.environ, {}, clear=True):

        evals.init_evaluator(args)
        
        # Verify agenerate_text intercepts n > 1 and returns combined generations
        combined_async_res = await mock_wrapper.agenerate_text("prompt", n=3)
        assert len(combined_async_res.generations[0]) == 3
        assert [g.text for g in combined_async_res.generations[0]] == ["gen_1", "gen_1", "gen_1"]

        # Verify generate_text intercepts n > 1 and returns combined generations
        combined_sync_res = mock_wrapper.generate_text("prompt", n=3)
        assert len(combined_sync_res.generations[0]) == 3
        assert [g.text for g in combined_sync_res.generations[0]] == ["gen_1", "gen_1", "gen_1"]



