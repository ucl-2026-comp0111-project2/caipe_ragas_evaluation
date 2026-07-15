import argparse
import pytest
import json
import asyncio
import os
import pandas as pd
import unittest.mock as mock
from pathlib import Path
from ragas_eval import evals
from ragas_eval.metrics import ContainsAnswer, _normalize
from ragas.dataset_schema import SingleTurnSample


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    evals.cleanup_evaluator()


def test_sanitize_kwargs():
    # Positive
    kwargs = {"temperature": 0.5, "top_p": 0.9, "n": 2}
    sanitized = evals._sanitize_kwargs(kwargs)
    assert "top_p" not in sanitized
    assert "n" not in sanitized
    assert sanitized["extra_headers"]["drop_params"] == "true"

    # Negative: empty kwargs
    assert evals._sanitize_kwargs({}) == {"extra_headers": {"drop_params": "true"}}


def test_detect_expected_schema():
    # Positive: claims
    messages_claims = [{"role": "system", "content": "Extract list of 'claims'."}]
    schema, is_nli = evals._detect_expected_schema(messages_claims)
    assert schema == "claims"
    assert is_nli is False

    # Positive: statements
    messages_statements = [{"role": "system", "content": "verdict in 'statements'"}]
    schema2, is_nli2 = evals._detect_expected_schema(messages_statements)
    assert schema2 == "statements"
    assert is_nli2 is True

    # Negative: unrelated
    messages_unrelated = [{"role": "system", "content": "Translate text"}]
    schema3, is_nli3 = evals._detect_expected_schema(messages_unrelated)
    assert schema3 is None
    assert is_nli3 is False


def test_track_token_usage():
    # Positive: usage dict
    evals.ragas_prompt_tokens = 0
    evals.ragas_completion_tokens = 0
    mock_resp = mock.Mock()
    mock_resp.usage = {"prompt_tokens": 10, "completion_tokens": 5}
    evals._track_token_usage(mock_resp)
    assert evals.ragas_prompt_tokens == 10
    assert evals.ragas_completion_tokens == 5

    # Negative: usage is None does not raise exception
    mock_resp_none = mock.Mock()
    mock_resp_none.usage = None
    evals._track_token_usage(mock_resp_none)  # Should not raise


def test_parse_json_content():
    # Positive: valid json
    content, repaired = evals._parse_json_content('{"a": 1}')
    assert content == {"a": 1}
    assert repaired is not None
    assert json.loads(repaired) == {"a": 1}

    # Negative: unrepairable json (json_repair recovers it as a list of strings)
    content, repaired = evals._parse_json_content("invalid { json")
    assert content == ["json"]


def test_parse_json_content_with_escaped_quotes():
    # Test that valid JSON with nested escaped quotes inside a string value
    # is parsed correctly and not corrupted/split by unescaping logic.
    raw_json = """{
      "classifications": [
        {
          "statement": "The default limits are 10 MiB.",
          "reason": "The context explicitly states: \\"Enforce configurable limits: max_file_size (default 10MiB)\\" and \\"Default limits: 10MiB per file\\". This is supported.",
          "attributed": 1
        }
      ]
    }"""
    content, repaired = evals._parse_json_content(raw_json)
    assert content is not None
    assert "classifications" in content
    assert len(content["classifications"]) == 1
    item = content["classifications"][0]
    assert item["attributed"] == 1
    assert "Enforce configurable limits" in item["reason"]


def test_normalize_key_name():
    # Positive
    assert evals._normalize_key_name("Claims ") == "claims"
    # Negative: non-string raises AttributeError (as designed)
    with pytest.raises(AttributeError):
        evals._normalize_key_name(123)  # type: ignore


def test_clean_keys():
    # Positive
    obj = {" Claims": 1, "nested": {" Statements ": 2}}
    cleaned = evals._clean_keys(obj)
    assert "claims" in cleaned
    assert "statements" in cleaned["nested"]

    # Negative: non-dict returned as is
    assert evals._clean_keys([1, 2]) == [1, 2]


def test_sanitize_values():
    # Positive
    obj = {"verdict": "yes", "attributed": "1", "nested": {"verdict": "no"}}
    sanitized = evals._sanitize_values(obj)
    assert sanitized["verdict"] == 1
    assert sanitized["attributed"] == 1
    assert sanitized["nested"]["verdict"] == 0

    # Negative: non-dict/non-list/non-matching keys as is
    assert evals._sanitize_values("unaffected") == "unaffected"


def test_fill_dict_statement_keys():
    # Positive: fill default statement keys
    obj = {}
    evals._fill_dict_statement_keys(obj)
    assert obj["statement"] == "Statement not provided"
    assert obj["reason"] == "Reason not provided"
    assert obj["verdict"] == 0
    assert obj["attributed"] == 0

    # Negative: do not overwrite
    obj = {"statement": "ok", "reason": "why", "verdict": 1}
    evals._fill_dict_statement_keys(obj)
    assert obj["statement"] == "ok"
    assert obj["reason"] == "why"
    assert obj["verdict"] == 1


def test_fill_dict_standard_keys():
    # Positive
    obj = {"reason": "", "verdict": None, "attributed": None}
    evals._fill_dict_standard_keys(obj)
    assert obj["reason"] == "Reason not provided"
    assert obj["verdict"] == 0
    assert obj["attributed"] == 0

    # Negative: do not overwrite if already valid
    obj = {"reason": "ok", "verdict": 1}
    evals._fill_dict_standard_keys(obj)
    assert obj["reason"] == "ok"
    assert obj["verdict"] == 1


def test_fill_missing_keys():
    # Positive: statements
    obj = {"statement": ""}
    filled = evals._fill_missing_keys(obj, "statements", True)
    assert filled["statement"] == "Statement not provided"

    # Negative: non-dict
    assert evals._fill_missing_keys("not-dict", "statements", True) == "not-dict"


def test_normalize_claims():
    # Positive: list of dicts
    assert evals._normalize_claims([{"claim": "A"}]) == {"claims": ["A"]}
    # Negative: non-dict elements remain in the list
    assert evals._normalize_claims([123]) == {"claims": [123]}


def test_normalize_statements():
    # Positive: NLI mode
    res = evals._normalize_statements([{"statement": "A", "verdict": 1}], is_nli=True)
    assert res["statements"][0]["statement"] == "A"

    # Positive: non-NLI mode
    res2 = evals._normalize_statements(["A", {"statement": "B"}], is_nli=False)
    assert res2["statements"] == ["A", "B"]

    # Negative: invalid non-dict/non-string elements skipped in NLI mode
    res3 = evals._normalize_statements([123], is_nli=True)
    assert len(res3["statements"]) == 0


def test_is_matching_keys_heuristic():
    # Positive
    assert evals._is_matching_keys_heuristic(["statement", "verdict"]) is True
    # Negative
    assert evals._is_matching_keys_heuristic(["random_key"]) is False


def test_normalize_fallback():
    # Positive
    assert evals._normalize_fallback([{"statement": "A", "verdict": 1}]) == {
        "statements": [{"statement": "A", "verdict": 1}]
    }
    # Negative: no matches
    assert evals._normalize_fallback([{"random": "value"}]) == [{"random": "value"}]


def test_normalize_list_structure():
    # Positive: claims
    assert evals._normalize_list_structure([{"claim": "A"}], "claims", False) == {
        "claims": ["A"]
    }
    # Negative: empty top-level schema returns fallback
    assert evals._normalize_list_structure([123], None, False) == [123]


def test_rename_top_level_key():
    # Positive: rename key
    d = {"claims": [1]}
    evals._rename_top_level_key(d, "statements")
    assert "statements" in d
    assert "claims" not in d

    # Negative: missing original key
    d2 = {"other": 1}
    evals._rename_top_level_key(d2, "statements")
    assert d2 == {"other": 1}


def test_normalize_dict_claims():
    # Positive
    d = {"claims": [{"claim": "A"}]}
    evals._normalize_dict_claims(d)
    assert d["claims"] == ["A"]

    # Negative: empty dict
    d2 = {}
    evals._normalize_dict_claims(d2)
    assert d2 == {}


def test_normalize_dict_statements():
    # Positive: non-nli
    d = {"statements": [{"statement": "A"}]}
    evals._normalize_dict_statements(d, is_nli=False)
    assert d["statements"] == ["A"]

    # Negative: empty dict
    d2 = {}
    evals._normalize_dict_statements(d2, is_nli=True)
    assert d2 == {}


def test_normalize_dict_classifications():
    # Positive
    from typing import Any

    d: dict[str, list[Any]] = {"classifications": ["item1", {"attributed": 0}]}
    evals._normalize_dict_classifications(d)
    assert d["classifications"][0]["statement"] == "item1"
    assert d["classifications"][0]["attributed"] == 1

    # Negative: empty dict
    d2 = {}
    evals._normalize_dict_classifications(d2)
    assert d2 == {}


def test_normalize_dict_structure():
    # Positive
    d = {"statements": [{"statement": "A"}]}
    evals._normalize_dict_structure(d, "statements", is_nli=False)
    assert d["statements"] == ["A"]

    # Negative: invalid structures
    d2 = {}
    evals._normalize_dict_structure(d2, "claims", is_nli=False)
    assert d2 == {}


def test_log_original_json():
    # Positive
    evals._log_original_json("{}", "claims", False)

    # Negative: handle invalid content (no crash)
    evals._log_original_json("invalid", "claims", False)


def test_normalize_json_response():
    # Positive
    res = evals._normalize_json_response(
        '{"claims": [{"claim": "A"}]}', "claims", False
    )
    assert json.loads(res) == {"claims": ["A"]}
    # Negative: invalid json returns empty string because repair fails
    assert evals._normalize_json_response("invalid", "claims", False) == ""


def test_load_configuration():
    # Positive
    args = argparse.Namespace(
        questions_path="path",
        datasource="enterprise_rag_bench",
        openai_api_key="key",
        openai_endpoint="url",
        openai_model_name="model",
        embeddings_model="embed",
        top_k=3,
        limit_per_category=5,
        limit=10,
        retrieval_only=True,
        generation_only=False,
        compute_model_eval=True,
    )
    config = evals.load_configuration(args)
    assert config["openai_api_key"] == "key"

    # Negative: check fallback to settings for empty namespace fields
    args_empty = argparse.Namespace(
        questions_path=None,
        datasource=None,
        openai_api_key=None,
        openai_endpoint=None,
        openai_model_name=None,
        embeddings_model=None,
        top_k=None,
        limit_per_category=None,
        limit=None,
        retrieval_only=None,
        generation_only=None,
        compute_model_eval=False,
    )
    config_empty = evals.load_configuration(args_empty)
    assert "openai_api_key" in config_empty


def test_resolve_and_sync_config():
    # Positive: dictionary input
    cfg = {
        "questions_path": "path",
        "openai_api_key": "dict_key",
        "openai_endpoint": "url",
        "openai_model_name": "model",
        "embeddings_model": "embed",
        "caipe_datasource_id": "ds",
        "ragas_datasource": "ds2",
        "rag_eval_top_k": 3,
        "rag_eval_retrieval_only": True,
        "rag_eval_generation_only": False,
    }
    synced = evals._resolve_and_sync_config(cfg)
    assert synced["openai_api_key"] == "dict_key"


def test_init_patched_openai_client():
    # Positive
    config = {"openai_api_key": "sk-key", "openai_endpoint": "http://endpoint"}
    client, original = evals._init_patched_openai_client(config)
    assert client.api_key == "sk-key"
    assert original is not None


def test_init_rag_client():
    # Positive
    config = {"compute_model_eval": False, "openai_model_name": "model"}
    client = evals._init_rag_client(config, None, mock.Mock())
    assert client is not None


def test_combine_generations():
    # Positive
    g1 = mock.Mock()
    g2 = mock.Mock()
    results = [mock.Mock(generations=[[g1]]), mock.Mock(generations=[[g2]])]
    res = evals._combine_generations(results)
    assert res == results[0]
    assert res.generations == [[g1, g2]]


def test_patch_text_methods():
    # Positive: patch agenerate_text and generate_text
    ragas_llm = mock.Mock()
    evals._patch_agenerate_text(ragas_llm)
    evals._patch_generate_text(ragas_llm)
    assert hasattr(ragas_llm, "agenerate_text")
    assert hasattr(ragas_llm, "generate_text")


def test_patch_bedrock_anthropic_llm():
    # Positive
    ragas_llm = mock.Mock()
    evals._patch_bedrock_anthropic_llm(ragas_llm, "bedrock/anthropic.claude-v2")

    # Negative: non-bedrock does not modify class
    ragas_llm_normal = mock.Mock()
    evals._patch_bedrock_anthropic_llm(ragas_llm_normal, "gpt-4")


def test_configure_ragas_llm_args():
    # Positive
    ragas_llm = mock.Mock()
    evals._configure_ragas_llm_args(ragas_llm, "gpt-4")
    # Negative: bedrock model
    evals._configure_ragas_llm_args(ragas_llm, "bedrock/claude")


def test_init_metrics():
    # Positive
    config = {"rag_eval_retrieval_only": True, "rag_eval_generation_only": False}
    metrics = evals._init_metrics(config, mock.Mock(), mock.Mock())
    assert len(metrics) == 2


def test_init_evaluator():
    # Positive
    config = {
        "questions_path": "path",
        "openai_api_key": "sk-key",
        "openai_endpoint": "http://endpoint",
        "openai_model_name": "model",
        "embeddings_model": "embed",
        "caipe_datasource_id": "ds",
        "ragas_datasource": "ds2",
        "rag_eval_top_k": 3,
        "rag_eval_retrieval_only": True,
        "rag_eval_generation_only": False,
        "compute_model_eval": False,
    }
    evals.init_evaluator(config)
    assert evals.metrics is not None
    assert evals.rag_client is not None


def test_load_enterprise_rag_bench_negative(tmp_path):
    # Negative: nonexistent path returns empty list instead of raising exception
    assert evals.load_enterprise_rag_bench(tmp_path / "nonexistent.jsonl") == []


def test_load_mock_dataset():
    # Positive
    mock_data = evals.load_mock_dataset()
    assert len(mock_data) > 0


def test_load_samples_from_source(tmp_path):
    # Positive: mock
    samples = evals._load_samples_from_source("mock", "")
    assert len(samples) > 0

    # Positive: hotpotqa
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
        f.write(json.dumps(data[0]) + "\n")
    samples = evals._load_samples_from_source("hotpotqa", str(jsonl_file))
    assert len(samples) == 1
    assert samples[0]["question"] == "Q1"

    # Negative: invalid source type returns empty list
    with pytest.raises(ValueError, match="Unsupported datasource"):
        evals._load_samples_from_source("unknown_source", "")


def test_filter_samples_by_category():
    # Positive: exact match without level
    samples = [{"category": "catA"}, {"category": "catB"}, {"category": "catA"}]
    filtered = evals._filter_samples_by_category(samples, 1)
    assert len(filtered) == 2
    assert [s["category"] for s in filtered] == ["catA", "catB"]

    # Positive: with level (category+level unique combination)
    samples_with_level = [
        {"category": "catA", "level": "easy"},
        {"category": "catA", "level": "easy"},  # should be filtered out
        {"category": "catA", "level": "hard"},  # should be kept (unique combination)
        {"category": "catB", "level": "easy"},
    ]
    filtered_level = evals._filter_samples_by_category(samples_with_level, 1)
    assert len(filtered_level) == 3
    assert filtered_level[0] == {"category": "catA", "level": "easy"}
    assert filtered_level[1] == {"category": "catA", "level": "hard"}
    assert filtered_level[2] == {"category": "catB", "level": "easy"}



def test_load_dataset():
    # Positive: load mock dataset
    dataset = evals.load_dataset(datasource="mock")
    assert dataset is not None

    # Negative: missing datasource
    with pytest.raises(ValueError, match="datasource must be specified"):
        evals.load_dataset(datasource="")


def test_parse_args():
    # Positive
    with mock.patch("sys.argv", ["script", "--openai-model-name", "gpt-4"]):
        args = evals._parse_args()
        assert args.openai_model_name == "gpt-4"


def test_validate_questions_path():
    # Positive: path is empty/not configured does not raise
    evals._validate_questions_path(
        {"ragas_datasource": "mock", "compute_model_eval": False, "questions_path": ""}
    )

    # Negative: path does not exist raises ValueError
    with pytest.raises(ValueError, match="questions_path must be specified"):
        evals._validate_questions_path(
            {
                "ragas_datasource": "enterprise_rag_bench",
                "compute_model_eval": True,
                "questions_path": "",
            }
        )

    with pytest.raises(ValueError, match="questions_path must be specified"):
        evals._validate_questions_path(
            {
                "ragas_datasource": "hotpotqa",
                "compute_model_eval": True,
                "questions_path": "",
            }
        )


def test_prepare_eval_dataset():
    # Positive
    df = pd.DataFrame(
        {
            "question": ["Q"],
            "response": ["A"],
            "retrieved_contexts": [["C"]],
            "reference": ["R"],
        }
    )
    dataset = evals._prepare_eval_dataset(df)
    assert dataset is not None


def test_run_evaluation():
    # Positive
    eval_dataset = mock.Mock()
    df = pd.DataFrame()
    with mock.patch("ragas_eval.evals.evaluate") as mock_evaluate:
        mock_evaluate.return_value = mock.Mock()
        metrics = evals._run_evaluation(eval_dataset, df, [], None, None)
        assert metrics == []


def test_analyze_failures():
    # Positive
    df = pd.DataFrame(
        {
            "factual_correctness": [0.9, 0.4],
            "expected_doc_ids": [["doc1"], ["doc2"]],
            "retrieved_doc_ids": [["doc1"], []],
        }
    )
    avg_recall, avg_precision = evals._analyze_failures(df)
    assert avg_recall == 0.5
    assert avg_precision == 0.5

    # Negative: empty DataFrame returns (0.0, 0.0)
    assert evals._analyze_failures(pd.DataFrame()) == (0.0, 0.0)


def test_log_metrics_summary():
    # Positive
    config = {
        "rag_eval_top_k": 5,
        "rag_eval_retrieval_only": False,
        "rag_eval_generation_only": False,
        "limit_per_category": 10,
        "compute_model_eval": False,
        "rag_eval_short_answer": True,
    }
    df = pd.DataFrame(
        {
            "latency": [1.0],
            "total_tokens": [100],
            "retrieval_recall": [1.0],
            "retrieval_precision": [1.0],
            "failure_cause": ["none"],
        }
    )
    evals._log_metrics_summary(config, df, [], 1.0, 1.0, 10.0)


def test_log_metrics_summary_negative():
    # Negative: Missing required columns should be handled gracefully without raising KeyError
    config = {
        "rag_eval_top_k": 5,
    }
    df = pd.DataFrame(
        {
            "some_other_column": [1.0],
        }
    )
    # Should not raise KeyError
    evals._log_metrics_summary(config, df, [], 1.0, 1.0, 10.0)



def test_save_evaluation_outputs_positive(tmp_path):
    # Positive
    df = pd.DataFrame(
        {
            "question": ["q1"],
            "latency": [1.0],
            "total_tokens": [100],
            "retrieval_recall": [1.0],
            "retrieval_precision": [1.0],
            "failure_cause": ["none"],
        }
    )
    # Mock output directory to save inside tmp_path
    (tmp_path / "evals").mkdir()
    with mock.patch("ragas_eval.evals.Path") as mock_path:
        mock_path.return_value = tmp_path
        mock_path.side_effect = lambda *args: Path(tmp_path, *args)

        asyncio.get_event_loop().run_until_complete(
            evals._save_evaluation_outputs(
                experiment_name="test_exp",
                df=df,
                metric_names=[],
                avg_recall=1.0,
                avg_precision=1.0,
                config_args={"limit": 10},
                evaluation_time=10.0,
                datasource="mock",
            )
        )

        # Verify files are created
        csv_file = tmp_path / "evals" / "experiments" / "test_exp.csv"
        json_file = tmp_path / "evals" / "experiments" / "test_exp_summary.json"
        assert csv_file.exists()
        assert json_file.exists()

        # Load and verify CSV contents
        csv_df = pd.read_csv(csv_file)
        for col in [
            "evaluator_evaluation_time_seconds",
            "evaluator_prompt_tokens",
            "evaluator_completion_tokens",
            "evaluator_total_tokens",
        ]:
            assert col in csv_df.columns

        # Load and verify JSON contents
        with open(json_file, "r") as f:
            data = json.load(f)
        assert data["average_retrieval_recall"] == 1.0
        assert data["average_retrieval_precision"] == 1.0
        assert data["metrics"]["retrieval_recall"] == 1.0
        assert data["metrics"]["retrieval_precision"] == 1.0


def test_save_evaluation_outputs_negative():
    # Negative: empty DataFrame or missing columns should be handled gracefully without raising KeyError
    df = pd.DataFrame()
    # Should not raise KeyError
    asyncio.get_event_loop().run_until_complete(
        evals._save_evaluation_outputs(
            experiment_name="test_exp_fail",
            df=df,
            metric_names=[],
            avg_recall=1.0,
            avg_precision=1.0,
            config_args={},
            evaluation_time=10.0,
            datasource="mock",
        )
    )


def test_save_evaluation_outputs_token_attribution(tmp_path):
    # Positive & Negative: test prompt-matching token attribution to rows
    df = pd.DataFrame(
        {
            "question": ["what is CAIPE?", "how to tune compaction?"],
            "response": ["CAIPE is a platform", "tune compaction by ms"],
            "reference": ["platform info", "compaction knobs"],
            "latency": [1.0, 2.0],
            "total_tokens": [100, 200],
            "retrieval_recall": [1.0, 1.0],
            "retrieval_precision": [1.0, 1.0],
            "failure_cause": ["none", "none"],
        }
    )

    # Set up mock traces with unique content mapping to rows
    evals.ragas_llm_traces = [
        # Match row 0 via question substring
        {
            "messages": [{"role": "user", "content": "Help me answer: what is CAIPE?"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
        },
        # Match row 1 via response substring
        {
            "messages": [{"role": "system", "content": "Check statement: tune compaction by ms against context"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
        },
        # Match row 0 via reference substring
        {
            "messages": [{"role": "user", "content": "Analyze references: platform info"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        },
        # Unmatched trace (should be ignored)
        {
            "messages": [{"role": "user", "content": "some unrelated question"}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 250, "total_tokens": 750},
        },
        # Invalid usage trace (should be ignored safely)
        {
            "messages": [{"role": "user", "content": "what is CAIPE?"}],
            "usage": "<Mock: MagicMock>",
        }
    ]

    (tmp_path / "evals").mkdir()
    with mock.patch("ragas_eval.evals.Path") as mock_path:
        mock_path.return_value = tmp_path
        mock_path.side_effect = lambda *args: Path(tmp_path, *args)

        asyncio.get_event_loop().run_until_complete(
            evals._save_evaluation_outputs(
                experiment_name="test_attr_exp",
                df=df,
                metric_names=[],
                avg_recall=1.0,
                avg_precision=1.0,
                config_args={},
                evaluation_time=5.0,
                datasource="mock",
            )
        )

        csv_file = tmp_path / "evals" / "experiments" / "test_attr_exp.csv"
        assert csv_file.exists()

        csv_df = pd.read_csv(csv_file)
        
        # Row 0 matches trace 1 (15p, 5c) and trace 3 (50p, 25c) -> total 65p, 30c
        assert csv_df.loc[0, "evaluator_prompt_tokens"] == 65
        assert csv_df.loc[0, "evaluator_completion_tokens"] == 30
        assert csv_df.loc[0, "evaluator_total_tokens"] == 95

        # Row 1 matches trace 2 (30p, 10c) -> total 30p, 10c
        assert csv_df.loc[1, "evaluator_prompt_tokens"] == 30
        assert csv_df.loc[1, "evaluator_completion_tokens"] == 10
        assert csv_df.loc[1, "evaluator_total_tokens"] == 40


class TestNormalize:
    def test_lowercase(self):
        assert _normalize("YES") == "yes"

    def test_strips_articles(self):
        assert _normalize("the Chief of Protocol") == "chief of protocol"
        assert _normalize("a cat and an owl") == "cat and owl"

    def test_strips_punctuation(self):
        assert _normalize("yes!") == "yes"
        assert _normalize("Chief of Protocol.") == "chief of protocol"

    def test_collapses_whitespace(self):
        assert _normalize("  yes   ") == "yes"

    def test_empty_string(self):
        assert _normalize("") == ""


class TestContainsAnswer:
    metric = ContainsAnswer()

    def _score(self, response: str, reference: str) -> float:
        sample = SingleTurnSample(
            user_input="dummy question",
            response=response,
            reference=reference,
        )
        return asyncio.get_event_loop().run_until_complete(
            self.metric._single_turn_ascore(sample)
        )

    def test_exact_match_yes(self):
        assert self._score("yes", "yes") == 1.0

    def test_verbose_yes_answer(self):
        # Model gives verbose answer, reference is "yes"
        assert (
            self._score(
                "Yes, both Scott Derrickson and Ed Wood were American directors.",
                "yes",
            )
            == 1.0
        )

    def test_contains_chief_of_protocol(self):
        assert (
            self._score(
                "Shirley Temple held the position of Chief of Protocol.",
                "Chief of Protocol",
            )
            == 1.0
        )

    def test_missing_answer(self):
        # Model says it cannot answer
        assert (
            self._score(
                "Based on the documents, I cannot determine the government position.",
                "Chief of Protocol",
            )
            == 0.0
        )

    def test_empty_reference_returns_zero(self):
        assert self._score("some response", "") == 0.0

    def test_empty_response_returns_zero(self):
        assert self._score("", "yes") == 0.0

    def test_case_insensitive(self):
        assert self._score("The answer is AMERICAN", "american") == 1.0

    def test_article_stripping(self):
        # "the Chief of Protocol" normalises to "chief of protocol"
        assert (
            self._score(
                "She served as the Chief of Protocol in Washington.",
                "Chief of Protocol",
            )
            == 1.0
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Metric validation and fallback tests
# ---------------------------------------------------------------------------
def test_contains_answer_validation():
    """Test that ContainsAnswer metric passes Ragas column validation checks (Positive)."""
    from ragas.validation import validate_required_columns
    from ragas.dataset_schema import EvaluationDataset
    from ragas_eval.metrics import ContainsAnswer

    df = pd.DataFrame(
        [
            {
                "user_input": "q",
                "response": "a",
                "reference": "r",
            }
        ]
    )
    dataset = EvaluationDataset.from_pandas(df)  # type: ignore
    metric = ContainsAnswer()
    validate_required_columns(dataset, [metric])


def test_contains_answer_validation_negative():
    """Test that ContainsAnswer metric fails validation if required columns are missing (Negative)."""
    from ragas.validation import validate_required_columns
    from ragas.dataset_schema import EvaluationDataset
    from ragas_eval.metrics import ContainsAnswer

    # Missing "response" column
    df = pd.DataFrame(
        [
            {
                "user_input": "q",
                "reference": "r",
            }
        ]
    )
    dataset = EvaluationDataset.from_pandas(df)  # type: ignore
    metric = ContainsAnswer()
    with pytest.raises(ValueError):
        validate_required_columns(dataset, [metric])


def test_legacy_embeddings_wrapper_methods():
    """Test that LegacyEmbeddingsWrapper forwards methods correctly to underlying embedding (Positive).

    Ragas 0.3.5 embedding clients implement the modern BaseRagasEmbedding interface (using embed_text/embed_texts),
    whereas legacy metrics (and underlying LangChain elements) expect standard LangChain Embeddings methods
    (using embed_query/embed_documents). The wrapper must expose both to prevent AttributeError crashes.
    """
    from ragas_eval.evals import LegacyEmbeddingsWrapper

    class MockEmbedding:
        def embed_text(self, text: str) -> list[float]:
            return [1.0, 2.0]

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0] for _ in texts]

    mock_emb = MockEmbedding()
    wrapper = LegacyEmbeddingsWrapper(mock_emb)

    assert wrapper.embed_query("test") == [1.0, 2.0]
    assert wrapper.embed_documents(["test"]) == [[1.0, 2.0]]
    assert wrapper.embed_text("test") == [1.0, 2.0]


def test_legacy_embeddings_wrapper_methods_negative():
    """Test that LegacyEmbeddingsWrapper methods handle empty/invalid values gracefully or throw (Negative)."""
    from ragas_eval.evals import LegacyEmbeddingsWrapper

    class MockEmbedding:
        def embed_text(self, text: str) -> list[float]:
            if not text:
                raise ValueError("empty text")
            return [1.0, 2.0]

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            if not texts:
                raise ValueError("empty list")
            return [[1.0, 2.0] for _ in texts]

    mock_emb = MockEmbedding()
    wrapper = LegacyEmbeddingsWrapper(mock_emb)

    with pytest.raises(ValueError, match="empty text"):
        wrapper.embed_query("")

    with pytest.raises(ValueError, match="empty list"):
        wrapper.embed_documents([])


@pytest.mark.anyio
async def test_legacy_embeddings_wrapper_aembed_standard():
    """Test that aembed_text succeeds immediately if the client is asynchronous (Positive)."""
    from ragas_eval.evals import LegacyEmbeddingsWrapper

    class AsyncMockEmbedding:
        async def aembed_text(self, text: str) -> list[float]:
            return [3.0, 4.0]

    mock_emb = AsyncMockEmbedding()
    wrapper = LegacyEmbeddingsWrapper(mock_emb)
    res = await wrapper.aembed_text("test")
    assert res == [3.0, 4.0]


@pytest.mark.anyio
async def test_legacy_embeddings_wrapper_aembed_fallback():
    """Test that aembed_text falls back to sync embed_text on TypeError (Negative/Fallback).

    Some embedding models/clients are synchronous and do not support async aembed_text. When evaluated,
    Ragas attempts to call `aembed_text` on the runner thread which raises TypeError or AttributeError.
    The wrapper must catch these errors and fall back to synchronous `embed_text` to prevent evaluation crashes.
    """
    from ragas_eval.evals import LegacyEmbeddingsWrapper

    class SyncMockEmbedding:
        def __init__(self):
            self.embed_called = False

        def embed_text(self, text: str) -> list[float]:
            self.embed_called = True
            return [1.0, 2.0]

        async def aembed_text(self, text: str) -> list[float]:
            raise TypeError("Cannot use aembed_text with a synchronous client.")

    mock_emb = SyncMockEmbedding()
    wrapper = LegacyEmbeddingsWrapper(mock_emb)

    res = await wrapper.aembed_text("test")
    assert res == [1.0, 2.0]
    assert mock_emb.embed_called is True


def test_load_hotpotqa(tmp_path):
    """Test load_hotpotqa parsing logic (Positive)."""
    from ragas_eval.evals import load_hotpotqa

    jsonl_file = tmp_path / "questions.jsonl"
    data = [
        {
            "user_input": "Q1",
            "reference": "Ref1",
            "expected_doc_ids": ["doc1"],
            "category": "cat1",
            "level": "hard",
        },
        {
            "user_input": "Q2",
            "reference": "Ref2",
            "expected_doc_ids": ["doc2"],
            "category": "cat2",
        }
    ]
    with open(jsonl_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    samples = load_hotpotqa(jsonl_file)
    assert len(samples) == 2
    assert samples[0]["question"] == "Q1"
    assert samples[0]["level"] == "hard"
    assert samples[1]["question"] == "Q2"
    assert samples[1]["level"] == "easy"



def test_load_hotpotqa_negative():
    """Test load_hotpotqa with a non-existent path returns empty list (Negative)."""
    from pathlib import Path
    from ragas_eval.evals import load_hotpotqa

    samples = load_hotpotqa(Path("/non/existent/path.jsonl"))
    assert len(samples) == 0


def test_init_metrics_short_answer_positive():
    """Test _init_metrics returns the short-answer stack when config flag is active (Positive)."""
    from ragas_eval.evals import _init_metrics

    config = {
        "rag_eval_retrieval_only": False,
        "rag_eval_generation_only": False,
        "rag_eval_short_answer": True,
    }
    metrics_list = _init_metrics(config, mock.Mock(), mock.Mock())
    names = [m.name for m in metrics_list]
    assert "semantic_similarity" in names
    assert "contains_answer" in names
    assert "factual_correctness" not in names


def test_init_metrics_short_answer_negative():
    """Test _init_metrics returns the standard stack when config flag is false (Negative)."""
    from ragas_eval.evals import _init_metrics

    config = {
        "rag_eval_retrieval_only": False,
        "rag_eval_generation_only": False,
        "rag_eval_short_answer": False,
    }
    metrics_list = _init_metrics(config, mock.Mock(), mock.Mock())
    names = [m.name for m in metrics_list]
    assert "factual_correctness" in names
    assert "contains_answer" not in names


def test_config_short_answer_resolution_regression():
    """Regression test: Ensure rag_eval_short_answer resolves correctly from CLI args."""
    from ragas_eval.evals import load_configuration
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--short-answer", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--datasource", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--openai-endpoint", type=str, default=None)
    parser.add_argument("--openai-model-name", type=str, default=None)
    parser.add_argument("--embeddings-model", type=str, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--questions-path", type=str, default=None)

    # CLI flag passed
    args_true = parser.parse_args(["--short-answer"])
    config_true = load_configuration(args_true)
    assert config_true["rag_eval_short_answer"] is True

    # CLI flag absent (uses default/env setting)
    args_false = parser.parse_args([])
    config_false = load_configuration(args_false)
    # Checks that it reads settings.rag_eval_short_answer
    from ragas_eval.config import settings

    assert config_false["rag_eval_short_answer"] == settings.rag_eval_short_answer


def test_config_short_answer_sync_regression():
    """Regression test: Ensure rag_eval_short_answer synchronizes to os.environ."""
    from ragas_eval.evals import _resolve_and_sync_config
    import os

    config = {
        "questions_path": "data/q.jsonl",
        "openai_api_key": "k",
        "openai_endpoint": "http://localhost:4000/v1",
        "openai_model_name": "m",
        "embeddings_model": "emb",
        "caipe_datasource_id": "ds",
        "ragas_datasource": "hotpotqa",
        "rag_eval_top_k": 3,
        "rag_eval_retrieval_only": False,
        "rag_eval_generation_only": False,
        "rag_eval_short_answer": True,
    }

    _resolve_and_sync_config(config)
    assert os.environ.get("RAG_EVAL_SHORT_ANSWER") == "true"

    config["rag_eval_short_answer"] = False
    _resolve_and_sync_config(config)
    assert os.environ.get("RAG_EVAL_SHORT_ANSWER") == "false"


def test_load_dataset_and_rename_file():
    """Test that load_dataset creates test_dataset.csv and it gets renamed correctly."""
    old_file = Path("evals") / "datasets" / "test_dataset.csv"
    new_file = Path("evals") / "datasets" / "test_experiment_name.csv"
    
    # Ensure starting clean
    for f in [old_file, new_file]:
        if f.exists():
            f.unlink()

    samples = [{"question": "q1", "category": "c1"}]
    with mock.patch("ragas_eval.evals._load_samples_from_source", return_value=samples):
        try:
            # 1. Load dataset (saves as test_dataset.csv)
            evals.load_dataset(
                limit=1,
                datasource="hotpotqa",
                limit_per_category=None,
                questions_path="dummy.jsonl"
            )
            assert old_file.exists()
            
            # 2. Simulate renaming logic in main()
            experiment_name = "test_experiment_name"
            if old_file.exists():
                renamed_path = Path("evals") / "datasets" / f"{experiment_name}.csv"
                old_file.rename(renamed_path)
                
            assert not old_file.exists()
            assert new_file.exists()
            
        finally:
            # Clean up files
            for f in [old_file, new_file]:
                if f.exists():
                    f.unlink()

def test_config_resolution_and_priority():
    """Verify that environment variables are loaded, CLI args take priority, and output printing is correct."""
    from ragas_eval.evals import load_configuration, _log_metrics_summary
    
    # 1. Test Env Var resolution
    with mock.patch.dict(os.environ, {"RAG_EVAL_SHORT_ANSWER": "true", "QUESTIONS_PATH": "env_q.jsonl"}):
        from ragas_eval.config import Settings
        # Re-initialize Settings to reload from env
        mock_settings = Settings()
        with mock.patch("ragas_eval.evals.settings", mock_settings):
            args = argparse.Namespace(
                questions_path=None,
                datasource="hotpotqa",
                top_k=None,
                openai_api_key=None,
                openai_endpoint=None,
                openai_model_name=None,
                embeddings_model=None,
                retrieval_only=False,
                generation_only=False,
                limit_per_category=None,
                limit=None,
                short_answer=False,
            )
            config = load_configuration(args)
            assert config["rag_eval_short_answer"] is True
            assert config["questions_path"] == "env_q.jsonl"
            
    # 2. Test CLI Override Priority
    with mock.patch.dict(os.environ, {"RAG_EVAL_SHORT_ANSWER": "false"}):
        mock_settings = Settings()
        with mock.patch("ragas_eval.evals.settings", mock_settings):
            args = argparse.Namespace(
                questions_path="cli_q.jsonl",
                datasource="hotpotqa",
                top_k=5,
                openai_api_key=None,
                openai_endpoint=None,
                openai_model_name=None,
                embeddings_model=None,
                retrieval_only=False,
                generation_only=False,
                limit_per_category=None,
                limit=None,
                short_answer=True,
            )
            config = load_configuration(args)
            assert config["rag_eval_short_answer"] is True
            assert config["questions_path"] == "cli_q.jsonl"

    # 3. Test Printing
    df = pd.DataFrame({
        "latency": [1.0],
        "total_tokens": [100],
        "retrieval_recall": [1.0],
        "retrieval_precision": [1.0],
        "failure_cause": ["none"],
    })
    with mock.patch("ragas_eval.evals.logger") as mock_logger:
        _log_metrics_summary(config, df, [], 1.0, 1.0, 10.0)
        # Verify that logger.info was called with config values
        mock_logger.info.assert_any_call("short_answer: True")
        mock_logger.info.assert_any_call("top_k: 5")


def test_experiment_name_prefix_with_datasource():
    """Verify that experiment name is correctly prefixed when datasource is non-empty."""
    config = {"ragas_datasource": "hotpotqa"}
    mock_results = mock.Mock()
    mock_results.name = "rag_run_12345"

    datasource_name = config.get("ragas_datasource")
    if isinstance(datasource_name, str) and datasource_name.strip():
        experiment_name = f"{datasource_name.strip()}_{mock_results.name}"
    else:
        experiment_name = mock_results.name

    assert experiment_name == "hotpotqa_rag_run_12345"


def test_experiment_name_prefix_without_datasource():
    """Verify that experiment name is not prefixed when datasource is empty, None, or a non-string."""
    for empty_val in [None, "", "   ", 123]:
        config = {"ragas_datasource": empty_val}
        mock_results = mock.Mock()
        mock_results.name = "rag_run_12345"

        datasource_name = config.get("ragas_datasource")
        if isinstance(datasource_name, str) and datasource_name.strip():
            experiment_name = f"{datasource_name.strip()}_{mock_results.name}"
        else:
            experiment_name = mock_results.name

        assert experiment_name == "rag_run_12345"


def test_run_evaluation_reason_extraction():
    """Test that _run_evaluation extracts the correct reasoning and appends it to df."""
    from ragas.metrics import Faithfulness, FactualCorrectness
    
    mock_results = mock.Mock()
    mock_df = pd.DataFrame({
        "faithfulness": [0.5],
        "factual_correctness": [1.0]
    })
    mock_results.to_pandas = mock.Mock(return_value=mock_df)
    
    mock_results.traces = [
        {
            "faithfulness": {
                "n_l_i_statement_prompt": {
                    "output": {
                        "statements": [
                            {"statement": "stmt 1", "reason": "reason 1", "verdict": 1},
                            {"statement": "stmt 2", "reason": "reason 2", "verdict": 0}
                        ]
                    }
                }
            },
            "factual_correctness": {
                "n_l_i_statement_prompt": {
                    "output": {
                        "statements": [
                            {"statement": "stmt 3", "reason": "reason 3", "verdict": 1}
                        ]
                    }
                }
            }
        }
    ]
    
    df = pd.DataFrame({"question": ["q1"]})
    metrics = [Faithfulness(), FactualCorrectness()]
    
    with mock.patch("ragas_eval.evals.evaluate", return_value=mock_results):
        metric_names = evals._run_evaluation(
            eval_dataset=mock.Mock(),
            df=df,
            metrics_list=metrics,
            legacy_embeddings=mock.Mock(),
            ragas_llm_obj=mock.Mock()
        )
        
    assert "faithfulness" in metric_names
    assert "factual_correctness" in metric_names
    assert "faithfulness_reason" in df.columns
    assert "factual_correctness_reason" in df.columns
    assert df.loc[0, "faithfulness_reason"] == "[1] stmt 1 -> reason 1; [0] stmt 2 -> reason 2"
    assert df.loc[0, "factual_correctness_reason"] == "[1] stmt 3 -> reason 3"


def test_extract_statements_positive_dict():
    # Test dictionary with 'statements' directly nested inside the prompt output
    data = {
        "n_l_i_statement_prompt": {
            "output": {"statements": [{"statement": "s1", "verdict": 1, "reason": "r1"}]}
        }
    }
    res = evals._extract_statements(data)
    assert len(res) == 1
    assert res[0]["statement"] == "s1"


def test_extract_statements_positive_dict_legacy():
    # Test dictionary with legacy 'n_l_i_statement_prompt' nested structure
    data = {
        "n_l_i_statement_prompt": {
            "output": {
                "statements": [{"statement": "s2", "verdict": 0, "reason": "r2"}]
            }
        }
    }
    res = evals._extract_statements(data)
    assert len(res) == 1
    assert res[0]["statement"] == "s2"


def test_extract_statements_positive_pydantic_model():
    # Mock a Pydantic model/object output nested under n_l_i_statement_prompt
    class MockOutput:
        statements = [{"statement": "s3", "verdict": 1, "reason": "r3"}]

    data = {
        "n_l_i_statement_prompt": {
            "output": MockOutput()
        }
    }
    res = evals._extract_statements(data)
    assert len(res) == 1
    assert res[0]["statement"] == "s3"


def test_extract_statements_negative_none_and_empty():
    # Test None
    assert evals._extract_statements(None) == []
    # Test empty dict
    assert evals._extract_statements({}) == []
    # Test random structure
    assert evals._extract_statements({"random_key": "val"}) == []





