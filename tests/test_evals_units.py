import argparse
import pytest
import json
import pandas as pd
import unittest.mock as mock
from ragas_eval import evals


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


def test_load_samples_from_source():
    # Positive: mock
    samples = evals._load_samples_from_source("mock", "")
    assert len(samples) > 0

    # Negative: invalid source type returns empty list
    with pytest.raises(ValueError, match="Unsupported datasource"):
        evals._load_samples_from_source("unknown_source", "")


def test_filter_samples_by_category():
    # Positive: exact match
    samples = [{"category": "catA"}, {"category": "catB"}]
    filtered = evals._filter_samples_by_category(samples, 1)
    assert len(filtered) == 2


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
    avg_recall = evals._analyze_failures(df)
    assert avg_recall == 0.5

    # Negative: empty DataFrame returns 0.0
    assert evals._analyze_failures(pd.DataFrame()) == 0.0


def test_log_metrics_summary():
    # Positive
    args = argparse.Namespace(limit=10)
    df = pd.DataFrame(
        {
            "latency": [1.0],
            "total_tokens": [100],
            "retrieval_recall": [1.0],
            "failure_cause": ["none"],
        }
    )
    evals._log_metrics_summary(args, df, [], 1.0)
