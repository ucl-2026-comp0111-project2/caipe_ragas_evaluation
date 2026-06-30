import json
import unittest.mock as mock

from ragas_eval import evals


def test_patched_ragas_evaluator_claims_list():
    """Test that patched evaluator correctly maps a list of claims in JSON format."""
    mock_choice = mock.Mock()
    mock_choice.message.content = "  [ {'claim': 'Ragas supports metrics',}, {'claim': 'Ragas is built on Python'} ]  "
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "Return the list of 'claims' extracted from text."}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    repaired_content = repaired_resp.choices[0].message.content
    
    parsed = json.loads(repaired_content)
    assert "claims" in parsed
    assert len(parsed["claims"]) == 2
    assert parsed["claims"][0] == "Ragas supports metrics"


def test_patched_ragas_evaluator_claims_list_negative():
    """Test that patched evaluator returns raw invalid non-json content unmodified."""
    mock_choice = mock.Mock()
    mock_choice.message.content = "completely invalid non-json text"
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "Return the list of 'claims' extracted."}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    repaired_content = repaired_resp.choices[0].message.content
    assert repaired_content == "completely invalid non-json text"


def test_patched_ragas_evaluator_markdown_json_unwrapping():
    """Test that patched evaluator unwraps markdown code fence syntax around JSON blocks."""
    # Test unwrapping markdown markdown fences ```json
    mock_choice = mock.Mock()
    mock_choice.message.content = "```json\n{\n  \"claims\": [\"claim 1\"]\n}\n```"
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "'claims'"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert parsed["claims"][0] == "claim 1"


def test_patched_ragas_evaluator_statements_is_nli_list():
    """Test that patched evaluator normalizes lists of statements for NLI evaluations."""
    mock_choice = mock.Mock()
    # List instead of dict returned
    mock_choice.message.content = '[ {"statement": "S1"}, "S2" ]'
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "'statements' and 'verdict'"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert "statements" in parsed
    assert parsed["statements"][0]["statement"] == "S1"
    assert parsed["statements"][1]["statement"] == "S2"
    assert parsed["statements"][1]["verdict"] == 1


def test_patched_ragas_evaluator_statements_non_nli_list():
    """Test that patched evaluator normalizes list of statements for non-NLI metrics."""
    mock_choice = mock.Mock()
    mock_choice.message.content = '[ {"statement": "S1"}, "S2" ]'
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "'statements'"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert parsed["statements"] == ["S1", "S2"]


def test_patched_ragas_evaluator_generic_statements_list_fallback():
    """Test that patched evaluator falls back to dict statements structure for generic list inputs."""
    mock_choice = mock.Mock()
    mock_choice.message.content = '[ {"statement": "S1"} ]'
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "extract statements fallback"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert parsed["statements"][0]["statement"] == "S1"


def test_patched_ragas_evaluator_generic_statements_list_string_fallback():
    """Test that patched evaluator correctly wraps lists of strings into structured statement dicts."""
    # Test list of strings fallback: executes elif parsed and isinstance(parsed[0], str)
    mock_choice = mock.Mock()
    mock_choice.message.content = '[ "S1", "S2" ]'
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "extract statements fallback"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert parsed["statements"][0]["statement"] == "S1"


def test_patched_ragas_evaluator_pop_and_rename():
    """Test that patched evaluator renames top-level keys to match expected schema target names."""
    mock_choice = mock.Mock()
    mock_choice.message.content = '{ "statements": [ {"statement": "S1"} ] }'
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "'classifications'"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert "classifications" in parsed
    assert parsed["classifications"][0]["statement"] == "S1"


def test_patched_ragas_evaluator_statements_normalize_nli():
    """Test that patched evaluator successfully normalizes mixed statement types in NLI dicts."""
    mock_choice = mock.Mock()
    mock_choice.message.content = '{ "statements": [ {"statement": "S1"}, "S2" ] }'
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "'statements' and 'verdict'"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert parsed["statements"][1]["statement"] == "S2"


def test_patched_ragas_evaluator_statements_normalize_non_nli():
    """Test that patched evaluator successfully normalizes mixed statement types in non-NLI dicts."""
    mock_choice = mock.Mock()
    mock_choice.message.content = '{ "statements": [ {"statement": "S1"}, "S2" ] }'
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "'statements'"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert parsed["statements"] == ["S1", "S2"]


def test_patched_ragas_evaluator_classifications_normalize_string_item():
    """Test that patched evaluator normalizes classification list string items into dictionaries."""
    mock_choice = mock.Mock()
    mock_choice.message.content = '{ "classifications": [ "S1" ] }'
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "'classifications'"}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert parsed["classifications"][0]["statement"] == "S1"


def test_patched_ragas_evaluator_nli_verdicts_negative():
    """Test that patched evaluator behaves correctly when JSON repair fails to fix corrupt content."""
    mock_choice = mock.Mock()
    mock_choice.message.content = "{ invalid json statements: }"
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "Produce a JSON containing 'statements'."}
        ]
    }
    
    def mock_repair_json(content, return_objects=False):
        """Mock JSON repair function that returns predefined broken json."""
        if return_objects:
            return None
        return "{ invalid json }"
        
    with mock.patch("json_repair.repair_json", side_effect=mock_repair_json):
        repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
        assert repaired_resp.choices[0].message.content == "{ invalid json }"


def test_patched_ragas_evaluator_classifications():
    """Test that patched evaluator sets default values for classification fields when they are empty."""
    mock_choice = mock.Mock()
    mock_choice.message.content = ' { "classifications": [ {"reason": "", "verdict": null, "attributed": null} ] } '
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "Extract 'classifications' from the text."}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    parsed = json.loads(repaired_resp.choices[0].message.content)
    assert "classifications" in parsed
    assert parsed["classifications"][0]["reason"] == "Reason not provided"
    assert parsed["classifications"][0]["verdict"] == 0
    assert parsed["classifications"][0]["attributed"] == 0


def test_patched_ragas_evaluator_classifications_negative():
    """Test that patched evaluator handles invalid classification text block without crash."""
    mock_choice = mock.Mock()
    mock_choice.message.content = "classifications: error payload"
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "messages": [
            {"role": "system", "content": "Extract 'classifications' from the text."}
        ]
    }
    
    repaired_resp = evals.patched_ragas_evaluator_llm_create(**kwargs)
    assert repaired_resp.choices[0].message.content == "classifications: error payload"


def test_patched_ragas_evaluator_bedrock_params_and_usage():
    """Test that patched evaluator filters unsupported parameters and sums evaluator token usage metrics."""
    mock_choice = mock.Mock()
    mock_choice.message.content = "{}"
    mock_response = mock.Mock()
    mock_response.choices = [mock_choice]
    
    # Test dict usage
    mock_response.usage = {
        "prompt_tokens": 50,
        "completion_tokens": 25
    }
    
    evals.ragas_prompt_tokens = 0
    evals.ragas_completion_tokens = 0
    
    mock_original_create = mock.Mock(return_value=mock_response)
    evals.original_create = mock_original_create
    
    kwargs = {
        "temperature": 0.5,
        "top_p": 0.9,
        "messages": [
            {"role": "system", "content": "General instructions."}
        ]
    }
    
    evals.patched_ragas_evaluator_llm_create(**kwargs)
    assert evals.ragas_prompt_tokens == 50
    
    # Test object usage
    mock_usage = mock.Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_response.usage = mock_usage
    
    evals.patched_ragas_evaluator_llm_create(**kwargs)
    assert evals.ragas_prompt_tokens == 60
