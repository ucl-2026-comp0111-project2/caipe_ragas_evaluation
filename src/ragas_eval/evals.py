import os
import sys
from datetime import datetime

# Disable Ragas anonymous telemetry and background thread
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, cast, Optional, Tuple
import argparse


from openai import OpenAI
import openai.resources.chat.completions
import pandas as pd

logger = logging.getLogger(__name__)

from ragas import Dataset, experiment, evaluate, EvaluationDataset  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.embeddings import BaseRagasEmbedding  # noqa: E402
from ragas.embeddings.base import embedding_factory  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.metrics import (  # noqa: E402
    FactualCorrectness,
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    SemanticSimilarity,
)
from ragas_eval.metrics import ContainsAnswer  # noqa: E402

# Add the current directory to the path so we can import rag module when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from ragas_eval.rag import default_rag_client  # noqa: E402
from ragas_eval.agentic_rag import default_agentic_rag_client  # noqa: E402

# Global evaluation components (initialized via init_evaluator)
openai_client: Any = None
original_create: Any = None
original_async_create: Any = None
rag_client: Any = None
ragas_llm: Any = None
ragas_embeddings: Any = None
metrics = []

# Global token counters for Ragas evaluator LLM calls
ragas_prompt_tokens = 0
ragas_completion_tokens = 0

# Global list to collect Ragas evaluator LLM call traces
ragas_llm_traces = []

GENERATED_STATEMENT_REASON = "Generated statement"

# Import consolidated configuration settings
from ragas_eval.config import settings  # noqa: E402

# Resolved configuration state for callbacks
rag_eval_retrieval_only: Optional[bool] = None
rag_eval_top_k: Optional[int] = None


def _sanitize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Avoid Bedrock and LiteLLM unsupported parameters or parameter conflict errors."""
    if "temperature" in kwargs and "top_p" in kwargs:
        kwargs.pop("top_p", None)

    if "n" in kwargs:
        kwargs.pop("n", None)

    extra_headers = kwargs.get("extra_headers") or {}
    extra_headers["drop_params"] = "true"
    kwargs["extra_headers"] = extra_headers
    return kwargs


def _detect_expected_schema(
    messages: List[Dict[str, Any]],
) -> tuple[Optional[str], bool]:
    """Detects expected schema and NLI status from system prompt."""
    expected_top_level = None
    is_nli = False
    system_content = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
            break

    if '"claims"' in system_content or "'claims'" in system_content:
        expected_top_level = "claims"
    elif '"statements"' in system_content or "'statements'" in system_content:
        expected_top_level = "statements"
        if (
            "StatementFaithfulnessAnswer" in system_content
            or "verdict" in system_content
        ):
            is_nli = True
    elif '"classifications"' in system_content or "'classifications'" in system_content:
        expected_top_level = "classifications"

    return expected_top_level, is_nli


def _track_token_usage(response: Any) -> None:
    """Track token usage from the evaluator model response."""
    global ragas_prompt_tokens, ragas_completion_tokens
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        if isinstance(usage, dict):
            ragas_prompt_tokens += usage.get("prompt_tokens", 0)
            ragas_completion_tokens += usage.get("completion_tokens", 0)
        else:
            ragas_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            ragas_completion_tokens += getattr(usage, "completion_tokens", 0) or 0


def _parse_json_content(content: str) -> tuple[Optional[Any], Optional[str]]:
    """Tries to repair and parse the JSON content, unescaping structural quotes as needed."""
    import json_repair
    import json

    # Try parsing the original content first to preserve valid escaped internal quotes
    repaired_content = json_repair.repair_json(content)
    try:
        parsed = json.loads(repaired_content)
        if parsed is not None:
            return parsed, repaired_content
    except Exception:
        pass

    # Fallback to unescaping structural quotes only if the original parsing fails
    unescaped = content.replace('\\\\"', '"').replace('\\"', '"')
    repaired_content = json_repair.repair_json(unescaped)
    try:
        parsed = json.loads(repaired_content)
        return parsed, repaired_content
    except Exception:
        return None, repaired_content


def _normalize_key_name(k: str) -> str:
    """Normalizes a single dictionary key name by stripping special characters and applying fuzzy matching."""
    k_clean = (
        k.strip()
        .replace('"', "")
        .replace("'", "")
        .replace("\\", "")
        .strip("_")
        .strip(":")
        .strip()
        .lower()
    )
    if "tate" in k_clean:
        if k_clean.endswith("s") or "statements" in k_clean:
            return "statements"
        return "statement"
    if "reas" in k_clean:
        return "reason"
    if "verd" in k_clean:
        return "verdict"
    if "attrib" in k_clean:
        return "attributed"
    return k_clean


def _clean_keys(obj: Any) -> Any:
    """Recursively normalizes dictionary keys by stripping special characters and applying fuzzy names."""
    if isinstance(obj, dict):
        return {_normalize_key_name(k): _clean_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_keys(item) for item in obj]
    return obj


def _sanitize_values(obj: Any) -> Any:
    """Recursively sanitizes values, mapping truthy/yes/supported strings or booleans to 1 and others to 0 for binary metrics."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k in ("verdict", "attributed"):
                if v in (
                    1,
                    "1",
                    True,
                    "true",
                    "yes",
                    "Yes",
                    "supported",
                    "Supported",
                    "support",
                    "Support",
                ):
                    new_dict[k] = 1
                else:
                    new_dict[k] = 0
            else:
                new_dict[k] = _sanitize_values(v)
        return new_dict
    elif isinstance(obj, list):
        return [_sanitize_values(item) for item in obj]
    return obj


def _fill_dict_statement_keys(obj: Dict[str, Any]) -> None:
    """Fills statement keys on a dictionary object."""
    if "statement" not in obj or not obj["statement"]:
        obj["statement"] = "Statement not provided"
    if "reason" not in obj or not obj["reason"]:
        obj["reason"] = "Reason not provided"
    if "verdict" not in obj and "attributed" not in obj:
        obj["verdict"] = 0
        obj["attributed"] = 0


def _fill_dict_standard_keys(obj: Dict[str, Any]) -> None:
    """Fills standard keys on a dictionary object."""
    if "reason" in obj and not obj["reason"]:
        obj["reason"] = "Reason not provided"
    if "verdict" in obj and obj["verdict"] is None:
        obj["verdict"] = 0
    if "attributed" in obj and obj["attributed"] is None:
        obj["attributed"] = 0


def _fill_missing_keys(
    obj: Any, expected_top_level: Optional[str], is_nli: bool
) -> Any:
    """Recursively checks and fills missing expected keys with default values to prevent schema validation failures."""
    if isinstance(obj, dict):
        is_statement_item = "statement" in obj or (
            expected_top_level in ("statements", "classifications") and is_nli
        )
        if is_statement_item:
            _fill_dict_statement_keys(obj)
        else:
            _fill_dict_standard_keys(obj)
        return {
            k: _fill_missing_keys(v, expected_top_level, is_nli) for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_fill_missing_keys(item, expected_top_level, is_nli) for item in obj]
    return obj


def _normalize_claims(parsed: List[Any]) -> Dict[str, List[Any]]:
    """Normalize a parsed list when claims are expected."""
    return {
        "claims": [
            (
                item.get("claim", item.get("statement", item))
                if isinstance(item, dict)
                else item
            )
            for item in parsed
        ]
    }


def _normalize_statements(parsed: List[Any], is_nli: bool) -> Dict[str, List[Any]]:
    """Normalize a parsed list when statements are expected."""
    if is_nli:
        parsed_items = []
        for item in parsed:
            if isinstance(item, dict):
                parsed_items.append(item)
            elif isinstance(item, str):
                parsed_items.append(
                    {
                        "statement": item,
                        "reason": GENERATED_STATEMENT_REASON,
                        "verdict": 1,
                    }
                )
        return {"statements": parsed_items}

    return {
        "statements": [
            (item.get("statement", item) if isinstance(item, dict) else item)
            for item in parsed
        ]
    }


def _is_matching_keys_heuristic(keys: Any) -> bool:
    """Helper check to verify if keys match statement/verdict/reason."""
    return any(
        k.strip().replace('"', "").replace("'", "").replace("\\", "").strip("_").lower()
        in ("statement", "verdict", "reason")
        for k in keys
    )


def _normalize_fallback(parsed: List[Any]) -> Any:
    """Fallback heuristics for list structures."""
    if (
        parsed
        and isinstance(parsed[0], dict)
        and _is_matching_keys_heuristic(parsed[0].keys())
    ):
        return {"statements": parsed}
    if parsed and isinstance(parsed[0], str):
        return {
            "statements": [
                {
                    "statement": s,
                    "reason": GENERATED_STATEMENT_REASON,
                    "verdict": 1,
                }
                for s in parsed
            ]
        }
    return parsed


def _normalize_list_structure(
    parsed: List[Any], expected_top_level: Optional[str], is_nli: bool
) -> Any:
    """Normalize structure if the model returned a list directly instead of dict."""
    if expected_top_level == "claims":
        return _normalize_claims(parsed)
    if expected_top_level == "statements":
        return _normalize_statements(parsed, is_nli)
    return _normalize_fallback(parsed)


def _rename_top_level_key(
    parsed: Dict[str, Any], expected_top_level: Optional[str]
) -> None:
    """Rename any existing statements/claims/classifications list key to the expected key if it's not present."""
    if not expected_top_level:
        return
    current_keys = list(parsed.keys())
    if expected_top_level not in current_keys:
        for k in current_keys:
            if k in (
                "statements",
                "claims",
                "classifications",
            ) and isinstance(parsed[k], list):
                parsed[expected_top_level] = parsed.pop(k)
                break


def _normalize_dict_claims(parsed: Dict[str, Any]) -> None:
    """Normalize claims list inside the dictionary structure."""
    if "claims" not in parsed:
        return
    claims_list = parsed["claims"]
    if isinstance(claims_list, list):
        parsed["claims"] = [
            (
                item.get("claim", item.get("statement", str(item)))
                if isinstance(item, dict)
                else str(item)
            )
            for item in claims_list
        ]


def _normalize_dict_statements(parsed: Dict[str, Any], is_nli: bool) -> None:
    """Normalize statements list inside the dictionary structure."""
    if "statements" not in parsed:
        return
    statements_list = parsed["statements"]
    if not isinstance(statements_list, list):
        return
    if is_nli:
        new_list = []
        for item in statements_list:
            if isinstance(item, dict):
                new_list.append(item)
            elif isinstance(item, str):
                new_list.append(
                    {
                        "statement": item,
                        "reason": GENERATED_STATEMENT_REASON,
                        "verdict": 1,
                    }
                )
        parsed["statements"] = new_list
    else:
        parsed["statements"] = [
            (item.get("statement", str(item)) if isinstance(item, dict) else str(item))
            for item in statements_list
        ]


def _normalize_dict_classifications(parsed: Dict[str, Any]) -> None:
    """Normalize classifications list inside the dictionary structure."""
    if "classifications" not in parsed:
        return
    classifications_list = parsed["classifications"]
    if isinstance(classifications_list, list):
        new_list = []
        for item in classifications_list:
            if isinstance(item, dict):
                new_list.append(item)
            elif isinstance(item, str):
                new_list.append(
                    {
                        "statement": item,
                        "reason": GENERATED_STATEMENT_REASON,
                        "attributed": 1,
                    }
                )
        parsed["classifications"] = new_list


def _normalize_dict_structure(
    parsed: Dict[str, Any], expected_top_level: Optional[str], is_nli: bool
) -> Dict[str, Any]:
    """Normalize dict keys and structure."""
    _rename_top_level_key(parsed, expected_top_level)

    if expected_top_level == "claims":
        _normalize_dict_claims(parsed)
    elif expected_top_level == "statements":
        _normalize_dict_statements(parsed, is_nli)
    elif expected_top_level == "classifications":
        _normalize_dict_classifications(parsed)

    return parsed


def _log_original_json(
    content: str, expected_top_level: Optional[str], is_nli: bool
) -> None:
    """Log original response content if it parses as JSON."""
    import json_repair
    import json

    try:
        clean_content = content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()
        orig_parsed = json.loads(json_repair.repair_json(clean_content))
        pretty_original = json.dumps(orig_parsed, indent=2)
    except Exception:
        pretty_original = content

    logger.debug(
        f"\n[RAGAS_EVALUATOR_LLM_PATCH] Expected Schema: {expected_top_level!r} (is_nli={is_nli})"
    )
    logger.debug(
        f"[RAGAS_EVALUATOR_LLM_PATCH] Original content (JSON):\n{pretty_original}"
    )


def _normalize_json_response(
    content: str, expected_top_level: Optional[str], is_nli: bool
) -> str:
    """Coordinates parsing, normalizing, cleaning, sanitizing, and filling missing keys of a JSON response."""
    import json

    _log_original_json(content, expected_top_level, is_nli)

    parsed, repaired_content = _parse_json_content(content)
    if repaired_content is None:
        return content

    if parsed is not None:
        if isinstance(parsed, list):
            parsed = _normalize_list_structure(parsed, expected_top_level, is_nli)

        if isinstance(parsed, dict):
            parsed = _normalize_dict_structure(parsed, expected_top_level, is_nli)

        try:
            parsed = _clean_keys(parsed)
            parsed = _sanitize_values(parsed)
            parsed = _fill_missing_keys(parsed, expected_top_level, is_nli)
            repaired_content = json.dumps(parsed)
        except Exception:
            logger.exception("[RAGAS_EVALUATOR_LLM_PATCH] Normalisation error")

    logger.debug(
        f"[RAGAS_EVALUATOR_LLM_PATCH] Mapped & Repaired content (JSON):\n{json.dumps(parsed, indent=2)}"
    )
    return repaired_content


def _sanitize_for_json(val: Any) -> Any:
    """Recursively converts Mock/MagicMock and other non-serializable objects to standard types for JSON export."""
    if type(val).__name__ in ("Mock", "MagicMock"):
        return f"<Mock: {type(val).__name__}>"
    if isinstance(val, dict):
        return {str(k): _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_for_json(item) for item in val]
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    try:
        # Check if json serializable
        json.dumps(val)
        return val
    except TypeError:
        return str(val)


def _add_llm_trace(
    messages: List[Dict[str, Any]],
    expected_top_level: Optional[str],
    is_nli: bool,
    original_content: Optional[str] = None,
    repaired_content: Optional[str] = None,
    response: Any = None,
    error: Optional[Exception] = None,
    json_parse_success: Optional[bool] = None,
) -> None:
    """Appends an LLM interaction trace to the global trace list."""
    global ragas_llm_traces
    trace: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "expected_top_level": expected_top_level,
        "is_nli": is_nli,
        "original_content": original_content,
        "repaired_content": repaired_content,
        "json_parse_success": json_parse_success,
    }
    if error is not None:
        trace["error"] = str(error)
    if response is not None:
        usage = getattr(response, "usage", None)
        if usage:
            if isinstance(usage, dict):
                trace["usage"] = usage
            elif hasattr(usage, "model_dump"):
                try:
                    trace["usage"] = usage.model_dump()
                except Exception:
                    trace["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    }
            elif hasattr(usage, "dict"):
                try:
                    trace["usage"] = usage.dict()
                except Exception:
                    trace["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    }
            else:
                trace["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
    ragas_llm_traces.append(_sanitize_for_json(trace))


def patched_ragas_evaluator_llm_create(*args, **kwargs):
    """Intercepts and patches Ragas evaluator LLM calls to handle schema normalisation, token tracking, and trace logging."""
    kwargs = _sanitize_kwargs(kwargs)
    messages = kwargs.get("messages", [])
    expected_top_level, is_nli = _detect_expected_schema(messages)

    response = None
    try:
        func = original_create
        response = func(*args, **kwargs)
        _track_token_usage(response)
    except Exception as e:
        logger.exception("[RAGAS_EVALUATOR_LLM_PATCH] OpenAI completion request failed")
        _add_llm_trace(messages, expected_top_level, is_nli, error=e)
        raise

    original_content = None
    repaired_content = None
    json_parse_success = None
    if hasattr(response, "choices") and response.choices:
        content = response.choices[0].message.content
        if content and isinstance(content, str):
            original_content = content
            s_stripped = content.strip()
            is_json = s_stripped.startswith(("{", "[", "`"))
            if is_json:
                parsed, _ = _parse_json_content(content)
                json_parse_success = (parsed is not None)
                repaired_content = _normalize_json_response(
                    content, expected_top_level, is_nli
                )
                response.choices[0].message.content = repaired_content

    _add_llm_trace(
        messages,
        expected_top_level,
        is_nli,
        original_content=original_content,
        repaired_content=repaired_content,
        response=response,
        json_parse_success=json_parse_success,
    )
    return response


async def patched_ragas_evaluator_llm_async_create(*args, **kwargs):
    """Intercepts and patches Ragas async evaluator LLM calls to handle schema normalisation, token tracking, and trace logging."""
    kwargs = _sanitize_kwargs(kwargs)
    messages = kwargs.get("messages", [])
    expected_top_level, is_nli = _detect_expected_schema(messages)

    response = None
    try:
        func = original_async_create
        response = await func(*args, **kwargs)
        _track_token_usage(response)
    except Exception as e:
        logger.exception("[RAGAS_EVALUATOR_LLM_PATCH] OpenAI async completion request failed")
        _add_llm_trace(messages, expected_top_level, is_nli, error=e)
        raise

    original_content = None
    repaired_content = None
    json_parse_success = None
    if hasattr(response, "choices") and response.choices:
        content = response.choices[0].message.content
        if content and isinstance(content, str):
            original_content = content
            s_stripped = content.strip()
            is_json = s_stripped.startswith(("{", "[", "`"))
            if is_json:
                parsed, _ = _parse_json_content(content)
                json_parse_success = (parsed is not None)
                repaired_content = _normalize_json_response(
                    content, expected_top_level, is_nli
                )
                response.choices[0].message.content = repaired_content

    _add_llm_trace(
        messages,
        expected_top_level,
        is_nli,
        original_content=original_content,
        repaired_content=repaired_content,
        response=response,
        json_parse_success=json_parse_success,
    )
    return response


def load_configuration(args) -> Dict[str, Any]:
    """Consolidates configuration from CLI args, environment, and settings.
    Does NOT mutate the global settings variable."""
    questions_path = (
        args.questions_path
        or os.environ.get("QUESTIONS_PATH")
        or settings.questions_path
    )
    datasource = (
        args.datasource
        or os.environ.get("RAGAS_DATASOURCE")
        or settings.ragas_datasource
    )

    config = {
        "questions_path": questions_path,
        "openai_api_key": args.openai_api_key
        or os.environ.get("OPENAI_API_KEY")
        or settings.openai_api_key,
        "openai_endpoint": args.openai_endpoint
        or os.environ.get("OPENAI_ENDPOINT")
        or settings.openai_endpoint,
        "openai_model_name": args.openai_model_name
        or os.environ.get("OPENAI_MODEL_NAME")
        or settings.openai_model_name,
        "embeddings_model": args.embeddings_model
        or os.environ.get("EMBEDDINGS_MODEL")
        or settings.embeddings_model,
        "caipe_datasource_id": args.datasource
        or os.environ.get("CAIPE_DATASOURCE_ID")
        or settings.caipe_datasource_id,
        "ragas_datasource": datasource,
        "rag_eval_top_k": (
            args.top_k if args.top_k is not None else settings.rag_eval_top_k
        ),
        "limit_per_category": (
            args.limit_per_category
            if args.limit_per_category is not None
            else settings.limit_per_category
        ),
        "ragas_limit": args.limit if args.limit is not None else settings.ragas_limit,
        "rag_eval_retrieval_only": args.retrieval_only
        or settings.rag_eval_retrieval_only,
        "rag_eval_generation_only": args.generation_only
        or settings.rag_eval_generation_only,
        "rag_eval_short_answer": getattr(args, "short_answer", False)
        or settings.rag_eval_short_answer,
        "compute_model_eval": getattr(args, "compute_model_eval", False),
        "agentic": getattr(args, "agentic", False),
        "agent_api_url": getattr(args, "agent_api_url", None)
        or settings.caipe_agent_api_url,
        "agent_api_timeout": getattr(args, "agent_api_timeout", 120.0),
        "trace_log": getattr(args, "enable_trace_log", False),
    }
    return config


def _resolve_and_sync_config(args_or_config: Any) -> Dict[str, Any]:
    """Resolve configuration and sync settings to environment variables."""
    if isinstance(args_or_config, dict):
        config = args_or_config
    else:
        config = load_configuration(args_or_config)

    # Synchronize settings back to os.environ for third-party tools
    os.environ["QUESTIONS_PATH"] = config["questions_path"] or ""
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    os.environ["OPENAI_ENDPOINT"] = config["openai_endpoint"]
    os.environ["OPENAI_MODEL_NAME"] = config["openai_model_name"]
    os.environ["EMBEDDINGS_MODEL"] = config["embeddings_model"]
    os.environ["CAIPE_DATASOURCE_ID"] = config["caipe_datasource_id"]
    os.environ["CAIPE_QUERY_ENDPOINT"] = settings.caipe_query_endpoint
    os.environ["RAGAS_DATASOURCE"] = config["ragas_datasource"]
    os.environ["RAG_EVAL_TOP_K"] = str(config["rag_eval_top_k"])
    os.environ["RAG_EVAL_RETRIEVAL_ONLY"] = str(
        config["rag_eval_retrieval_only"]
    ).lower()
    os.environ["RAG_EVAL_GENERATION_ONLY"] = str(
        config["rag_eval_generation_only"]
    ).lower()
    os.environ["RAG_EVAL_SHORT_ANSWER"] = str(
        config.get("rag_eval_short_answer", False)
    ).lower()
    return config


def _init_patched_openai_client(config: Dict[str, Any]) -> tuple[OpenAI, Any]:
    """Initialize OpenAI client and patch its chat completions method."""
    client = OpenAI(
        api_key=config["openai_api_key"],
        base_url=config["openai_endpoint"],
        timeout=None,
        default_headers={"drop_params": "true"},
    )
    
    global original_create, original_async_create
    
    # Patch globally on the class level so that both sync and async clients created
    # by LangChain / Ragas internally are intercepted and tracked
    if openai.resources.chat.completions.Completions.create != patched_ragas_evaluator_llm_create:
        original_create = openai.resources.chat.completions.Completions.create
        openai.resources.chat.completions.Completions.create = patched_ragas_evaluator_llm_create
        
    if openai.resources.chat.completions.AsyncCompletions.create != patched_ragas_evaluator_llm_async_create:
        original_async_create = openai.resources.chat.completions.AsyncCompletions.create
        openai.resources.chat.completions.AsyncCompletions.create = patched_ragas_evaluator_llm_async_create
        
    return client, original_create


def cleanup_evaluator():
    """Restores original patched client behaviors and resets globals."""
    global openai_client, original_create, original_async_create, rag_client, ragas_llm, ragas_embeddings, metrics
    
    if original_create is not None:
        openai.resources.chat.completions.Completions.create = original_create
        original_create = None
    if original_async_create is not None:
        openai.resources.chat.completions.AsyncCompletions.create = original_async_create
        original_async_create = None
        
    openai_client = None
    rag_client = None
    ragas_llm = None
    ragas_embeddings = None
    metrics = []


def _init_rag_client(
    config: Dict[str, Any], dataset: Optional[Dataset], openai_client: OpenAI
) -> Any:
    """Initialize the RAG client based on compute_model_eval flag."""
    # Agentic mode: route through caipe-supervisor A2A endpoint
    if config.get("agentic"):
        return default_agentic_rag_client(
            logdir="evals/logs",
            agent_api_url=config.get("agent_api_url"),
            timeout=config.get("agent_api_timeout", 120.0),
            insecure=settings.insecure_ssl,
            trace_log=config.get("trace_log", False),
        )

    if not config["compute_model_eval"]:
        return default_rag_client(
            llm_client=openai_client,
            logdir="evals/logs",
            model_name=config["openai_model_name"],
            insecure=settings.insecure_ssl,
        )

    from ragas_eval.precomputed_rag import PrecomputedRAG

    if dataset is not None and hasattr(dataset, "samples") and dataset.samples:
        return PrecomputedRAG(preloaded_samples=dataset.samples)
    if dataset is not None and len(dataset) > 0:
        return PrecomputedRAG(preloaded_samples=list(dataset))

    questions_path = config["questions_path"]
    if not questions_path:
        raise ValueError(
            "questions_path must be specified when compute_model_eval is enabled."
        )
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    return PrecomputedRAG(dataset_path=questions_path)


def _combine_generations(results: List[Any]) -> Any:
    """Combine generations list from multiple single-run calls when n > 1."""
    combined = []
    for res in results:
        combined.extend(res.generations[0])
    results[0].generations = [combined]
    return results[0]


def _patch_agenerate_text(ragas_llm: Any) -> None:
    """Patch agenerate_text to avoid issues with n > 1."""
    original_agenerate = ragas_llm.agenerate_text

    async def patched_agenerate_text(prompt, n=1, **kwargs):
        if n > 1:
            results = []
            for _ in range(n):
                res = await original_agenerate(prompt, n=1, **kwargs)
                results.append(res)
            return _combine_generations(results)
        return await original_agenerate(prompt, n=1, **kwargs)

    ragas_llm.agenerate_text = patched_agenerate_text


def _patch_generate_text(ragas_llm: Any) -> None:
    """Patch generate_text to avoid issues with n > 1."""
    original_generate = ragas_llm.generate_text

    def patched_generate_text(prompt, n=1, **kwargs):
        if n > 1:
            results = []
            for _ in range(n):
                res = original_generate(prompt, n=1, **kwargs)
                results.append(res)
            return _combine_generations(results)
        return original_generate(prompt, n=1, **kwargs)

    ragas_llm.generate_text = patched_generate_text


def _patch_bedrock_anthropic_llm(ragas_llm: Any, model_name: str) -> None:
    """Wrap agenerate_text and generate_text for Bedrock and Anthropic models to avoid issues with n > 1."""
    if not ("bedrock" in model_name.lower() or "anthropic" in model_name.lower()):
        return

    if hasattr(ragas_llm, "agenerate_text"):
        _patch_agenerate_text(ragas_llm)

    if hasattr(ragas_llm, "generate_text"):
        _patch_generate_text(ragas_llm)


def _configure_ragas_llm_args(ragas_llm: Any, model_name: str) -> None:
    """Configure model arguments like max_tokens and temperature on ragas_llm."""
    model_args: Dict[str, Any] = {
        "max_tokens": 8192,
        "temperature": 0.2,
    }
    if "qwen" in model_name or "ollama" in model_name:
        model_args["extra_body"] = {"options": {"num_ctx": 32768, "num_predict": 8192}}

    if hasattr(ragas_llm, "model_args") and hasattr(ragas_llm.model_args, "update"):
        ragas_llm.model_args.update(model_args)
    elif hasattr(ragas_llm, "langchain_llm"):
        llm = ragas_llm.langchain_llm
        if hasattr(llm, "max_tokens") and "max_tokens" in model_args:
            llm.max_tokens = model_args["max_tokens"]
        if hasattr(llm, "temperature") and "temperature" in model_args:
            llm.temperature = model_args["temperature"]
        if "extra_body" in model_args and hasattr(llm, "model_kwargs"):
            llm.model_kwargs.update(model_args["extra_body"])


def _init_metrics(
    config: Dict[str, Any], ragas_llm: Any, ragas_embeddings: Any
) -> List[Any]:
    """Initialize Ragas metrics based on run mode flags.

    Pass --short-answer for HotpotQA-style datasets with single-word/phrase references.
    This retains all baseline metrics (FactualCorrectness, Faithfulness, AnswerRelevancy,
    ContextPrecision, ContextRecall) and adds SemanticSimilarity and ContainsAnswer.
    """
    retrieval_only = config["rag_eval_retrieval_only"]
    generation_only = config["rag_eval_generation_only"]
    short_answer = config.get("rag_eval_short_answer", False)

    if retrieval_only:
        return [
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ]
    if generation_only:
        base_metrics = [
            FactualCorrectness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ]
        if short_answer:
            base_metrics.extend([
                SemanticSimilarity(embeddings=ragas_embeddings),
                ContainsAnswer(),
            ])
        return base_metrics

    base_metrics = [
        FactualCorrectness(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]
    if short_answer:
        base_metrics.extend([
            SemanticSimilarity(embeddings=ragas_embeddings),
            ContainsAnswer(),
        ])
    return base_metrics


def init_evaluator(args_or_config: Any, dataset: Optional[Dataset] = None):
    """Initializes global evaluation settings, clients, LLMs, embeddings, and metrics based on consolidated config."""
    global openai_client, original_create, rag_client, ragas_llm, ragas_embeddings, metrics
    global rag_eval_retrieval_only, rag_eval_top_k
    global ragas_prompt_tokens, ragas_completion_tokens, ragas_llm_traces

    # Reset token counters for a fresh run
    ragas_prompt_tokens = 0
    ragas_completion_tokens = 0
    ragas_llm_traces = []

    config = _resolve_and_sync_config(args_or_config)

    # 1. Store execution state for callbacks
    rag_eval_retrieval_only = config["rag_eval_retrieval_only"]
    rag_eval_top_k = config["rag_eval_top_k"]

    # 2. Setup OpenAI client
    openai_client, original_create = _init_patched_openai_client(config)

    # 3. Setup RAG client
    rag_client = _init_rag_client(config, dataset, openai_client)

    # 4. Configure Ragas LLM with explicit input and output constraints
    model_name_val = config["openai_model_name"]
    base_url = config["openai_endpoint"]
    ragas_llm = cast(Any, llm_factory(model=model_name_val, base_url=base_url))

    # 5. Patch LLM and configure arguments
    _patch_bedrock_anthropic_llm(ragas_llm, model_name_val)
    _configure_ragas_llm_args(ragas_llm, model_name_val)

    # 6. Initialize embeddings and metrics
    ragas_embeddings = cast(
        BaseRagasEmbedding,
        embedding_factory(model=config["embeddings_model"], client=openai_client),
    )
    metrics = _init_metrics(config, ragas_llm, ragas_embeddings)


def load_enterprise_rag_bench(jsonl_path: Path) -> list[dict[str, Any]]:
    """Loads dataset from enterprise_rag_bench_questions.jsonl"""
    data_samples = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                data_samples.append(
                    {
                        "question_id": item.get("question_id"),
                        "question": item["user_input"],
                        "user_input": item["user_input"],
                        "reference": item["reference"],
                        "expected_doc_ids": item.get("expected_doc_ids", []),
                        "category": item.get("category", "basic"),
                    }
                )
    return data_samples


def load_hotpotqa(jsonl_path: Path) -> list[dict[str, Any]]:
    """Loads dataset from hotpotqa_questions.jsonl"""
    data_samples = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                data_samples.append(
                    {
                        "question_id": item.get("question_id"),
                        "question": item["user_input"],
                        "user_input": item["user_input"],
                        "reference": item["reference"],
                        "expected_doc_ids": item.get("expected_doc_ids", []),
                        "category": item.get("category", "basic"),
                        "level": item.get("level", "easy"),
                    }
                )
    return data_samples



def load_mock_dataset() -> list[dict[str, Any]]:
    """Loads a mock/fallback dataset for simple testing"""
    return [
        {
            "question_id": "mock_0001",
            "question": "What is ragas 0.3",
            "user_input": "What is ragas 0.3",
            "reference": "Ragas 0.3 focuses on experimentation as the central pillar, providing abstractions for datasets, experiments, and metrics. It supports evaluations for RAG, LLM workflows, and Agents.",
            "category": "basic",
        },
        {
            "question_id": "mock_0002",
            "question": "how are experiment results stored in ragas 0.3?",
            "user_input": "how are experiment results stored in ragas 0.3?",
            "reference": "Experiment results are stored in different backends like local or GDrive, typically under an experiments/ folder in the backend storage.",
            "category": "basic",
        },
        {
            "question_id": "mock_0003",
            "question": "What metrics are supported in ragas 0.3?",
            "user_input": "What metrics are supported in ragas 0.3?",
            "reference": "Ragas 0.3 provides abstractions for discrete, numerical, and ranking metrics.",
            "category": "semantic",
        },
    ]


def _load_samples_from_source(
    datasource_type: str, questions_path: Optional[str]
) -> List[Dict[str, Any]]:
    """Loads raw data samples depending on the datasource type."""
    if datasource_type == "enterprise_rag_bench":
        if not questions_path:
            raise ValueError(
                "questions_path must be specified when using enterprise_rag_bench datasource."
            )
        jsonl_path = Path(questions_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Questions file not found: {jsonl_path}")
        return load_enterprise_rag_bench(jsonl_path)

    if datasource_type == "hotpotqa":
        if not questions_path:
            raise ValueError(
                "questions_path must be specified when using hotpotqa datasource."
            )
        jsonl_path = Path(questions_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Questions file not found: {jsonl_path}")
        return load_hotpotqa(jsonl_path)

    if datasource_type == "mock":
        return load_mock_dataset()

    raise ValueError(
        f"Unsupported datasource: {datasource_type!r}. Only 'enterprise_rag_bench', 'hotpotqa', or 'mock' is supported."
    )


def _filter_samples_by_category(
    data_samples: List[Dict[str, Any]], limit_per_category: int
) -> List[Dict[str, Any]]:
    """Filters data samples by category count limit."""
    category_counts = {}
    filtered_samples = []
    for sample in data_samples:
        cat = sample.get("category", "basic")
        if "level" in sample:
            cat = f"{cat}_{sample['level']}"
        count = category_counts.get(cat, 0)
        if count < limit_per_category:
            filtered_samples.append(sample)
            category_counts[cat] = count + 1
    return filtered_samples



def load_dataset(
    limit: Optional[int] = None,
    datasource: Optional[str] = None,
    limit_per_category: Optional[int] = None,
    questions_path: Optional[str] = None,
):
    """Loads dataset for evaluation based on the configured datasource type and limit."""
    if not datasource:
        raise ValueError("datasource must be specified.")
    datasource_type = datasource.lower()
    dataset_name = "test_dataset"

    dataset = Dataset(
        name=dataset_name,
        backend="local/csv",
        root_dir="evals",
    )

    data_samples = _load_samples_from_source(datasource_type, questions_path)

    if not data_samples:
        raise ValueError(f"No questions loaded from datasource: {datasource_type}")

    # Filter by category limit if configured
    if limit_per_category is not None:
        data_samples = _filter_samples_by_category(data_samples, limit_per_category)

    if limit is not None:
        data_samples = data_samples[:limit]

    for sample in data_samples:
        dataset.append(sample)

    dataset.save()
    return dataset


@experiment()
async def run_experiment(row):
    """
    Phase 1: Run your RAG client pipeline exclusively.
    Do not run evaluations inside this individual row loop.
    """

    retrieval_only = (
        rag_eval_retrieval_only
        if rag_eval_retrieval_only is not None
        else settings.rag_eval_retrieval_only
    )
    start_time = time.time()

    top_k_val = (
        rag_eval_top_k if rag_eval_top_k is not None else settings.rag_eval_top_k
    )

    if retrieval_only:
        retrieved_docs = rag_client.retrieve_documents(row["question"], top_k=top_k_val) or []
        latency = time.time() - start_time
        retrieved_contexts = [doc["content"] for doc in retrieved_docs]
        retrieved_doc_ids = [
            (doc.get("metadata") or {}).get("doc_id")
            for doc in retrieved_docs
            if (doc.get("metadata") or {}).get("doc_id")
        ]
        return {
            **row,
            "response": "N/A",
            "retrieved_contexts": retrieved_contexts,
            "retrieved_doc_ids": retrieved_doc_ids,
            "latency": latency,
            "total_tokens": 0,
            "log_file": " ",
        }

    response_data = rag_client.query(row["question"], top_k=top_k_val)
    latency = time.time() - start_time

    answer = response_data.get("answer", "")
    retrieved_docs = response_data.get("retrieved_docs") or []
    retrieved_contexts = [doc["content"] for doc in retrieved_docs]
    retrieved_doc_ids = [
        (doc.get("metadata") or {}).get("doc_id")
        for doc in retrieved_docs
        if (doc.get("metadata") or {}).get("doc_id")
    ]
    usage = response_data.get("usage") or {}
    total_tokens = usage.get("total_tokens", 0)
    log_file = response_data.get("logs", " ")

    logger.debug(f"\n[RAG PIPELINE RUN] Query: {row['question']!r}")
    logger.debug(f"  Generated Response: {answer!r}")
    logger.debug("  Retrieved Contexts:")
    for idx, ctx in enumerate(retrieved_contexts):
        logger.debug(f"    [{idx+1}] {ctx[:200]!r}...")

    return {
        **row,
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_doc_ids": retrieved_doc_ids,
        "latency": latency,
        "total_tokens": total_tokens,
        "log_file": log_file,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Ragas evaluations")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of questions to evaluate"
    )
    parser.add_argument(
        "--datasource", type=str, default=None, help="Override active datasource"
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Override top_k documents to retrieve"
    )
    parser.add_argument(
        "--openai-api-key", type=str, default=None, help="OpenAI/LiteLLM API key"
    )
    parser.add_argument(
        "--openai-endpoint", type=str, default=None, help="OpenAI/LiteLLM endpoint URL"
    )
    parser.add_argument(
        "--openai-model-name", type=str, default=None, help="Model name for generation"
    )
    parser.add_argument(
        "--embeddings-model", type=str, default=None, help="Model name for embeddings"
    )
    parser.add_argument(
        "--retrieval-only", action="store_true", help="Run retrieval evaluation only"
    )
    parser.add_argument(
        "--generation-only", action="store_true", help="Run generation evaluation only"
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=None,
        help="Limit number of questions to evaluate per category",
    )
    parser.add_argument(
        "--questions-path",
        type=str,
        default=None,
        help="Path to benchmark questions JSONL dataset",
    )
    parser.add_argument(
        "--compute-model-eval",
        action="store_true",
        help="Use PrecomputedRAG to evaluate pre-existing model answers and contexts in the datasource",
    )
    parser.add_argument(
        "--enable-trace-log",
        action="store_true",
        help="Capture the raw agentic SSE stream logs in a separate file (agentic_run_{run_id}.log)",
    )
    parser.add_argument(
        "--short-answer",
        action="store_true",
        help="Use SemanticSimilarity + ContainsAnswer (for HotpotQA-style short-answer datasets)",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use AgenticRAG — routes queries through caipe-supervisor's A2A endpoint "
        "instead of rag-server directly. Requires the rag_context patch applied to "
        "agent.py in your CAIPE instance.",
    )
    parser.add_argument(
        "--agent-api-url",
        default=None,
        help="Override the agent API endpoint URL (default: CAIPE_AGENT_API_URL env or http://localhost:8000).",
    )
    parser.add_argument(
        "--agent-api-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for calls to the agent API (default: 120.0).",
    )
    return parser.parse_args()


def _validate_questions_path(config: Dict[str, Any]) -> None:
    """Validate path if using enterprise_rag_bench, hotpotqa or model evaluation is enabled."""
    datasource_type = config["ragas_datasource"].lower()
    if (
        datasource_type in ("enterprise_rag_bench", "hotpotqa")
        or config["compute_model_eval"]
    ):
        questions_path = config["questions_path"]
        if not questions_path:
            raise ValueError(
                "questions_path must be specified (via CLI --questions-path or env QUESTIONS_PATH) "
                "when using enterprise_rag_bench, hotpotqa or compute_model_eval."
            )
        path_obj = Path(questions_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Questions file not found: {path_obj}")


def _prepare_eval_dataset(df: pd.DataFrame) -> EvaluationDataset:
    """Convert generated results into an EvaluationDataset batch."""
    samples = []
    for idx, (_, r) in enumerate(df.iterrows()):
        logger.debug(f"\n[RAGAS EVAL INPUT SAMPLE {idx+1}]")
        logger.debug(f"  User Input: {r['question']!r}")
        logger.debug(f"  Response: {r['response']!r}")
        logger.debug(f"  Retrieved Contexts: {r['retrieved_contexts']}")
        logger.debug(f"  Reference: {r['reference']!r}")
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                response=r["response"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["reference"],
            )
        )
    return EvaluationDataset(samples=samples)


def _extract_statements(val: Any) -> List[Any]:
    """Safely extracts statements list from n_l_i_statement_prompt output (dict or Pydantic)."""
    if not isinstance(val, dict):
        return []

    prompt_data = val.get("n_l_i_statement_prompt")
    if not isinstance(prompt_data, dict):
        return []

    output_obj = prompt_data.get("output")
    if not output_obj:
        return []

    # If output_obj is a Pydantic model / custom object
    if hasattr(output_obj, "statements"):
        return getattr(output_obj, "statements") or []

    # If output_obj is a dict
    if isinstance(output_obj, dict):
        return output_obj.get("statements") or []

    return []


def _run_evaluation(
    eval_dataset: EvaluationDataset,
    df: pd.DataFrame,
    metrics_list: List[Any],
    legacy_embeddings: Any,
    ragas_llm_obj: Any,
) -> List[str]:
    """Run Ragas evaluate metrics across the dataset and update scores."""
    from ragas import RunConfig

    # Update embedding on answer relevancy metric to use wrapped object
    for m in metrics_list:
        if hasattr(m, "embeddings"):
            m.embeddings = legacy_embeddings

    # Pre-initialize all metric columns to 0.0 to prevent KeyError in case of job failures
    metric_names = [m.name for m in metrics_list]
    for col in metric_names:
        df[col] = 0.0
    df["faithfulness_reason"] = ""
    df["factual_correctness_reason"] = ""

    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics_list,
            llm=ragas_llm_obj,
            embeddings=legacy_embeddings,
            run_config=RunConfig(timeout=cast(Any, None), max_workers=1),
            show_progress=True,
        )

        # Merge the evaluation scores back into our tracking DataFrame
        scores_df = cast(Any, results).to_pandas()

        # Clean up column layout boundaries for merging operational behaviors
        for col in metric_names:
            matching_cols = [
                c for c in scores_df.columns if c == col or c.startswith(f"{col}(")
            ]
            if matching_cols:
                df[col] = scores_df[matching_cols[0]].values

        # Extract reasons from traces if available
        reasons_dict: Dict[str, List[str]] = {
            "faithfulness": [],
            "factual_correctness": []
        }
        traces = getattr(results, "traces", [])
        for trace in traces:
            # 1. Faithfulness reasoning
            faith_reason = ""
            if "faithfulness" in trace:
                statements = _extract_statements(trace["faithfulness"])
                if statements:
                    faith_reasons_list = []
                    for stmt in statements:
                        if isinstance(stmt, dict):
                            verdict = stmt.get("verdict")
                            statement = stmt.get("statement")
                            reason = stmt.get("reason")
                        else:
                            verdict = getattr(stmt, "verdict", None)
                            statement = getattr(stmt, "statement", None)
                            reason = getattr(stmt, "reason", None)
                        faith_reasons_list.append(f"[{verdict}] {statement} -> {reason}")
                    faith_reason = "; ".join(faith_reasons_list)
            reasons_dict["faithfulness"].append(faith_reason)

            # 2. Factual correctness reasoning
            fc_reason = ""
            if "factual_correctness" in trace:
                statements = _extract_statements(trace["factual_correctness"])
                if statements:
                    fc_reasons_list = []
                    for stmt in statements:
                        if isinstance(stmt, dict):
                            verdict = stmt.get("verdict")
                            statement = stmt.get("statement")
                            reason = stmt.get("reason")
                        else:
                            verdict = getattr(stmt, "verdict", None)
                            statement = getattr(stmt, "statement", None)
                            reason = getattr(stmt, "reason", None)
                        fc_reasons_list.append(f"[{verdict}] {statement} -> {reason}")
                    fc_reason = "; ".join(fc_reasons_list)
            reasons_dict["factual_correctness"].append(fc_reason)

        if "faithfulness" in metric_names and len(reasons_dict["faithfulness"]) == len(df):
            df["faithfulness_reason"] = reasons_dict["faithfulness"]
        if "factual_correctness" in metric_names and len(reasons_dict["factual_correctness"]) == len(df):
            df["factual_correctness_reason"] = reasons_dict["factual_correctness"]
    except Exception as e:
        logger.exception(f"Error during Ragas evaluate execution: {e}")
        # Fallback to 0.0 for all metrics on failure
        for col in metric_names:
            df[col] = 0.0

    return metric_names


def _analyze_failures(df: pd.DataFrame) -> Tuple[float, float]:
    """Label failure causes and compute retrieval recall and precision statistics."""
    if df.empty:
        return 0.0, 0.0

    df["failure_cause"] = "none"
    if "factual_correctness" in df.columns:
        df.loc[df["factual_correctness"] < 0.5, "failure_cause"] = (
            "incorrect_generation"
        )
    if "faithfulness" in df.columns:
        df.loc[df["faithfulness"] < 0.5, "failure_cause"] = "hallucination"
    if "context_recall" in df.columns:
        df.loc[df["context_recall"] < 0.5, "failure_cause"] = "poor_retrieval"

    recalls = []
    precisions = []
    for _, r in df.iterrows():
        expected = set(r.get("expected_doc_ids") or [])
        retrieved = set(r.get("retrieved_doc_ids") or [])
        if expected:
            hit = expected & retrieved
            recall = len(hit) / len(expected)
            precision = len(hit) / len(retrieved) if retrieved else 0.0
            recalls.append(recall)
            precisions.append(precision)
        else:
            recalls.append(None)
            precisions.append(None)
    df["retrieval_recall"] = recalls
    df["retrieval_precision"] = precisions
    valid_recalls = [r for r in recalls if r is not None]
    valid_precisions = [p for p in precisions if p is not None]

    avg_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0
    avg_precision = (
        sum(valid_precisions) / len(valid_precisions) if valid_precisions else 0.0
    )
    return avg_recall, avg_precision


def _calculate_llm_parsing_stats() -> Dict[str, int]:
    """Calculates operational statistics for the evaluator LLM calls."""
    global ragas_llm_traces
    total_calls = len(ragas_llm_traces)
    api_errors = sum(1 for t in ragas_llm_traces if t.get("error") is not None)
    successful_calls = total_calls - api_errors
    empty_responses = sum(
        1 for t in ragas_llm_traces 
        if t.get("error") is None and t.get("original_content") is None
    )
    json_parse_success_count = sum(
        1 for t in ragas_llm_traces if t.get("json_parse_success") is True
    )
    json_parse_failure_count = sum(
        1 for t in ragas_llm_traces if t.get("json_parse_success") is False
    )
    non_json_responses = sum(
        1 for t in ragas_llm_traces 
        if t.get("error") is None 
        and t.get("original_content") is not None 
        and t.get("json_parse_success") is None
    )
    return {
        "total_calls": total_calls,
        "api_errors": api_errors,
        "successful_calls": successful_calls,
        "empty_responses": empty_responses,
        "json_parse_success": json_parse_success_count,
        "json_parse_failures": json_parse_failure_count,
        "non_json_responses": non_json_responses,
    }


def _log_metrics_summary(
    config: Dict[str, Any],
    df: pd.DataFrame,
    metric_names: List[str],
    avg_recall: float,
    avg_precision: float,
    evaluation_time: float,
) -> None:
    """Log configuration and summary metrics."""
    keys_to_print = [
        "ragas_datasource",
        "rag_eval_top_k",
        "rag_eval_retrieval_only",
        "rag_eval_generation_only",
        "limit_per_category",
        "compute_model_eval",
        "rag_eval_short_answer",
    ]

    logger.info("\n--- RUN CONFIGURATION ---")
    for k in keys_to_print:
        if k in config:
            print_key = k.replace("rag_eval_", "").replace("ragas_", "")
            logger.info(f"{print_key}: {config[k]}")

    logger.info("\n--- OPERATIONAL BEHAVIOR ---")
    logger.info("RAG Pipeline:")
    if not df.empty and "latency" in df.columns:
        logger.info(f"  P50 Latency: {df['latency'].median():.2f}s")
        logger.info(f"  P95 Latency: {df['latency'].quantile(0.95):.2f}s")
    else:
        logger.info("  P50 Latency: 0.00s")
        logger.info("  P95 Latency: 0.00s")
        
    if not df.empty and "total_tokens" in df.columns:
        logger.info(f"  Total Tokens: {int(df['total_tokens'].sum())}")
    else:
        logger.info("  Total Tokens: 0")
        
    logger.info("")
    logger.info("Ragas Evaluator:")
    logger.info(f"  Evaluation Time: {evaluation_time:.2f}s")
    total_ragas_tokens = ragas_prompt_tokens + ragas_completion_tokens
    logger.info(f"  Prompt Tokens: {ragas_prompt_tokens}")
    logger.info(f"  Completion Tokens: {ragas_completion_tokens}")
    logger.info(f"  Total Evaluator Tokens: {total_ragas_tokens}")

    stats = _calculate_llm_parsing_stats()
    logger.info("\n--- EVALUATOR LLM PARSING & QUALITY STATS ---")
    logger.info(f"  Total LLM Calls: {stats['total_calls']}")
    logger.info(f"  Successful LLM Responses: {stats['successful_calls']}")
    logger.info(f"  API/Connection Errors: {stats['api_errors']}")
    logger.info(f"  Empty Choices/Responses: {stats['empty_responses']}")
    logger.info(f"  Successful JSON Parses: {stats['json_parse_success']}")
    logger.info(f"  JSON Parsing/Repair Failures: {stats['json_parse_failures']}")
    logger.info(f"  Plain Text / Non-JSON Responses: {stats['non_json_responses']}")

    logger.info("\n--- QUALITY METRICS AVERAGE ---")
    for m in metric_names:
        avg_val = df[m].mean() if (not df.empty and m in df.columns) else 0.0
        logger.info(f"Average {m}: {avg_val:.2f}")

    valid_recalls = [r for r in df["retrieval_recall"] if r is not None] if (not df.empty and "retrieval_recall" in df.columns) else []
    if valid_recalls:
        logger.info(f"Average retrieval_recall: {avg_recall:.2f}")

    valid_precisions = [p for p in df["retrieval_precision"] if p is not None] if (not df.empty and "retrieval_precision" in df.columns) else []
    if valid_precisions:
        logger.info(f"Average retrieval_precision: {avg_precision:.2f}")

    logger.info("\n--- FAILURE CAUSE ANALYSIS ---")
    if not df.empty and "failure_cause" in df.columns:
        logger.info(f"\n{df['failure_cause'].value_counts()}")
    else:
        logger.info("\nNo failure cause data available.")


async def _save_evaluation_outputs(
    experiment_name: str,
    df: pd.DataFrame,
    metric_names: List[str],
    avg_recall: float,
    avg_precision: float,
    config_args: Dict[str, Any],
    evaluation_time: float,
    datasource: Optional[str],
) -> None:
    """Save the results DataFrame to CSV and companion JSON summary asynchronously."""
    global ragas_prompt_tokens, ragas_completion_tokens
    output_dir = Path(".") / "evals" / "experiments"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"{experiment_name}.csv"

    # Copy DataFrame to avoid modifying caller's data
    df = df.copy()

    # Initialize per-row token counters in df
    df["evaluator_prompt_tokens"] = 0
    df["evaluator_completion_tokens"] = 0
    df["evaluator_total_tokens"] = 0
    df["evaluator_evaluation_time_seconds"] = evaluation_time

    # Match and attribute trace token usage back to individual rows
    for trace in ragas_llm_traces:
        usage = trace.get("usage")
        if not isinstance(usage, dict):
            continue
        p_tokens = usage.get("prompt_tokens", 0)
        c_tokens = usage.get("completion_tokens", 0)

        # Concatenate prompt message content to check for substrings
        prompt_text = " ".join(
            [
                m.get("content", "")
                for m in trace.get("messages", [])
                if isinstance(m, dict)
            ]
        )

        matched_idx = None
        for idx, row in df.iterrows():
            q = row.get("question")
            ans = row.get("response")
            ref = row.get("reference")

            # Check if any identifier unique to this row appears in the prompt messages
            if (
                (q and q in prompt_text)
                or (ans and ans in prompt_text)
                or (ref and ref in prompt_text)
            ):
                matched_idx = idx
                break

        if matched_idx is not None:
            df.at[matched_idx, "evaluator_prompt_tokens"] += p_tokens
            df.at[matched_idx, "evaluator_completion_tokens"] += c_tokens
            df.at[matched_idx, "evaluator_total_tokens"] += p_tokens + c_tokens

    # Create a summary row with overall averages
    summary_row: dict[str, Any] = dict.fromkeys(df.columns, "")
    summary_row["question"] = "AVERAGE_METRICS"
    summary_row["latency"] = df["latency"].mean() if (not df.empty and "latency" in df.columns) else 0.0
    summary_row["total_tokens"] = df["total_tokens"].mean() if (not df.empty and "total_tokens" in df.columns) else 0.0
    for m in metric_names:
        summary_row[m] = df[m].mean() if (not df.empty and m in df.columns) else 0.0
    summary_row["retrieval_recall"] = avg_recall
    summary_row["retrieval_precision"] = avg_precision
    summary_row["failure_cause"] = "N/A"
    summary_row["evaluator_evaluation_time_seconds"] = evaluation_time
    summary_row["evaluator_prompt_tokens"] = df["evaluator_prompt_tokens"].mean() if not df.empty else 0.0
    summary_row["evaluator_completion_tokens"] = df["evaluator_completion_tokens"].mean() if not df.empty else 0.0
    summary_row["evaluator_total_tokens"] = df["evaluator_total_tokens"].mean() if not df.empty else 0.0

    df_with_summary = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    await asyncio.to_thread(df_with_summary.to_csv, csv_path, index=False)
    logger.info(f"\nDetailed results saved to: {csv_path.resolve()}")

    parsing_stats = _calculate_llm_parsing_stats()
    summary_json_path = output_dir / f"{experiment_name}_summary.json"
    summary_data = {
        "experiment_name": experiment_name,
        "datasource": datasource,
        "config_args": config_args,
        "p50_latency": df["latency"].median() if (not df.empty and "latency" in df.columns) else 0.0,
        "p95_latency": df["latency"].quantile(0.95) if (not df.empty and "latency" in df.columns) else 0.0,
        "total_tokens": int(df["total_tokens"].sum()) if (not df.empty and "total_tokens" in df.columns) else 0,
        "metrics": {
            **{m: (df[m].mean() if (not df.empty and m in df.columns) else 0.0) for m in metric_names},
            "retrieval_recall": avg_recall,
            "retrieval_precision": avg_precision,
        },
        "average_retrieval_recall": avg_recall,
        "average_retrieval_precision": avg_precision,
        "ragas_evaluator_usage": {
            "evaluation_time_seconds": evaluation_time,
            "prompt_tokens": ragas_prompt_tokens,
            "completion_tokens": ragas_completion_tokens,
            "total_tokens": ragas_prompt_tokens + ragas_completion_tokens,
        },
        "evaluator_parsing_stats": parsing_stats,
    }

    def save_json():
        with open(summary_json_path, "w") as f:
            json.dump(summary_data, f, indent=4)

        traces_path = Path(".") / "evals" / "logs" / f"{experiment_name}_evaluator_traces.json"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traces_path, "w") as f:
            json.dump(ragas_llm_traces, f, indent=2)
        logger.info(f"Evaluator LLM traces saved to: {traces_path.resolve()}")

    await asyncio.to_thread(save_json)
    logger.info(f"Summary metrics saved to: {summary_json_path.resolve()}")


class LegacyEmbeddingsWrapper:
    """Wrapper class to adapt Ragas embeddings to LangChain-compatible interface expected by legacy metrics."""

    def __init__(self, emb: Any):
        """Initializes LegacyEmbeddingsWrapper with the underlying embedding model."""
        self.emb = emb

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string into a vector."""
        return self.emb.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of document strings into a list of vectors."""
        return self.emb.embed_texts(texts)

    def embed_text(self, text: str) -> List[float]:
        """Adapt to modern BaseRagasEmbedding interface."""
        return self.emb.embed_text(text)

    async def aembed_text(self, text: str) -> List[float]:
        """Adapt to modern BaseRagasEmbedding async interface, falling back to sync if client lacks async support."""
        try:
            return await self.emb.aembed_text(text)
        except (TypeError, AttributeError):
            return self.emb.embed_text(text)


async def main():
    """Main function to parse CLI arguments, run RAG generation, and evaluate results using Ragas metrics."""
    args = _parse_args()

    # 1. Load configuration (resolving CLI args and environment variables in a single place)
    config = load_configuration(args)

    # 2. Validate path if using enterprise_rag_bench or model evaluation is enabled
    _validate_questions_path(config)

    # 3. Load dataset first (this reads the file once and applies limits/filters)
    dataset = load_dataset(
        limit=config["ragas_limit"],
        datasource=config["ragas_datasource"],
        limit_per_category=config["limit_per_category"],
        questions_path=config["questions_path"],
    )
    logger.info(f"Dataset loaded successfully: {dataset}")

    # 4. Initialize evaluator using the pre-filtered dataset and consolidated configuration
    init_evaluator(config, dataset=dataset)

    datasource_name = config.get("ragas_datasource")
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(datasource_name, str) and datasource_name.strip():
        experiment_name = f"{datasource_name.strip()}_{now_str}"
    else:
        experiment_name = now_str

    try:
        logger.info("\n--- PHASE 1: RUNNING RAG PIPELINE GENERATION ---")
        experiment_results = await run_experiment.arun(dataset, name=experiment_name)

        # Convert generated results directly into a pandas frame
        df = pd.DataFrame(experiment_results)

        logger.info("\n--- PHASE 2: BATCH EVALUATION VIA RAGAS ---")
        eval_dataset = _prepare_eval_dataset(df)

        legacy_embeddings: Any = LegacyEmbeddingsWrapper(ragas_embeddings)

        # Run batch evaluation smoothly outside of active async worker threads
        logger.info("Evaluating metrics across whole dataset...")
        start_eval_time = time.time()
        metric_names = _run_evaluation(
            eval_dataset=eval_dataset,
            df=df,
            metrics_list=metrics,
            legacy_embeddings=legacy_embeddings,
            ragas_llm_obj=ragas_llm,
        )
        evaluation_time = time.time() - start_eval_time

        avg_recall, avg_precision = _analyze_failures(df)

        _log_metrics_summary(
            config, df, metric_names, avg_recall, avg_precision, evaluation_time
        )

        config_args = {
            k: v for k, v in vars(args).items() if v is not None and k != "openai_api_key"
        }

        # Combine detailed log files if they exist
        combined_log_path_str = " "
        if "log_file" in df.columns:
            log_files = df["log_file"].dropna().unique()
            valid_log_files = []
            combined_logs = []
            for lf in log_files:
                lf_str = str(lf).strip()
                if lf_str and lf_str != "N/A" and os.path.exists(lf_str):
                    try:
                        with open(lf_str, "r") as f:
                            combined_logs.append(json.load(f))
                        valid_log_files.append(lf_str)
                    except Exception as e:
                        logger.warning(f"Failed to read log file {lf_str}: {e}")

            if combined_logs:
                combined_log_path = Path("evals") / "logs" / f"{experiment_name}_detailed.json"
                combined_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(combined_log_path, "w") as f:
                    json.dump(combined_logs, f, indent=2)
                logger.info(f"Combined detailed logs saved to: {combined_log_path.resolve()}")
                combined_log_path_str = str(combined_log_path)

                # Clean up the individual log files
                for lf_str in valid_log_files:
                    try:
                        os.remove(lf_str)
                    except Exception as e:
                        logger.warning(f"Failed to delete individual log file {lf_str}: {e}")

        df["log_file"] = combined_log_path_str

        await _save_evaluation_outputs(
            experiment_name=experiment_name,
            df=df,
            metric_names=metric_names,
            avg_recall=avg_recall,
            avg_precision=avg_precision,
            config_args=config_args,
            evaluation_time=evaluation_time,
            datasource=datasource_name,
        )

        # Rename the dataset CSV to match the experiment name
        old_dataset_path = Path("evals") / "datasets" / "test_dataset.csv"
        if old_dataset_path.exists():
            new_dataset_path = Path("evals") / "datasets" / f"{experiment_name}.csv"
            await asyncio.to_thread(old_dataset_path.rename, new_dataset_path)
            logger.info(
                f"Dataset renamed to match experiment: {new_dataset_path.resolve()}"
            )
    finally:
        cleanup_evaluator()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    asyncio.run(main())
