"""Streamlined row-level token tracking using instance trace-climbing propagation."""

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class RowLevelTokenCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.run_id_2_row_idx: Dict[str, int] = {}
        self.parent_maps: Dict[str, str] = {}
        self.row_2_run_id_token: Dict[int, List[tuple]] = {}
        self._lock = threading.Lock()

    def _register_run(self, run_id: UUID, parent_run_id: Optional[UUID], metadata: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            r_id = str(run_id)
            p_id = str(parent_run_id) if parent_run_id else None
            
            if p_id:
                self.parent_maps[r_id] = p_id

            metadata = metadata or {}
            type_str = str(metadata.get("type")).lower()
            
            # FIX 1: Substring check to safely bypass Enum structures like <ChainType.ROW: 'row'>
            if "row" in type_str:
                row_index = metadata.get("row_index")
                if row_index is not None:
                    self.run_id_2_row_idx[r_id] = row_index
                    logger.debug(f"[Handler] Root ROW registered: run_id={r_id} -> row_index={row_index}")

    def _get_row_index(self, run_id: str) -> Optional[int]:
        """FIX 2: Climb up the trace nodes recursively to inherit the root row index."""
        current = run_id
        while current:
            if current in self.run_id_2_row_idx:
                return self.run_id_2_row_idx[current]
            current = self.parent_maps.get(current)
        return None

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        self._register_run(run_id, parent_run_id, kwargs.get("metadata"))

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        self._register_run(run_id, parent_run_id, kwargs.get("metadata"))

    # FIX 3: Capture chat model initialization lifecycle parameters 
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[Any]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        self._register_run(run_id, parent_run_id, kwargs.get("metadata"))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        with self._lock:
            r_id = str(run_id)
            p_id = str(parent_run_id) if parent_run_id else None
            
            row_idx = self._get_row_index(r_id)
            if row_idx is not None:
                usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
                p = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                c = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
                
                if row_idx not in self.row_2_run_id_token:
                    self.row_2_run_id_token[row_idx] = []
                self.row_2_run_id_token[row_idx].append((p, c, run_id, parent_run_id))
                logger.debug(f"[Handler] Attributed row {row_idx}: prompt={p}, completion={c} (run_id={r_id})")

    def get_row_usage(self, row_index: int) -> Dict[str, Any]:
        with self._lock:
            trace_tuples = self.row_2_run_id_token.get(row_index, [])
            if not trace_tuples:
                return {
                    "evaluator_prompt_tokens": 0,
                    "evaluator_completion_tokens": 0,
                    "evaluator_total_tokens": 0,
                    "ragas_row_run_id": "N/A",
                    "ragas_batch_run_id": "N/A"
                }
            total_prompt = sum(t[0] for t in trace_tuples)
            total_completion = sum(t[1] for t in trace_tuples)
            return {
                "evaluator_prompt_tokens": total_prompt,
                "evaluator_completion_tokens": total_completion,
                "evaluator_total_tokens": total_prompt + total_completion,
                "ragas_row_run_id": str(trace_tuples[0][2]),
                "ragas_batch_run_id": str(trace_tuples[0][3]) if trace_tuples[0][3] else "N/A"
            }


class AsyncRowLevelTokenCallbackHandler(AsyncCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.run_id_2_row_idx: Dict[str, int] = {}
        self.parent_maps: Dict[str, str] = {}
        self.row_2_run_id_token: Dict[int, List[tuple]] = {}
        self._lock = threading.Lock()

    def _register_run(self, run_id: UUID, parent_run_id: Optional[UUID], metadata: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            r_id = str(run_id)
            p_id = str(parent_run_id) if parent_run_id else None
            
            if p_id:
                self.parent_maps[r_id] = p_id

            metadata = metadata or {}
            type_str = str(metadata.get("type")).lower()
            if "row" in type_str:
                row_index = metadata.get("row_index")
                if row_index is not None:
                    self.run_id_2_row_idx[r_id] = row_index
                    logger.debug(f"[AsyncHandler] Root ROW registered: run_id={r_id} -> row_index={row_index}")

    def _get_row_index(self, run_id: str) -> Optional[int]:
        current = run_id
        while current:
            if current in self.run_id_2_row_idx:
                return self.run_id_2_row_idx[current]
            current = self.parent_maps.get(current)
        return None

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        self._register_run(run_id, parent_run_id, kwargs.get("metadata"))

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        self._register_run(run_id, parent_run_id, kwargs.get("metadata"))

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[Any]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        self._register_run(run_id, parent_run_id, kwargs.get("metadata"))

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        with self._lock:
            r_id = str(run_id)
            p_id = str(parent_run_id) if parent_run_id else None
            
            row_idx = self._get_row_index(r_id)
            if row_idx is not None:
                usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
                p = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                c = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
                
                if row_idx not in self.row_2_run_id_token:
                    self.row_2_run_id_token[row_idx] = []
                self.row_2_run_id_token[row_idx].append((p, c, run_id, parent_run_id))
                logger.debug(f"[AsyncHandler] Attributed row {row_idx}: prompt={p}, completion={c} (run_id={r_id})")

    def get_row_usage(self, row_index: int) -> Dict[str, Any]:
        with self._lock:
            trace_tuples = self.row_2_run_id_token.get(row_index, [])
            if not trace_tuples:
                return {
                    "evaluator_prompt_tokens": 0,
                    "evaluator_completion_tokens": 0,
                    "evaluator_total_tokens": 0,
                    "ragas_row_run_id": "N/A",
                    "ragas_batch_run_id": "N/A"
                }
            total_prompt = sum(t[0] for t in trace_tuples)
            total_completion = sum(t[1] for t in trace_tuples)
            return {
                "evaluator_prompt_tokens": total_prompt,
                "evaluator_completion_tokens": total_completion,
                "evaluator_total_tokens": total_prompt + total_completion,
                "ragas_row_run_id": str(trace_tuples[0][2]),
                "ragas_batch_run_id": str(trace_tuples[0][3]) if trace_tuples[0][3] else "N/A"
            }
