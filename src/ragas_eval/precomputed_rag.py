import os
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from ragas_eval.rag import CaipeRetriever

logger = logging.getLogger(__name__)


def normalize_string(s: Any) -> str:
    """Normalize whitespace, casing, and punctuation for robust string comparison."""
    if not isinstance(s, str):
        return ""
    # Lowercase and strip
    s = s.lower().strip()
    # Normalize whitespaces
    return " ".join(s.split())


class PrecomputedRAG:
    """
    Mock RAG client that yields precomputed answers and retrieved contexts.
    1. Uses the reference (ideal answer) as the generated response.
    2. Queries CaipeRetriever live using the reference answer to retrieve ideal contexts.
    """

    def __init__(self, dataset_path: Optional[str] = None, preloaded_samples: Optional[List[Any]] = None):
        """Initializes the PrecomputedRAG instance with the path to the precomputed dataset or preloaded samples."""
        self.dataset_path = dataset_path
        self.data_by_question: Dict[str, Dict[str, Any]] = {}
        self.retriever = CaipeRetriever()
        if preloaded_samples is not None:
            self.load_from_samples(preloaded_samples)
        else:
            self.load_dataset()

    def load_from_samples(self, samples: List[Any]):
        """Indexes preloaded samples (either SingleTurnSample objects or dictionaries)."""
        for item in samples:
            if hasattr(item, "user_input"):
                question = item.user_input
                reference = item.reference or ""
                category = getattr(item, "category", "basic")
            else:
                question = item.get("question") or item.get("user_input")
                reference = item.get("reference") or ""
                category = item.get("category") or "basic"

            if question is None:
                continue

            norm_q = normalize_string(str(question))
            self.data_by_question[norm_q] = {
                "question": question,
                "answer": reference,
                "reference": reference,
                "category": category,
            }
        logger.info(f"Successfully indexed {len(self.data_by_question)} questions from preloaded samples.")
        print(f"Successfully indexed {len(self.data_by_question)} questions from preloaded samples.")

    def load_dataset(self):
        """Loads and indexes the precomputed dataset file."""
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided to load the dataset.")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Precomputed data file not found: {self.dataset_path}")

        logger.info(f"Loading precomputed RAG dataset from {self.dataset_path}")
        print(f"Loading precomputed RAG dataset from {self.dataset_path}")

        if self.dataset_path.endswith(".jsonl"):
            df = pd.read_json(self.dataset_path, lines=True)
        elif self.dataset_path.endswith(".json"):
            df = pd.read_json(self.dataset_path)
        else:
            df = pd.read_csv(self.dataset_path)

        for _, row in df.iterrows():
            row_dict = row.to_dict()

            # Identify query/question key
            question = row_dict.get("question")
            if question is None or pd.isna(question):
                question = row_dict.get("user_input")
            if question is None or pd.isna(question):
                continue

            reference = row_dict.get("reference") or ""
            category = row_dict.get("category") or "basic"

            # 1. Use reference (ideal answer) as the generated response!
            answer = reference

            norm_q = normalize_string(str(question))
            self.data_by_question[norm_q] = {
                "question": question,
                "answer": answer,
                "reference": reference,
                "category": category,
            }

        logger.info(f"Successfully indexed {len(self.data_by_question)} questions from dataset.")
        print(f"Successfully indexed {len(self.data_by_question)} questions from dataset.")

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve contexts live from CaipeRetriever using the reference answer."""
        norm_q = normalize_string(query)
        data = self.data_by_question.get(norm_q)

        if not data:
            # Try fuzzy check: substring match if exact fails
            for key, val in self.data_by_question.items():
                if norm_q in key or key in norm_q:
                    data = val
                    break

        if not data:
            logger.warning(f"No precomputed data found for query: {query!r}")
            print(f"WARNING: No precomputed data found for query: {query!r}")
            return []

        # 2. Use the reference (ideal answer) to retrieve the docs from CaipeRetriever
        reference_query = data["reference"]

        logger.info(f"Retrieving documents using reference query: {reference_query[:100]}...")
        print(f"Retrieving documents using reference query: {reference_query[:100]}...")

        top_docs = self.retriever.get_top_k(reference_query, k=top_k)

        retrieved_docs = []
        for idx, score in top_docs:
            if idx < len(self.retriever.documents):
                content = self.retriever.documents[idx]
                metadata = {}
                if hasattr(self.retriever, "documents_metadata") and idx < len(
                    self.retriever.documents_metadata
                ):
                    metadata = self.retriever.documents_metadata[idx]

                retrieved_docs.append(
                    {
                        "content": content,
                        "similarity_score": score,
                        "document_id": idx,
                        "metadata": metadata,
                    }
                )
        return retrieved_docs

    def query(self, question: str, top_k: int = 3, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Return precomputed ideal answer and retrieved documents."""
        norm_q = normalize_string(question)
        data = self.data_by_question.get(norm_q)

        if not data:
            # Try fuzzy check: substring match if exact fails
            for key, val in self.data_by_question.items():
                if norm_q in key or key in norm_q:
                    data = val
                    break

        if not data:
            logger.warning(f"No precomputed data found for question: {question!r}")
            print(f"WARNING: No precomputed data found for question: {question!r}")
            return {
                "answer": "No precomputed answer available.",
                "run_id": run_id or "precomputed_run",
                "retrieved_docs": [],
                "usage": {},
                "logs": " "
            }

        retrieved_docs = self.retrieve_documents(question, top_k)
        return {
            "answer": data["answer"],
            "run_id": run_id or "precomputed_run",
            "retrieved_docs": retrieved_docs,
            "usage": {},
            "logs": " "
        }
