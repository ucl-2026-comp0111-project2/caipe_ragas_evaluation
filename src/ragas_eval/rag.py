import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# Import consolidated configuration settings
from ragas_eval.config import settings

logger = logging.getLogger(__name__)

DOCUMENTS = [
    "Ragas are melodic frameworks in Indian classical music.",
    "There are many types of ragas, each with its own mood and time of day.",
    "Ragas are used to evoke specific emotions in the listener.",
    "The performance of a raga involves improvisation within a set structure.",
    "Ragas can be performed on various instruments or sung vocally.",
]


@dataclass
class TraceEvent:
    """Single event in the RAG application trace"""

    event_type: str
    component: str
    data: Dict[str, Any]


class BaseRetriever:
    """
    Base class for retrievers.
    Subclasses should implement the fit and get_top_k methods.
    """

    def __init__(self):
        """Initializes the BaseRetriever with an empty documents list."""
        self.documents = []

    def fit(self, documents: List[str]):
        """Store the documents"""
        self.documents = documents

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k most relevant documents for the query."""
        raise NotImplementedError("Subclasses should implement this method.")


class SimpleKeywordRetriever(BaseRetriever):
    """Ultra-simple keyword matching retriever"""

    def __init__(self):
        """Initializes SimpleKeywordRetriever by calling the base class constructor."""
        super().__init__()

    def _count_keyword_matches(self, query: str, document: str) -> int:
        """Count how many query words appear in the document"""
        query_words = query.lower().split()
        document_words = document.lower().split()
        matches = 0
        for word in query_words:
            if word in document_words:
                matches += 1
        return matches

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Get top k documents by keyword match count"""
        scores = []

        for i, doc in enumerate(self.documents):
            match_count = self._count_keyword_matches(query, doc)
            scores.append((i, match_count))

        # Sort by match count (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]


class CaipeRetriever(BaseRetriever):
    """Retriever that queries the CAIPE knowledge base endpoint"""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        datasource_id: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """Initializes CaipeRetriever with endpoint URL, datasource ID, and OIDC token settings."""
        super().__init__()
        self.endpoint_url = endpoint_url or settings.caipe_query_endpoint
        self.datasource_id = datasource_id or settings.caipe_datasource_id
        self._token = token or settings.caipe_oidc_token
        self.documents_metadata = []

    @property
    def token(self):
        """Returns the OIDC authentication token, retrieving or raising an error if missing."""
        if not self._token:
            raise ValueError(
                "OIDC Token is required but was not provided. Please set CAIPE_OIDC_TOKEN environment variable."
            )
        return self._token

    def fit(self, documents: List[str]):
        """CaipeRetriever doesn't support local fitting, it uses the remote KB"""
        self.documents = documents
        self.documents_metadata = [{} for _ in documents]

    def _parse_items(self, results: Any) -> List[dict]:
        """Extract items list from the query results."""
        if isinstance(results, list):
            return results
        if isinstance(results, dict):
            return results.get("results", [])
        return []

    def _extract_doc_content(self, item: dict, doc_obj: dict) -> str:
        """Extract page content/text from item or its nested document object."""
        content = (
            item.get("page_content")
            or doc_obj.get("page_content")
            or item.get("content")
            or doc_obj.get("content")
            or item.get("text")
            or doc_obj.get("text")
            or ""
        )
        if not content and "metadata" in item:
            meta = item["metadata"]
            if isinstance(meta, dict):
                content = meta.get("content") or meta.get("text") or ""
        return content

    def _extract_doc_id(self, item: dict, doc_obj: dict) -> Optional[str]:
        """Extract the document ID from metadata or document object."""
        item_meta = item.get("metadata") or {}
        if not isinstance(item_meta, dict):
            item_meta = {}
        doc_meta = doc_obj.get("metadata") or {}
        if not isinstance(doc_meta, dict):
            doc_meta = {}
        return (
            doc_obj.get("document_id")
            or item_meta.get("document_id")
            or doc_meta.get("document_id")
            or doc_obj.get("doc_id")
            or item_meta.get("doc_id")
            or doc_meta.get("doc_id")
        )

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Query CAIPE knowledge base"""
        # Clear local document cache to prevent memory leak across multiple queries
        self.documents = []
        self.documents_metadata = []

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "limit": k,
            "filters": {"datasource_id": self.datasource_id},
        }

        try:
            response = requests.post(self.endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json()

            items = self._parse_items(results)
            scores = []
            for item in items:
                score = item.get("score", 1.0)

                doc_obj = (
                    item.get("document", {})
                    if isinstance(item.get("document"), dict)
                    else {}
                )

                content = self._extract_doc_content(item, doc_obj)

                try:
                    idx = self.documents.index(content)
                except ValueError:
                    self.documents.append(content)
                    doc_id = self._extract_doc_id(item, doc_obj)
                    self.documents_metadata.append({"doc_id": doc_id} if doc_id else {})
                    idx = len(self.documents) - 1

                scores.append((idx, score))

            return scores[:k]
        except Exception:
            logger.exception("Error querying CAIPE KB")
            return []


class BaseRAG:
    """
    Simple RAG system that:
    1. accepts a llm client
    2. uses simple keyword matching to retrieve relevant documents
    3. uses the llm client to generate a response based on the retrieved documents when a query is made
    """

    def __init__(
        self,
        llm_client,
        model_name: str,
        retriever: Optional[BaseRetriever] = None,
        system_prompt: Optional[str] = None,
        logdir: str = "logs",
    ):
        """
        Initialize RAG system

        Args:
            llm_client: LLM client with a generate() method
            model_name: Name of the OpenAI/generation model to use
            retriever: Document retriever (defaults to SimpleKeywordRetriever)
            system_prompt: System prompt template for generation
            logdir: Directory for trace log files
        """
        self.llm_client = llm_client
        self.retriever = retriever or SimpleKeywordRetriever()
        self.model_name = model_name
        self.system_prompt = (
            system_prompt
            or """Answer the following question based on the provided documents:
                                Question: {query}
                                Documents:
                                {context}
                                Answer:
                            """
        )
        self.is_fitted = False
        self.traces = []
        self.logdir = logdir

        # Create log directory if it doesn't exist
        os.makedirs(self.logdir, exist_ok=True)

        # Initialize tracing
        self.traces.append(
            TraceEvent(
                event_type="init",
                component="rag_system",
                data={
                    "retriever_type": type(self.retriever).__name__,
                    "system_prompt_length": len(self.system_prompt),
                    "logdir": self.logdir,
                },
            )
        )

    @property
    def documents(self):
        """Proxy to the retriever's documents"""
        return self.retriever.documents

    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base"""
        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "add_documents",
                    "num_new_documents": len(documents),
                    "total_documents_before": len(self.documents),
                    "document_lengths": [len(doc) for doc in documents],
                },
            )
        )

        current_docs = list(self.documents)
        current_docs.extend(documents)
        # Refit retriever with all documents
        self.retriever.fit(current_docs)
        self.is_fitted = True

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def set_documents(self, documents: List[str]):
        """Set documents (replacing any existing ones)"""
        old_doc_count = len(self.documents)

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "set_documents",
                    "num_new_documents": len(documents),
                    "old_document_count": old_doc_count,
                    "document_lengths": [len(doc) for doc in documents],
                },
            )
        )

        self.retriever.fit(documents)
        self.is_fitted = True

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for the query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing document info
        """
        # For CaipeRetriever, we allow retrieval even if not fitted locally
        if not self.is_fitted and not isinstance(self.retriever, CaipeRetriever):
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_start",
                    "query": query,
                    "query_length": len(query),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        top_docs = self.retriever.get_top_k(query, k=top_k)

        retrieved_docs = []
        for idx, score in top_docs:
            if score > 0:  # Only include documents with positive similarity scores
                # Safety check for index
                if idx < len(self.documents):
                    content = self.documents[idx]
                    metadata = {}
                    if hasattr(self.retriever, "documents_metadata") and idx < len(
                        self.retriever.documents_metadata
                    ):
                        metadata = self.retriever.documents_metadata[idx]
                else:
                    # This should rarely happen but handles edge cases where document wasn't stored
                    content = "Document content not available."
                    metadata = {}

                retrieved_docs.append(
                    {
                        "content": content,
                        "similarity_score": score,
                        "document_id": idx,
                        "metadata": metadata,
                    }
                )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_complete",
                    "num_retrieved": len(retrieved_docs),
                    "scores": [doc["similarity_score"] for doc in retrieved_docs],
                    "document_ids": [doc["document_id"] for doc in retrieved_docs],
                },
            )
        )

        return retrieved_docs

    def generate_response(
        self,
        query: str,
        top_k: int = 3,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Generate response to query using retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve
            retrieved_docs: Optional list of previously retrieved documents
            model_name: Optional model override for generation

        Returns:
            Generated response
        """
        # For CaipeRetriever, we allow retrieval even if not fitted locally
        if not self.is_fitted and not isinstance(self.retriever, CaipeRetriever):
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        # Retrieve relevant documents if not provided
        if retrieved_docs is None:
            retrieved_docs = self.retrieve_documents(query, top_k)

        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate response using LLM client
        prompt = self.system_prompt.format(query=query, context=context)
        model_name = model_name or self.model_name

        self.traces.append(
            TraceEvent(
                event_type="llm_call",
                component="openai_api",
                data={
                    "operation": "generate_response",
                    "model": model_name,
                    "query": query,
                    "prompt_length": len(prompt),
                    "context_length": len(context),
                    "num_context_docs": len(retrieved_docs),
                },
            )
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,  # Increased to prevent truncation
            )

            if not response.choices:
                return "No response generated from the LLM."

            response_text = response.choices[0].message.content.strip()

            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="openai_api",
                    data={
                        "operation": "generate_response",
                        "response_length": len(response_text),
                        "usage": (
                            response.usage.model_dump() if response.usage else None
                        ),
                        "model": model_name,
                    },
                )
            )

            return response_text

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="openai_api",
                    data={"operation": "generate_response", "error": str(e)},
                )
            )
            return f"Error generating response: {str(e)}"

    def query(
        self, question: str, top_k: int = 3, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response

        Args:
            question: User question
            top_k: Number of documents to retrieve
            run_id: Optional run ID for tracing (auto-generated if not provided)

        Returns:
            Dictionary containing response and retrieved documents
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"

        # Reset traces for this query
        self.traces = []

        self.traces.append(
            TraceEvent(
                event_type="query_start",
                component="rag_system",
                data={
                    "run_id": run_id,
                    "question": question,
                    "question_length": len(question),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        try:
            retrieved_docs = self.retrieve_documents(question, top_k)
            response = self.generate_response(
                question, top_k, retrieved_docs=retrieved_docs
            )

            # Extract usage from traces
            usage = None
            for trace in self.traces:
                if trace.event_type == "llm_response" and "usage" in trace.data:
                    usage = trace.data["usage"]
                    break

            result = {
                "answer": response,
                "run_id": run_id,
                "retrieved_docs": retrieved_docs,
                "usage": usage,
            }

            self.traces.append(
                TraceEvent(
                    event_type="query_complete",
                    component="rag_system",
                    data={
                        "run_id": run_id,
                        "success": True,
                        "response_length": len(response),
                        "num_retrieved": len(retrieved_docs),
                        "total_tokens": usage.get("total_tokens") if usage else None,
                    },
                )
            )

            logs_path = self.export_traces_to_log(run_id, question, result)
            return {
                "answer": response,
                "run_id": run_id,
                "retrieved_docs": retrieved_docs,
                "usage": usage,
                "logs": logs_path,
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="rag_system",
                    data={"run_id": run_id, "operation": "query", "error": str(e)},
                )
            )

            # Return error result
            logs_path = self.export_traces_to_log(run_id, question, None)
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
                "logs": logs_path,
            }

    def export_traces_to_log(
        self,
        run_id: str,
        query: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Export traces to a log file with run_id"""
        timestamp = datetime.now().isoformat()
        log_filename = (
            f"rag_run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        )
        log_filepath = os.path.join(self.logdir, log_filename)

        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": query,
            "result": result,
            "num_documents": len(self.documents),
            "traces": [asdict(trace) for trace in self.traces],
        }

        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"RAG traces exported to: {log_filepath}")
        return log_filepath


def default_rag_client(
    llm_client,
    logdir: str = "logs",
    use_caipe: bool = True,
    model_name: Optional[str] = None,
) -> BaseRAG:
    """
    Create a default RAG client with OpenAI LLM and optional retriever.

    Args:
        llm_client: LLM client
        logdir: Directory for trace logs
        use_caipe: Whether to use CaipeRetriever (defaults to True)
        model_name: Optional name of the model to use
    Returns:
        BaseRAG instance
    """
    if use_caipe:
        retriever = CaipeRetriever()
    else:
        retriever = SimpleKeywordRetriever()

    resolved_model_name = model_name or settings.openai_model_name
    client = BaseRAG(
        llm_client=llm_client,
        model_name=resolved_model_name,
        retriever=retriever,
        logdir=logdir,
    )

    if not use_caipe:
        client.add_documents(DOCUMENTS)  # Add default documents for local retriever

    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Load configuration from settings
    api_key = settings.openai_api_key
    base_url = settings.openai_endpoint
    model_name = settings.openai_model_name

    # Initialize RAG system with custom LLM configuration
    llm = OpenAI(api_key=api_key, base_url=base_url)
    retriever = CaipeRetriever()
    rag_client = BaseRAG(
        llm_client=llm, model_name=model_name, retriever=retriever, logdir="logs"
    )

    # Run query with tracing
    query = "What is the caipe project?"
    logger.info(f"Query: {query}")
    response = rag_client.query(query, top_k=3)

    logger.info(f"Response: {response['answer']}")
    logger.info(f"Logs exported to: {response['logs']}")
