"""
Retrieval service.

Loads the persisted Chroma vector store (built in notebooks/rag_pipeline.ipynb)
ONCE at application startup, and exposes a function to retrieve the top-k
most relevant chunks for a given question.
"""
import logging
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

from app.core.config import settings

logger = logging.getLogger("rag_assistant")


class RetrievalService:
    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def load(self):
        """Load the persisted vector store. Call once at startup."""
        logger.info("Loading vector store from %s", settings.VECTOR_STORE_DIR)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        self._client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
        self._collection = self._client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
        logger.info(
            "Vector store loaded. Collection '%s' has %d chunks.",
            settings.COLLECTION_NAME,
            self._collection.count(),
        )

    def retrieve(self, question: str, top_k: int = None) -> List[Dict]:
        """Return top_k relevant chunks as a list of dicts:
        {document, chunk_id, text, distance}
        """
        if self._collection is None:
            raise RuntimeError("RetrievalService.load() must be called before retrieve().")

        k = top_k or settings.TOP_K
        results = self._collection.query(query_texts=[question], n_results=k)

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, meta, chunk_id, dist in zip(docs, metas, ids, distances):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "document": (meta or {}).get("source", "unknown"),
                    "distance": dist,
                }
            )
        return chunks


# Singleton instance used across the app
retrieval_service = RetrievalService()
