"""
Paleontology Research Assistant -- Persistent Memory
V2: Replaces sentence-transformers (neural embeddings, ~300 MB) with TF-IDF
    + cosine similarity via scikit-learn (~1 MB). ChromaDB is kept for
    persistent document storage; similarity is computed locally in retrieve().
    Documents are stored with a dummy 1-D embedding to satisfy ChromaDB's
    storage requirement without triggering any model download.
    Collection renamed to 'paleo_research_v2' to avoid dimension-mismatch
    errors against existing v1 collections (384-D sentence-transformer vectors).
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_DB_PATH = str(Path(__file__).parent / "memory_db")
_COLLECTION = "paleo_research_v2"  # v1 used 384-D sentence-transformer embeddings


class PaleoMemory:
    """
    Persistent research session store backed by ChromaDB with TF-IDF retrieval.

    ChromaDB provides durable on-disk storage; TfidfVectorizer + cosine_similarity
    handle similarity ranking at query time. No neural model is loaded.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=_DB_PATH)
        # embedding_function=None: embeddings are supplied manually (dummy values).
        # All real similarity work is done by TF-IDF in retrieve().
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=None,
        )
        logger.info(
            "PaleoMemory initialised: %d session(s) in store.", self._collection.count()
        )

    def save(self, query: str, response: str) -> None:
        """Store the full annotated response. Dummy embedding satisfies ChromaDB storage."""
        try:
            self._collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[[1.0]],
                documents=[response],
                metadatas=[{
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                }],
            )
            logger.info(
                "Memory: saved session for %r (%d total).",
                query[:60],
                self._collection.count(),
            )
        except Exception as exc:
            logger.warning("Memory save failed: %s", exc)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Return up to k prior sessions most similar to query.

        Fetches all stored session queries from ChromaDB, fits a TfidfVectorizer
        over them plus the new query, and ranks by cosine similarity. The new
        query is the last row of the TF-IDF matrix.

        Each returned entry is {"query": str, "response": str, "timestamp": str}.
        Returns an empty list if the store is empty or an error occurs.
        """
        try:
            total = self._collection.count()
            if total == 0:
                return []

            result = self._collection.get(include=["documents", "metadatas"])
            stored_queries = [m["query"] for m in result["metadatas"]]
            stored_responses = result["documents"]

            # Append the new query as the last item; its vector is tfidf_matrix[-1].
            corpus = stored_queries + [query]
            tfidf_matrix = TfidfVectorizer().fit_transform(corpus)
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

            top_indices = similarities.argsort()[::-1][:min(k, total)]
            sessions = [
                {
                    "query": stored_queries[i],
                    "response": stored_responses[i],
                    "timestamp": result["metadatas"][i].get("timestamp", ""),
                }
                for i in top_indices
            ]
            logger.info(
                "Memory: retrieved %d session(s) for %r.", len(sessions), query[:60]
            )
            return sessions
        except Exception as exc:
            logger.warning("Memory retrieve failed: %s", exc)
            return []

    def count(self) -> int:
        """Return the number of stored sessions."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all stored sessions and recreate the empty collection."""
        try:
            self._client.delete_collection(_COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION,
                embedding_function=None,
            )
            logger.info("Memory: cleared all sessions.")
        except Exception as exc:
            logger.warning("Memory clear failed: %s", exc)


# Module-level singleton — imported by agent.py and app.py.
# Loaded once per process; Streamlit's module cache keeps it alive across reruns.
memory = PaleoMemory()
