"""
Paleontology Research Assistant -- Persistent Memory
Stores completed research sessions in a local ChromaDB vector database and
retrieves semantically similar prior sessions to inject as context into new
queries. The database persists across app restarts at ./memory_db.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DB_PATH = str(Path(__file__).parent / "memory_db")
_COLLECTION = "paleo_research"
_EMBED_MODEL = "all-MiniLM-L6-v2"


class PaleoMemory:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Embeddings are generated with a local sentence-transformers model
    (all-MiniLM-L6-v2, ~80 MB) so no external embedding API is needed.
    The model and client are loaded once at instantiation and reused.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=_DB_PATH)
        self._collection = self._client.get_or_create_collection(_COLLECTION)
        self._model = SentenceTransformer(_EMBED_MODEL)
        logger.info(
            "PaleoMemory initialised: %d session(s) in store.", self._collection.count()
        )

    def save(self, query: str, response: str) -> None:
        """Embed the query and store the full response as a document."""
        try:
            embedding = self._model.encode(query).tolist()
            self._collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
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
        Return up to k prior sessions most semantically similar to query.

        Each entry is {"query": str, "response": str, "timestamp": str}.
        Returns an empty list if the store is empty or the query fails.
        """
        try:
            total = self._collection.count()
            if total == 0:
                return []
            results = self._collection.query(
                query_embeddings=[self._model.encode(query).tolist()],
                n_results=min(k, total),
            )
            sessions = [
                {
                    "query": results["metadatas"][0][i]["query"],
                    "response": results["documents"][0][i],
                    "timestamp": results["metadatas"][0][i].get("timestamp", ""),
                }
                for i in range(len(results["ids"][0]))
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
            self._collection = self._client.get_or_create_collection(_COLLECTION)
            logger.info("Memory: cleared all sessions.")
        except Exception as exc:
            logger.warning("Memory clear failed: %s", exc)


# Module-level singleton — imported by agent.py and app.py.
# Loaded once per process; Streamlit's module cache keeps it alive across reruns.
memory = PaleoMemory()
