# Chroma_Lib.py
#
from __future__ import annotations
"""Light‑weight ChromaDB helper for single‑user, local‑first apps.

Key simplifications vs. the original:
• no per‑user path hierarchy – caller passes explicit `storage_path`.
• no global metrics, Prometheus, rate‑limit or unload timers.
• model selection is explicit: every public call takes `model_id_override`; if
  absent and no default was configured, a `ValueError` is raised – no fall‑backs.
• minimal locking (only for Chroma client operations that mutate the store).
• designed to be IDE‑indexable and <500 lines so it ships nicely inside a
  Textual executable.
"""
#
# Imports
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from itertools import islice
import threading
#
# Third-Party Libraries
import numpy as np
import chromadb
from chromadb import Settings
from chromadb.api.models import Collection
from chromadb.api.types import QueryResult
#
# Local Imports
#
########################################################################################################################
#
# ---- tiny public type -------------------------------------------------------
ChromaInclude = Literal["documents", "metadatas", "distances", "embeddings"]


class ChromaDBManager:
    """A *thin* wrapper around a single embedded Chroma instance."""

    def __init__(
        self,
        storage_path: Path | str,
        embedding_config: Dict[str, Any],  # validated by caller
        default_model_id: str | None = None,
    ) -> None:
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.embedding_config = embedding_config
        self.default_model_id = default_model_id

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self.client = chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    # --------------------------------------------------------------------- util
    @staticmethod
    def _batched(iterable, n):
        it = iter(iterable)
        while (batch := list(islice(it, n))):
            yield batch

    def _resolve_model_id(self, override: str | None) -> str:
        model_id = override or self.default_model_id
        if not model_id:
            raise ValueError(
                "No embedding model selected. Pass `model_id_override` or set a default.")
        return model_id

    # ---------------------------------------------------------- collection I/O
    def get_or_create_collection(self, name: str) -> Collection:
        with self._lock:
            return self.client.get_or_create_collection(name=name)

    # ------------------------------------------------------------ write helper
    def store(
        self,
        collection: str,
        texts: List[str],
        embeddings: List[List[float]] | np.ndarray,
        ids: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
    ) -> None:
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        if not (len(texts) == len(embeddings) == len(ids)):
            raise ValueError("texts / embeddings / ids length mismatch")

        col = self.get_or_create_collection(collection)
        with self._lock:
            col.upsert(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )

    # ----------------------------------------------------------- vector search
    def vector_search(
        self,
        query: str,
        collection: str,
        k: int,
        *,
        model_id_override: Optional[str] = None,
        include: Optional[List[ChromaInclude]] = None,
        where: Optional[Dict[str, Any]] = None,
        create_embedding_fn,  # injected dep
    ) -> List[Dict[str, Any]]:
        model_id = self._resolve_model_id(model_id_override)
        embedding = create_embedding_fn(query, model_id)

        col = self.get_or_create_collection(collection)
        result: QueryResult = col.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=include or ["documents", "distances"]
        )

        docs = []
        if result["documents"] and result["documents"][0]:
            for i in range(len(result["documents"][0])):
                docs.append({
                    "id": result["ids"][0][i],
                    "content": result["documents"][0][i],
                    "distance": result["distances"][0][i],
                    "metadata": (result["metadatas"][0][i] if result.get("metadatas") else None),
                })
        return docs

#
# End of Chroma_Lib.py
########################################################################################################################
