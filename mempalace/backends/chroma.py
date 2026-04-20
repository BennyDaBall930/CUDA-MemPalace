"""ChromaDB-backed MemPalace collection adapter."""

from __future__ import annotations

import logging
import os
import sqlite3

import chromadb

from .base import BaseCollection
from .exact_index import mark_exact_index_dirty
from .torch_cuda_search import query_collection_with_torch

logger = logging.getLogger(__name__)


def _fix_blob_seq_ids(palace_path: str):
    """Fix ChromaDB 0.6.x -> 1.5.x migration bug: BLOB seq_ids -> INTEGER.

    ChromaDB 0.6.x stored seq_id as big-endian 8-byte BLOBs. ChromaDB 1.5.x
    expects INTEGER. The auto-migration doesn't convert existing rows, causing
    the Rust compactor to crash with "mismatched types; Rust type u64 (as SQL
    type INTEGER) is not compatible with SQL type BLOB".

    Must run BEFORE PersistentClient is created (the compactor fires on init).
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return
    try:
        with sqlite3.connect(db_path) as conn:
            for table in ("embeddings", "max_seq_id"):
                try:
                    rows = conn.execute(
                        f"SELECT rowid, seq_id FROM {table} WHERE typeof(seq_id) = 'blob'"
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                if not rows:
                    continue
                updates = [(int.from_bytes(blob, byteorder="big"), rowid) for rowid, blob in rows]
                conn.executemany(f"UPDATE {table} SET seq_id = ? WHERE rowid = ?", updates)
                logger.info("Fixed %d BLOB seq_ids in %s", len(updates), table)
            conn.commit()
    except Exception:
        logger.exception("Could not fix BLOB seq_ids in %s", db_path)


def _resolve_search_backend(search_backend: str | None, search_device: str | None) -> str:
    backend = (search_backend or "auto").strip().lower()
    if backend == "exact":
        backend = "torch"
    if backend != "auto":
        return backend

    try:
        import torch
    except ImportError:
        return "chroma"

    device = (search_device or "auto").strip().lower()
    if device == "cpu":
        return "chroma"
    if device.startswith("cuda"):
        return "torch" if torch.cuda.is_available() else "chroma"
    return "torch" if torch.cuda.is_available() else "chroma"


def _is_embedding_conflict_error(exc: Exception) -> bool:
    return "Embedding function conflict" in str(exc)


class ChromaCollection(BaseCollection):
    """Thin adapter over a ChromaDB collection."""

    def __init__(
        self,
        collection,
        *,
        palace_path: str | None = None,
        embedding_function=None,
        search_backend: str = "chroma",
        search_device: str = "auto",
        search_tile_size: int = 32768,
        exact_kernel_backend: str | None = None,
    ):
        self._collection = collection
        self._palace_path = palace_path
        self._embedding_function = embedding_function
        self._search_backend = _resolve_search_backend(search_backend, search_device)
        self._search_device = search_device
        self._search_tile_size = max(1, int(search_tile_size))
        self._exact_kernel_backend = exact_kernel_backend

    def add(self, *, documents, ids, metadatas=None):
        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)
        if self._palace_path:
            mark_exact_index_dirty(self._palace_path)

    def upsert(self, *, documents, ids, metadatas=None):
        self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
        if self._palace_path:
            mark_exact_index_dirty(self._palace_path)

    def update(self, **kwargs):
        self._collection.update(**kwargs)
        if self._palace_path:
            mark_exact_index_dirty(self._palace_path)

    def query(self, **kwargs):
        if (
            self._search_backend == "torch"
            and self._embedding_function is not None
            and kwargs.get("query_texts")
        ):
            try:
                return query_collection_with_torch(
                    self._collection,
                    self._embedding_function,
                    palace_path=self._palace_path,
                    device=self._search_device,
                    tile_size=self._search_tile_size,
                    kernel_backend=self._exact_kernel_backend,
                    **kwargs,
                )
            except Exception:
                logger.exception("Torch search backend failed; falling back to Chroma query")
        return self._collection.query(**kwargs)

    def get(self, **kwargs):
        return self._collection.get(**kwargs)

    def delete(self, **kwargs):
        self._collection.delete(**kwargs)
        if self._palace_path:
            mark_exact_index_dirty(self._palace_path)

    def count(self):
        return self._collection.count()


class ChromaBackend:
    """Factory for MemPalace's default ChromaDB backend."""

    def __init__(
        self,
        *,
        embedding_function=None,
        search_backend: str = "chroma",
        search_device: str = "auto",
        search_tile_size: int = 32768,
        exact_kernel_backend: str | None = None,
    ):
        # Per-instance client cache: palace_path -> chromadb.PersistentClient
        self._clients: dict = {}
        self._embedding_function = embedding_function
        self._search_backend = search_backend
        self._search_device = search_device
        self._search_tile_size = search_tile_size
        self._exact_kernel_backend = exact_kernel_backend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self, palace_path: str):
        """Return a cached PersistentClient for *palace_path*, creating one if needed."""
        if palace_path not in self._clients:
            _fix_blob_seq_ids(palace_path)
            self._clients[palace_path] = chromadb.PersistentClient(path=palace_path)
        return self._clients[palace_path]

    # ------------------------------------------------------------------
    # Public static helpers (for callers that manage their own caching)
    # ------------------------------------------------------------------

    @staticmethod
    def make_client(palace_path: str):
        """Create and return a fresh PersistentClient (fix BLOB seq_ids first).

        Intended for long-lived callers (e.g. mcp_server) that keep their own
        inode/mtime-based client cache.
        """
        _fix_blob_seq_ids(palace_path)
        return chromadb.PersistentClient(path=palace_path)

    @staticmethod
    def backend_version() -> str:
        """Return the installed chromadb package version string."""
        return chromadb.__version__

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def get_collection(self, palace_path: str, collection_name: str, create: bool = False):
        if not create and not os.path.isdir(palace_path):
            raise FileNotFoundError(palace_path)

        if create:
            os.makedirs(palace_path, exist_ok=True)
            try:
                os.chmod(palace_path, 0o700)
            except (OSError, NotImplementedError):
                pass

        client = self._client(palace_path)
        embedding_function = self._embedding_function
        try:
            if create:
                collection = client.get_or_create_collection(
                    collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=embedding_function,
                )
            else:
                collection = client.get_collection(
                    collection_name,
                    embedding_function=embedding_function,
                )
        except ValueError as exc:
            if not _is_embedding_conflict_error(exc):
                raise
            logger.info(
                "Embedding function conflict for %s; reopening collection without overriding persisted embedder",
                collection_name,
            )
            collection = client.get_collection(collection_name)
            embedding_function = None
        return ChromaCollection(
            collection,
            palace_path=palace_path,
            embedding_function=embedding_function,
            search_backend=self._search_backend,
            search_device=self._search_device,
            search_tile_size=self._search_tile_size,
            exact_kernel_backend=self._exact_kernel_backend,
        )

    def get_or_create_collection(
        self, palace_path: str, collection_name: str
    ) -> "ChromaCollection":
        """Shorthand for get_collection(..., create=True)."""
        return self.get_collection(palace_path, collection_name, create=True)

    def delete_collection(self, palace_path: str, collection_name: str) -> None:
        """Delete *collection_name* from the palace at *palace_path*."""
        self._client(palace_path).delete_collection(collection_name)

    def create_collection(
        self, palace_path: str, collection_name: str, hnsw_space: str = "cosine"
    ) -> "ChromaCollection":
        """Create (not get-or-create) *collection_name* with cosine HNSW space."""
        collection = self._client(palace_path).create_collection(
            collection_name,
            metadata={"hnsw:space": hnsw_space},
            embedding_function=self._embedding_function,
        )
        return ChromaCollection(
            collection,
            palace_path=palace_path,
            embedding_function=self._embedding_function,
            search_backend=self._search_backend,
            search_device=self._search_device,
            search_tile_size=self._search_tile_size,
            exact_kernel_backend=self._exact_kernel_backend,
        )
