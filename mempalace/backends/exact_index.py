"""Persisted exact-search sidecar built from the Chroma source of truth."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .exact_search import query_corpus_with_torch


_FETCH_PAGE_SIZE = 1000
_INDEX_FILENAME = "mempalace_exact_index.npz"
_MANIFEST_FILENAME = "mempalace_exact_index_manifest.json"
_STALE_FILENAME = "mempalace_exact_index.stale"
_INDEX_VERSION = 1


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value)


def mark_exact_index_dirty(palace_path: str) -> None:
    os.makedirs(palace_path, exist_ok=True)
    stale_path = os.path.join(palace_path, _STALE_FILENAME)
    with open(stale_path, "w", encoding="utf-8") as handle:
        handle.write("stale\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _write_npz_atomic(path: str, **arrays: Any) -> None:
    temp_path = path + ".tmp"
    with open(temp_path, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _metadata_matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_metadata_matches(metadata, clause) for clause in where["$and"])
    return all(metadata.get(key) == value for key, value in where.items())


def _document_matches(document: str, where_document: dict[str, Any] | None) -> bool:
    if not where_document:
        return True
    needle = where_document.get("$contains")
    if needle is not None:
        return str(needle) in document
    return True


class PersistedExactIndex:
    """Materialized exact-search view of a MemPalace Chroma collection."""

    def __init__(self, palace_path: str):
        self.palace_path = os.path.abspath(palace_path)
        self.index_path = os.path.join(self.palace_path, _INDEX_FILENAME)
        self.manifest_path = os.path.join(self.palace_path, _MANIFEST_FILENAME)
        self.stale_path = os.path.join(self.palace_path, _STALE_FILENAME)
        self.source_db_path = os.path.join(self.palace_path, "chroma.sqlite3")
        self._loaded_corpus: dict[str, Any] | None = None
        self._loaded_token: tuple[int, int] | None = None

    def _source_db_mtime_ns(self) -> int:
        try:
            return int(os.stat(self.source_db_path).st_mtime_ns)
        except OSError:
            return 0

    def _index_token(self) -> tuple[int, int] | None:
        try:
            return (
                int(os.stat(self.index_path).st_mtime_ns),
                int(os.stat(self.manifest_path).st_mtime_ns),
            )
        except OSError:
            return None

    def _read_manifest(self) -> dict[str, Any] | None:
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def _fetch_corpus(self, collection: Any) -> dict[str, Any]:
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []
        offset = 0

        while True:
            batch = collection.get(
                limit=_FETCH_PAGE_SIZE,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            batch_ids = _safe_list(batch.get("ids"))
            if not batch_ids:
                break

            batch_documents = _safe_list(batch.get("documents"))
            batch_metadatas = _safe_list(batch.get("metadatas"))
            batch_embeddings = _safe_list(batch.get("embeddings"))

            ids.extend(batch_ids)
            documents.extend(batch_documents)
            metadatas.extend(batch_metadatas or [{} for _ in batch_ids])
            embeddings.extend(batch_embeddings)

            offset += len(batch_ids)
            if len(batch_ids) < _FETCH_PAGE_SIZE:
                break

        order = sorted(range(len(ids)), key=lambda idx: ids[idx])
        sorted_ids = [ids[idx] for idx in order]
        sorted_documents = [documents[idx] for idx in order]
        sorted_metadatas = [metadatas[idx] for idx in order]
        if embeddings:
            sorted_embeddings = np.asarray([embeddings[idx] for idx in order], dtype=np.float32)
        else:
            sorted_embeddings = np.empty((0, 0), dtype=np.float32)

        return {
            "ids": sorted_ids,
            "documents": sorted_documents,
            "metadatas": sorted_metadatas,
            "embeddings": sorted_embeddings,
        }

    def _write_corpus(self, corpus: dict[str, Any]) -> None:
        metadata_json = [
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            for metadata in corpus["metadatas"]
        ]
        _write_npz_atomic(
            self.index_path,
            embeddings=np.asarray(corpus["embeddings"], dtype=np.float32),
            ids=np.asarray(corpus["ids"], dtype=str),
            documents=np.asarray(corpus["documents"], dtype=str),
            metadatas_json=np.asarray(metadata_json, dtype=str),
        )
        _write_json_atomic(
            self.manifest_path,
            {
                "version": _INDEX_VERSION,
                "count": len(corpus["ids"]),
                "source_db_mtime_ns": self._source_db_mtime_ns(),
            },
        )
        try:
            os.remove(self.stale_path)
        except FileNotFoundError:
            pass

    def rebuild(self, collection: Any) -> dict[str, Any]:
        corpus = self._fetch_corpus(collection)
        self._write_corpus(corpus)
        self._loaded_corpus = corpus
        self._loaded_token = self._index_token()
        return corpus

    def _is_current(self, collection: Any) -> bool:
        if os.path.exists(self.stale_path):
            return False
        token = self._index_token()
        manifest = self._read_manifest()
        if token is None or manifest is None:
            return False
        if manifest.get("version") != _INDEX_VERSION:
            return False
        if int(manifest.get("source_db_mtime_ns", -1)) != self._source_db_mtime_ns():
            return False
        try:
            return int(manifest.get("count", -1)) == int(collection.count())
        except Exception:
            return True

    def load(self) -> dict[str, Any]:
        token = self._index_token()
        if token is not None and self._loaded_token == token and self._loaded_corpus is not None:
            return self._loaded_corpus

        with np.load(self.index_path, allow_pickle=False) as payload:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            ids = [str(value) for value in payload["ids"].tolist()]
            documents = [str(value) for value in payload["documents"].tolist()]
            metadata_json = [str(value) for value in payload["metadatas_json"].tolist()]

        corpus = {
            "ids": ids,
            "documents": documents,
            "metadatas": [json.loads(value) for value in metadata_json],
            "embeddings": embeddings,
        }
        self._loaded_corpus = corpus
        self._loaded_token = token
        return corpus

    def ensure_current(self, collection: Any) -> dict[str, Any]:
        if not self._is_current(collection):
            return self.rebuild(collection)
        return self.load()

    def _filtered_corpus(
        self,
        corpus: dict[str, Any],
        *,
        where: dict[str, Any] | None,
        where_document: dict[str, Any] | None,
    ) -> dict[str, Any]:
        selected_indices = [
            idx
            for idx, (metadata, document) in enumerate(zip(corpus["metadatas"], corpus["documents"]))
            if _metadata_matches(metadata, where) and _document_matches(document, where_document)
        ]
        if not selected_indices:
            embedding_width = 0
            if getattr(corpus["embeddings"], "ndim", 0) == 2:
                embedding_width = int(corpus["embeddings"].shape[1])
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": np.empty((0, embedding_width), dtype=np.float32),
            }

        return {
            "ids": [corpus["ids"][idx] for idx in selected_indices],
            "documents": [corpus["documents"][idx] for idx in selected_indices],
            "metadatas": [corpus["metadatas"][idx] for idx in selected_indices],
            "embeddings": np.asarray(corpus["embeddings"][selected_indices], dtype=np.float32),
        }

    def query(
        self,
        collection: Any,
        embedding_function: Any,
        *,
        query_texts: list[str],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        device: str = "auto",
        tile_size: int = 32768,
        kernel_backend: str | None = None,
    ) -> dict[str, Any]:
        corpus = self.ensure_current(collection)
        filtered = self._filtered_corpus(corpus, where=where, where_document=where_document)
        return query_corpus_with_torch(
            filtered,
            embedding_function,
            query_texts=query_texts,
            n_results=n_results,
            device=device,
            tile_size=tile_size,
            kernel_backend=kernel_backend,
        )
