"""Exact search over MemPalace embeddings with optional persisted sidecar reuse."""

from __future__ import annotations

import logging
from typing import Any

from .exact_index import PersistedExactIndex
from .exact_search import query_corpus_with_torch


logger = logging.getLogger(__name__)

_FETCH_PAGE_SIZE = 1000


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value)


def _fetch_embeddings(collection: Any, *, where: dict | None, where_document: dict | None) -> dict[str, Any]:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    offset = 0

    while True:
        kwargs = {
            "limit": _FETCH_PAGE_SIZE,
            "offset": offset,
            "include": ["documents", "metadatas", "embeddings"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        batch = collection.get(**kwargs)
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

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }


def query_collection_with_torch(
    collection: Any,
    embedding_function: Any,
    *,
    query_texts: list[str],
    n_results: int = 5,
    where: dict | None = None,
    where_document: dict | None = None,
    device: str = "auto",
    tile_size: int = 32768,
    kernel_backend: str | None = None,
    palace_path: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Search MemPalace exactly, reusing a persisted sidecar when available."""

    query_texts = list(query_texts or [])
    if palace_path:
        try:
            return PersistedExactIndex(palace_path).query(
                collection,
                embedding_function,
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                device=device,
                tile_size=tile_size,
                kernel_backend=kernel_backend,
            )
        except Exception:
            logger.exception("Persisted exact index query failed; falling back to live exact fetch")

    corpus = _fetch_embeddings(collection, where=where, where_document=where_document)
    return query_corpus_with_torch(
        corpus,
        embedding_function,
        query_texts=query_texts,
        n_results=n_results,
        device=device,
        tile_size=tile_size,
        kernel_backend=kernel_backend,
    )
