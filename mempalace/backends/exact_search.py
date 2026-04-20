"""Helpers for deterministic exact search over persisted embedding corpora."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np

from .cuda_exact_kernel import score_vector_with_custom_kernel, topk_with_custom_kernel


logger = logging.getLogger(__name__)


def empty_query_result(query_count: int) -> dict[str, Any]:
    return {
        "ids": [[] for _ in range(query_count)],
        "documents": [[] for _ in range(query_count)],
        "metadatas": [[] for _ in range(query_count)],
        "distances": [[] for _ in range(query_count)],
    }


def resolve_search_device(requested_device: str | None) -> str:
    device = (requested_device or "auto").strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"

    if device in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested for exact search but unavailable; falling back to CPU")
        return "cpu"

    return device


def normalize_numpy_rows(rows: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.size == 0:
        return matrix.reshape(0, 0)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1e-12
    return matrix / norms


def _build_id_rank_vector(ids: Sequence[str]) -> list[int]:
    """Return numeric ranks that match MemPalace's lexicographic ID tie-break."""

    rank_by_id = {row_id: rank for rank, row_id in enumerate(sorted(ids))}
    return [rank_by_id[row_id] for row_id in ids]


def query_corpus_with_torch(
    corpus: dict[str, Any],
    embedding_function: Any,
    *,
    query_texts: Iterable[str],
    n_results: int = 5,
    device: str = "auto",
    tile_size: int = 32768,
    kernel_backend: str | None = None,
) -> dict[str, Any]:
    """Exact cosine search over an in-memory corpus with deterministic ordering."""

    try:
        import torch
    except ImportError as exc:
        raise ValueError("torch search backend requires torch to be installed") from exc

    query_texts = list(query_texts or [])
    if not query_texts:
        return empty_query_result(0)

    top_k = max(1, int(n_results))
    resolved_device = resolve_search_device(device)

    ids = list(corpus.get("ids", []) or [])
    documents = list(corpus.get("documents", []) or [])
    metadatas = list(corpus.get("metadatas", []) or [])
    embeddings = corpus.get("embeddings", [])

    if not ids:
        return empty_query_result(len(query_texts))

    corpus_matrix = normalize_numpy_rows(embeddings)
    if corpus_matrix.size == 0 or corpus_matrix.shape[0] == 0:
        return empty_query_result(len(query_texts))

    query_matrix = normalize_numpy_rows(embedding_function(query_texts))
    if query_matrix.size == 0:
        return empty_query_result(len(query_texts))

    corpus_tensor = torch.from_numpy(corpus_matrix)
    query_tensor = torch.from_numpy(query_matrix)
    if resolved_device != "cpu":
        corpus_tensor = corpus_tensor.to(resolved_device)
        query_tensor = query_tensor.to(resolved_device)
        id_ranks_tensor = torch.tensor(
            _build_id_rank_vector(ids),
            dtype=torch.long,
            device=resolved_device,
        )
    else:
        id_ranks_tensor = None

    actual_tile_size = max(1, int(tile_size))
    result_ids: list[list[str]] = []
    result_documents: list[list[str]] = []
    result_metadatas: list[list[dict[str, Any]]] = []
    result_distances: list[list[float]] = []

    for query_row in query_tensor:
        custom_topk = None
        custom_scores = None
        if resolved_device != "cpu":
            custom_topk = topk_with_custom_kernel(
                corpus_tensor,
                query_row,
                id_ranks_tensor,
                top_k=top_k,
                backend=kernel_backend,
                tile_size=actual_tile_size,
            )
            if custom_topk is None:
                custom_scores = score_vector_with_custom_kernel(
                    corpus_tensor,
                    query_row,
                    backend=kernel_backend,
                    tile_size=actual_tile_size,
                )

        if custom_topk is not None:
            top_scores, top_indices = custom_topk
            ordered_indices = [
                int(index)
                for index in top_indices.detach().cpu().numpy().astype(np.int64, copy=False)
            ]
            ordered_scores = [
                float(score)
                for score in top_scores.detach().cpu().numpy().astype(np.float32, copy=False)
            ]
        elif custom_scores is not None:
            score_vector = custom_scores.detach().cpu().numpy().astype(np.float32, copy=False)
            ordered_indices = sorted(
                range(len(ids)),
                key=lambda idx: (-float(score_vector[idx]), ids[idx]),
            )[:top_k]
            ordered_scores = [float(score_vector[idx]) for idx in ordered_indices]
        else:
            score_vector = np.empty(int(corpus_tensor.shape[0]), dtype=np.float32)
            for start in range(0, int(corpus_tensor.shape[0]), actual_tile_size):
                stop = min(start + actual_tile_size, int(corpus_tensor.shape[0]))
                chunk = corpus_tensor[start:stop]
                chunk_scores = torch.matmul(chunk, query_row)
                if int(chunk_scores.shape[0]) == 0:
                    continue
                score_vector[start:stop] = chunk_scores.detach().cpu().numpy()
            ordered_indices = sorted(
                range(len(ids)),
                key=lambda idx: (-float(score_vector[idx]), ids[idx]),
            )[:top_k]
            ordered_scores = [float(score_vector[idx]) for idx in ordered_indices]
        result_ids.append([ids[idx] for idx in ordered_indices])
        result_documents.append([documents[idx] for idx in ordered_indices])
        result_metadatas.append([metadatas[idx] for idx in ordered_indices])
        result_distances.append([float(1.0 - score) for score in ordered_scores])

    return {
        "ids": result_ids,
        "documents": result_documents,
        "metadatas": result_metadatas,
        "distances": result_distances,
    }
