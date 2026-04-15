"""Optional custom CUDA scoring/top-k kernel hook for exact MemPalace search.

The proven torch exact path remains the semantic source of truth.  This module
only provides a narrow acceleration seam: an optional extension may compute the
full score vector or deterministic top-k for one query.  If the extension is
missing, disabled, or incomplete, callers get ``None`` and fall back to the
proven torch exact path.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any


logger = logging.getLogger(__name__)

DEFAULT_KERNEL_BACKEND = "auto"
_EXTENSION_MODULE = "mempalace.backends._cuda_exact_kernel"
_DISABLED_WARNED = False
_UNAVAILABLE_WARNED = False


def resolve_kernel_backend(requested_backend: str | None = None) -> str:
    backend = (
        requested_backend
        or os.environ.get("MEMPALACE_EXACT_KERNEL_BACKEND")
        or DEFAULT_KERNEL_BACKEND
    )
    return backend.strip().lower()


def _load_extension() -> Any | None:
    if sys.platform == "win32":
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, "bin")
            if os.path.isdir(cuda_bin):
                try:
                    os.add_dll_directory(cuda_bin)
                except (AttributeError, OSError):
                    pass
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    try:
        return importlib.import_module(_EXTENSION_MODULE)
    except ImportError:
        return None


def custom_kernel_available() -> bool:
    extension = _load_extension()
    return extension is not None and hasattr(extension, "score_vector")


def custom_topk_available() -> bool:
    extension = _load_extension()
    return extension is not None and hasattr(extension, "topk")


def score_vector_with_custom_kernel(
    corpus_tensor: Any,
    query_tensor: Any,
    *,
    backend: str | None = None,
    tile_size: int = 32768,
) -> Any | None:
    """Return a score tensor from a custom scorer, or None for safe fallback."""

    global _DISABLED_WARNED, _UNAVAILABLE_WARNED
    resolved = resolve_kernel_backend(backend)
    if resolved in {"", "auto", "torch", "pytorch", "none", "off", "disabled"}:
        if resolved not in {"auto", ""}:
            return None

    if resolved not in {"auto", "", "custom", "cuda", "extension"}:
        if not _DISABLED_WARNED:
            logger.warning("Unknown exact kernel backend %r; falling back to torch", resolved)
            _DISABLED_WARNED = True
        return None

    extension = _load_extension()
    if extension is None or not hasattr(extension, "score_vector"):
        if resolved != "auto" and not _UNAVAILABLE_WARNED:
            logger.warning("Custom CUDA exact kernel requested but extension is unavailable")
            _UNAVAILABLE_WARNED = True
        return None

    return extension.score_vector(corpus_tensor, query_tensor, int(tile_size))


def topk_with_custom_kernel(
    corpus_tensor: Any,
    query_tensor: Any,
    id_ranks_tensor: Any,
    *,
    top_k: int,
    backend: str | None = None,
    tile_size: int = 32768,
) -> tuple[Any, Any] | None:
    """Return ``(scores, indices)`` from a custom top-k kernel, or ``None``.

    ``id_ranks_tensor`` is a CUDA int64 vector produced by Python from
    lexicographic MemPalace IDs.  The compiled kernel uses it as the exact
    deterministic tie-breaker, preserving the reference order of
    ``(-score, id)`` without teaching CUDA about strings.
    """

    global _DISABLED_WARNED, _UNAVAILABLE_WARNED
    resolved = resolve_kernel_backend(backend)
    if resolved in {"", "auto", "torch", "pytorch", "none", "off", "disabled"}:
        if resolved not in {"auto", ""}:
            return None

    if resolved not in {"auto", "", "custom", "cuda", "extension"}:
        if not _DISABLED_WARNED:
            logger.warning("Unknown exact kernel backend %r; falling back to torch", resolved)
            _DISABLED_WARNED = True
        return None

    extension = _load_extension()
    if extension is None or not hasattr(extension, "topk"):
        if resolved != "auto" and not _UNAVAILABLE_WARNED:
            logger.warning("Custom CUDA top-k requested but extension is unavailable")
            _UNAVAILABLE_WARNED = True
        return None

    scores, indices = extension.topk(
        corpus_tensor,
        query_tensor,
        id_ranks_tensor,
        int(top_k),
        int(tile_size),
    )
    return scores, indices
