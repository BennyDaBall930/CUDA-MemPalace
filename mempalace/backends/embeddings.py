"""Embedding-function helpers for Chroma-backed MemPalace collections."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from chromadb.api.types import DefaultEmbeddingFunction, Documents, EmbeddingFunction, Embeddings


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 32


def _resolve_torch_device(requested_device: Optional[str]) -> str:
    device = (requested_device or "auto").strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"

    if device in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested for embeddings but unavailable; falling back to CPU")
        return "cpu"

    return device


class TransformersEmbeddingFunction(EmbeddingFunction[Documents]):
    """Local transformers embedder with CUDA support via torch."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.requested_device = device or "auto"
        self.device = _resolve_torch_device(self.requested_device)
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self._tokenizer = None
        self._model = None
        self._torch = None

    @staticmethod
    def name() -> str:
        return "transformers"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "TransformersEmbeddingFunction":
        TransformersEmbeddingFunction.validate_config(config)
        return TransformersEmbeddingFunction(
            model_name=config.get("model_name", DEFAULT_MODEL_NAME),
            device=config.get("device", "auto"),
            max_length=config.get("max_length", DEFAULT_MAX_LENGTH),
            batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
        )

    def get_config(self) -> Dict[str, Any]:
        # Device stays runtime-only so a CPU-built palace can still be queried on CUDA later.
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

    def max_tokens(self) -> int:
        return self.max_length

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        if not isinstance(config.get("model_name", DEFAULT_MODEL_NAME), str):
            raise ValueError("model_name must be a string")

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ValueError(
                "transformers embedding backend requires torch and transformers to be installed"
            ) from exc

        self.device = _resolve_torch_device(self.requested_device)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []

        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None

        embeddings = []
        with self._torch.inference_mode():
            for start in range(0, len(input), self.batch_size):
                batch = list(input[start : start + self.batch_size])
                tokens = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self._model(**tokens)
                hidden = outputs.last_hidden_state
                attention_mask = tokens["attention_mask"].unsqueeze(-1).type_as(hidden)
                pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(
                    min=1.0
                )
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.extend(pooled.detach().cpu().to(self._torch.float32).tolist())

        return embeddings


def build_embedding_function(config: Any) -> EmbeddingFunction[Documents]:
    """Return the configured embedding function while keeping CPU defaults intact."""

    backend = (getattr(config, "embedding_backend", "default") or "default").strip().lower()
    model_name = getattr(config, "embedding_model", DEFAULT_MODEL_NAME)
    device = getattr(config, "embedding_device", "auto")

    if backend in {"default", "chroma", "onnx"}:
        return DefaultEmbeddingFunction()

    if backend in {"transformers", "cuda"}:
        return TransformersEmbeddingFunction(model_name=model_name, device=device)

    if backend == "auto":
        resolved_device = _resolve_torch_device(device)
        if resolved_device.startswith("cuda"):
            return TransformersEmbeddingFunction(model_name=model_name, device=resolved_device)
        return DefaultEmbeddingFunction()

    raise ValueError(f"Unknown embedding backend: {backend}")
