import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from mempalace.backends import cuda_exact_kernel
from mempalace.backends.chroma import ChromaBackend, ChromaCollection, _resolve_search_backend
from mempalace.backends.embeddings import TransformersEmbeddingFunction, build_embedding_function
from mempalace.backends.exact_search import query_corpus_with_torch
from mempalace.backends.torch_cuda_search import query_collection_with_torch


class _FakeSearchCollection:
    def __init__(self):
        self.query_calls = []
        self.get_calls = []
        self.rows = [
            {
                "id": "1",
                "document": "query target",
                "metadata": {"wing": "project", "room": "backend", "source_file": "a.md"},
                "embedding": [1.0, 0.0],
            },
            {
                "id": "2",
                "document": "middle",
                "metadata": {"wing": "project", "room": "backend", "source_file": "b.md"},
                "embedding": [0.6, 0.4],
            },
            {
                "id": "3",
                "document": "far away",
                "metadata": {"wing": "notes", "room": "planning", "source_file": "c.md"},
                "embedding": [0.0, 1.0],
            },
        ]

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"kind": "fallback"}

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        limit = kwargs.get("limit", len(self.rows))
        offset = kwargs.get("offset", 0)
        where = kwargs.get("where")
        rows = list(self.rows)
        if where:
            rows = [
                row
                for row in rows
                if all(row["metadata"].get(key) == value for key, value in where.items())
            ]
        batch_rows = rows[offset : offset + limit]
        return {
            "ids": [row["id"] for row in batch_rows],
            "documents": [row["document"] for row in batch_rows],
            "metadatas": [row["metadata"] for row in batch_rows],
            "embeddings": np.asarray(
                [row["embedding"] for row in batch_rows], dtype=np.float32
            ),
        }

    def add(self, **kwargs):
        return None

    def upsert(self, **kwargs):
        return None

    def update(self, **kwargs):
        return None

    def delete(self, **kwargs):
        return None

    def count(self):
        return len(self.rows)


class _FakeTieSearchCollection:
    def __init__(self):
        self.get_calls = []
        self.rows = [
            {"id": "b", "document": "doc b", "metadata": {}, "embedding": [1.0, 0.0]},
            {"id": "a", "document": "doc a", "metadata": {}, "embedding": [1.0, 0.0]},
            {"id": "c", "document": "doc c", "metadata": {}, "embedding": [1.0, 0.0]},
        ]

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        limit = kwargs.get("limit", len(self.rows))
        offset = kwargs.get("offset", 0)
        batch_rows = self.rows[offset : offset + limit]
        return {
            "ids": [row["id"] for row in batch_rows],
            "documents": [row["document"] for row in batch_rows],
            "metadatas": [row["metadata"] for row in batch_rows],
            "embeddings": np.asarray(
                [row["embedding"] for row in batch_rows], dtype=np.float32
            ),
        }

    def count(self):
        return len(self.rows)


def _embedding_for_text(text: str) -> list[float]:
    if (
        "query target" in text
        or "new exact" in text
        or "doc a" in text
        or "doc b" in text
        or "anything" in text
    ):
        return [1.0, 0.0]
    if "middle" in text:
        return [0.6, 0.4]
    return [0.0, 1.0]


class _FakeMutableSearchCollection:
    def __init__(self):
        self.query_calls = []
        self.get_calls = []
        self.rows = [
            {
                "id": "1",
                "document": "query target",
                "metadata": {"wing": "project", "room": "backend", "source_file": "a.md"},
                "embedding": _embedding_for_text("query target"),
            },
            {
                "id": "2",
                "document": "far away",
                "metadata": {"wing": "notes", "room": "planning", "source_file": "c.md"},
                "embedding": _embedding_for_text("far away"),
            },
        ]

    def _row(self, row_id, document, metadata):
        return {
            "id": row_id,
            "document": document,
            "metadata": metadata,
            "embedding": _embedding_for_text(document),
        }

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"kind": "fallback"}

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        limit = kwargs.get("limit", len(self.rows))
        offset = kwargs.get("offset", 0)
        rows = list(self.rows)
        if kwargs.get("ids"):
            requested = set(kwargs["ids"])
            rows = [row for row in rows if row["id"] in requested]
        batch_rows = rows[offset : offset + limit]
        return {
            "ids": [row["id"] for row in batch_rows],
            "documents": [row["document"] for row in batch_rows],
            "metadatas": [row["metadata"] for row in batch_rows],
            "embeddings": np.asarray(
                [row["embedding"] for row in batch_rows], dtype=np.float32
            ),
        }

    def add(self, **kwargs):
        for row_id, document, metadata in zip(
            kwargs.get("ids", []),
            kwargs.get("documents", []),
            kwargs.get("metadatas", []) or [{} for _ in kwargs.get("ids", [])],
        ):
            self.rows.append(self._row(row_id, document, metadata))

    def upsert(self, **kwargs):
        for row_id, document, metadata in zip(
            kwargs.get("ids", []),
            kwargs.get("documents", []),
            kwargs.get("metadatas", []) or [{} for _ in kwargs.get("ids", [])],
        ):
            self.rows = [row for row in self.rows if row["id"] != row_id]
            self.rows.append(self._row(row_id, document, metadata))

    def update(self, **kwargs):
        documents = kwargs.get("documents")
        metadatas = kwargs.get("metadatas")
        for index, row_id in enumerate(kwargs.get("ids", [])):
            for row in self.rows:
                if row["id"] != row_id:
                    continue
                if documents:
                    row["document"] = documents[index]
                    row["embedding"] = _embedding_for_text(documents[index])
                if metadatas:
                    row["metadata"] = metadatas[index]

    def delete(self, **kwargs):
        ids = set(kwargs.get("ids", []))
        self.rows = [row for row in self.rows if row["id"] not in ids]

    def count(self):
        return len(self.rows)


def test_build_embedding_function_default_config():
    cfg = SimpleNamespace(
        embedding_backend="auto",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",
    )
    embedding_function = build_embedding_function(cfg)
    assert embedding_function.__class__.__name__ == "DefaultEmbeddingFunction"


def test_build_embedding_function_transformers_config():
    cfg = SimpleNamespace(
        embedding_backend="transformers",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",
    )
    embedding_function = build_embedding_function(cfg)
    assert isinstance(embedding_function, TransformersEmbeddingFunction)
    assert embedding_function.device == "cpu"


def test_build_embedding_function_auto_prefers_transformers_when_cuda_available(monkeypatch):
    cfg = SimpleNamespace(
        embedding_backend="auto",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="auto",
    )

    monkeypatch.setattr("mempalace.backends.embeddings._resolve_torch_device", lambda device: "cuda")
    embedding_function = build_embedding_function(cfg)

    assert isinstance(embedding_function, TransformersEmbeddingFunction)
    assert embedding_function.device == "cuda"


def test_query_collection_with_torch_returns_chroma_shaped_results():
    fake = _FakeSearchCollection()
    embedding_function = lambda texts: [[1.0, 0.0] for _ in texts]

    result = query_collection_with_torch(
        fake,
        embedding_function,
        query_texts=["query target"],
        n_results=2,
        device="cpu",
        tile_size=2,
    )

    assert result["ids"] == [["1", "2"]]
    assert result["documents"] == [["query target", "middle"]]
    assert result["metadatas"][0][0]["source_file"] == "a.md"
    assert len(result["distances"][0]) == 2
    assert fake.get_calls[0]["include"] == ["documents", "metadatas", "embeddings"]


def test_query_collection_with_torch_stabilizes_equal_scores_by_id():
    fake = _FakeTieSearchCollection()
    embedding_function = lambda texts: [[1.0, 0.0] for _ in texts]

    result = query_collection_with_torch(
        fake,
        embedding_function,
        query_texts=["anything"],
        n_results=2,
        device="cpu",
        tile_size=2,
    )

    assert result["ids"] == [["a", "b"]]
    assert result["documents"] == [["doc a", "doc b"]]


def test_query_corpus_with_torch_handles_k_wider_than_corpus():
    result = query_corpus_with_torch(
        {
            "ids": ["2", "1"],
            "documents": ["far", "target"],
            "metadatas": [{}, {}],
            "embeddings": [[0.0, 1.0], [1.0, 0.0]],
        },
        lambda texts: [[1.0, 0.0] for _ in texts],
        query_texts=["target"],
        n_results=5,
        device="cpu",
    )

    assert result["ids"] == [["1", "2"]]
    assert len(result["distances"][0]) == 2


def test_query_corpus_with_torch_zero_vectors_remain_deterministic():
    result = query_corpus_with_torch(
        {
            "ids": ["b", "a", "c"],
            "documents": ["doc b", "doc a", "doc c"],
            "metadatas": [{}, {}, {}],
            "embeddings": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        },
        lambda texts: [[1.0, 0.0] for _ in texts],
        query_texts=["anything"],
        n_results=3,
        device="cpu",
    )

    assert result["ids"] == [["a", "b", "c"]]
    assert result["distances"] == [[1.0, 1.0, 1.0]]


def test_custom_kernel_topk_loader_uses_extension(monkeypatch):
    calls = []

    def _fake_topk(corpus_tensor, query_tensor, id_ranks_tensor, top_k, tile_size):
        calls.append((corpus_tensor, query_tensor, id_ranks_tensor, top_k, tile_size))
        return "scores", "indices"

    fake_extension = SimpleNamespace(topk=_fake_topk)
    monkeypatch.setitem(sys.modules, "mempalace.backends._cuda_exact_kernel", fake_extension)

    result = cuda_exact_kernel.topk_with_custom_kernel(
        "corpus",
        "query",
        "ranks",
        top_k=7,
        backend="extension",
        tile_size=11,
    )

    assert result == ("scores", "indices")
    assert calls == [("corpus", "query", "ranks", 7, 11)]


def test_custom_kernel_topk_disabled_returns_none(monkeypatch):
    fake_extension = SimpleNamespace(topk=lambda *args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setitem(sys.modules, "mempalace.backends._cuda_exact_kernel", fake_extension)

    assert (
        cuda_exact_kernel.topk_with_custom_kernel(
            "corpus",
            "query",
            "ranks",
            top_k=2,
            backend="off",
        )
        is None
    )


def test_exact_search_custom_kernel_hook_preserves_ranking(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("custom CUDA kernel hook only runs on CUDA devices")

    calls = []

    def _fake_custom_kernel(corpus_tensor, query_tensor, *, backend=None, tile_size=32768):
        calls.append({"backend": backend, "tile_size": tile_size})
        return torch.matmul(corpus_tensor, query_tensor)

    monkeypatch.setattr(
        "mempalace.backends.exact_search.score_vector_with_custom_kernel",
        _fake_custom_kernel,
    )
    monkeypatch.setattr(
        "mempalace.backends.exact_search.topk_with_custom_kernel",
        lambda *args, **kwargs: None,
    )

    result = query_corpus_with_torch(
        {
            "ids": ["b", "a", "c"],
            "documents": ["doc b", "doc a", "doc c"],
            "metadatas": [{}, {}, {}],
            "embeddings": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        },
        lambda texts: [[1.0, 0.0] for _ in texts],
        query_texts=["anything"],
        n_results=2,
        device="cuda",
        tile_size=2,
        kernel_backend="extension",
    )

    assert calls == [{"backend": "extension", "tile_size": 2}]
    assert result["ids"] == [["a", "b"]]


def test_query_collection_with_torch_uses_persisted_exact_index(tmp_path):
    fake = _FakeSearchCollection()
    embedding_function = lambda texts: [[1.0, 0.0] for _ in texts]

    first = query_collection_with_torch(
        fake,
        embedding_function,
        query_texts=["query target"],
        n_results=2,
        device="cpu",
        tile_size=2,
        palace_path=str(tmp_path),
    )
    first_get_count = len(fake.get_calls)

    second = query_collection_with_torch(
        fake,
        embedding_function,
        query_texts=["query target"],
        n_results=2,
        device="cpu",
        tile_size=2,
        palace_path=str(tmp_path),
    )

    assert first["ids"] == [["1", "2"]]
    assert second["ids"] == [["1", "2"]]
    assert len(fake.get_calls) == first_get_count
    assert (tmp_path / "mempalace_exact_index.npz").exists()
    assert (tmp_path / "mempalace_exact_index_manifest.json").exists()


def test_query_collection_with_torch_rebuilds_when_source_db_changes(tmp_path):
    fake = _FakeSearchCollection()
    embedding_function = lambda texts: [[1.0, 0.0] for _ in texts]
    db_path = tmp_path / "chroma.sqlite3"
    db_path.write_text("db", encoding="utf-8")

    query_collection_with_torch(
        fake,
        embedding_function,
        query_texts=["query target"],
        n_results=2,
        device="cpu",
        tile_size=2,
        palace_path=str(tmp_path),
    )
    fake.get_calls.clear()

    os.utime(db_path, None)

    query_collection_with_torch(
        fake,
        embedding_function,
        query_texts=["query target"],
        n_results=2,
        device="cpu",
        tile_size=2,
        palace_path=str(tmp_path),
    )

    assert fake.get_calls


def test_chroma_collection_marks_exact_index_dirty_after_write_and_rebuilds(tmp_path):
    fake = _FakeMutableSearchCollection()
    collection = ChromaCollection(
        fake,
        palace_path=str(tmp_path),
        embedding_function=lambda texts: [[1.0, 0.0] for _ in texts],
        search_backend="torch",
        search_device="cpu",
        search_tile_size=4,
    )

    first = collection.query(query_texts=["query target"], n_results=2)
    assert first["ids"] == [["1", "2"]]
    assert not (tmp_path / "mempalace_exact_index.stale").exists()

    collection.add(
        documents=["new exact"],
        ids=["4"],
        metadatas=[{"wing": "project", "room": "backend", "source_file": "d.md"}],
    )
    assert (tmp_path / "mempalace_exact_index.stale").exists()

    rebuilt = collection.query(query_texts=["query target"], n_results=2)

    assert rebuilt["ids"] == [["1", "4"]]
    assert not (tmp_path / "mempalace_exact_index.stale").exists()


def test_chroma_collection_stale_finality_after_delete_and_add(tmp_path):
    fake = _FakeMutableSearchCollection()
    collection = ChromaCollection(
        fake,
        palace_path=str(tmp_path),
        embedding_function=lambda texts: [[1.0, 0.0] for _ in texts],
        search_backend="torch",
        search_device="cpu",
        search_tile_size=4,
    )

    initial = collection.query(query_texts=["query target"], n_results=2)
    assert initial["ids"] == [["1", "2"]]

    collection.delete(ids=["1"])
    collection.add(
        documents=["new exact"],
        ids=["4"],
        metadatas=[{"wing": "project", "room": "backend", "source_file": "d.md"}],
    )

    final = collection.query(query_texts=["query target"], n_results=2)

    assert "1" not in final["ids"][0]
    assert final["ids"] == [["4", "2"]]
    assert not (tmp_path / "mempalace_exact_index.stale").exists()


def test_chroma_collection_torch_query_falls_back_to_raw_query(monkeypatch):
    fake = _FakeSearchCollection()
    collection = ChromaCollection(
        fake,
        embedding_function=lambda texts: [[1.0, 0.0] for _ in texts],
        search_backend="torch",
        search_device="cpu",
        search_tile_size=4,
    )

    monkeypatch.setattr(
        "mempalace.backends.chroma.query_collection_with_torch",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    assert collection.query(query_texts=["query target"], n_results=1) == {"kind": "fallback"}
    assert fake.query_calls == [{"query_texts": ["query target"], "n_results": 1}]


def test_chroma_collection_passes_exact_kernel_backend_to_torch_query(monkeypatch):
    fake = _FakeSearchCollection()
    query_mock = MagicMock(return_value={"kind": "torch"})
    collection = ChromaCollection(
        fake,
        embedding_function=lambda texts: [[1.0, 0.0] for _ in texts],
        search_backend="torch",
        search_device="cuda",
        search_tile_size=4,
        exact_kernel_backend="extension",
    )

    monkeypatch.setattr("mempalace.backends.chroma.query_collection_with_torch", query_mock)

    assert collection.query(query_texts=["query target"], n_results=1) == {"kind": "torch"}
    assert query_mock.call_args.kwargs["kernel_backend"] == "extension"


def test_chroma_backend_passes_embedding_function_to_client(monkeypatch, tmp_path):
    fake_collection = object()
    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection
    embedding_function = object()

    monkeypatch.setattr("mempalace.backends.chroma.chromadb.PersistentClient", lambda path: fake_client)

    backend = ChromaBackend(
        embedding_function=embedding_function,
        search_backend="torch",
        search_device="cpu",
        search_tile_size=128,
    )
    result = backend.get_collection(str(tmp_path), collection_name="mempalace_drawers", create=True)

    fake_client.get_or_create_collection.assert_called_once_with(
        "mempalace_drawers",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
    )
    assert isinstance(result, ChromaCollection)


def test_chroma_backend_reopens_existing_collection_on_embedding_conflict(monkeypatch, tmp_path):
    fake_collection = object()
    fake_client = MagicMock()
    fake_client.get_or_create_collection.side_effect = ValueError(
        "Embedding function conflict: new: transformers vs persisted: default"
    )
    fake_client.get_collection.return_value = fake_collection

    monkeypatch.setattr("mempalace.backends.chroma.chromadb.PersistentClient", lambda path: fake_client)

    backend = ChromaBackend(
        embedding_function=object(),
        search_backend="torch",
        search_device="cuda",
        search_tile_size=128,
    )
    result = backend.get_collection(str(tmp_path), collection_name="mempalace_drawers", create=True)

    fake_client.get_or_create_collection.assert_called_once()
    fake_client.get_collection.assert_called_once_with("mempalace_drawers")
    assert isinstance(result, ChromaCollection)
    assert result._embedding_function is None


def test_resolve_search_backend_auto_uses_chroma_when_cpu_requested():
    assert _resolve_search_backend("auto", "cpu") == "chroma"


def test_resolve_search_backend_auto_uses_torch_when_cuda_available(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    assert _resolve_search_backend("auto", "auto") == "torch"


def test_resolve_search_backend_exact_alias_maps_to_torch():
    assert _resolve_search_backend("exact", "cpu") == "torch"
