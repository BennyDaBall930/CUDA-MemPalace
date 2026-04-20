from __future__ import annotations

import importlib

import pytest


def _load_extension_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for compiled exact-kernel parity tests")

    try:
        extension = importlib.import_module("mempalace.backends._cuda_exact_kernel")
    except ImportError:
        pytest.skip("optional CUDA exact kernel extension is not built")

    return extension, torch


@pytest.mark.parametrize(
    ("rows", "dims", "tile_size"),
    [
        (1, 3, 2),
        (4, 2, 3),
        (33, 7, 5),
        (257, 17, 64),
    ],
)
def test_cuda_score_vector_matches_torch(rows, dims, tile_size):
    extension, torch = _load_extension_or_skip()
    torch.manual_seed(20260415 + rows + dims)

    corpus = torch.randn(rows, dims, device="cuda", dtype=torch.float32)
    query = torch.randn(dims, device="cuda", dtype=torch.float32)
    corpus = torch.nn.functional.normalize(corpus, p=2, dim=1).contiguous()
    query = torch.nn.functional.normalize(query, p=2, dim=0).contiguous()

    actual = extension.score_vector(corpus, query, tile_size)
    expected = torch.matmul(corpus, query)

    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_cuda_topk_matches_reference_tie_order():
    extension, torch = _load_extension_or_skip()

    corpus = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
        ],
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    query = torch.tensor([1.0, 0.0], device="cuda", dtype=torch.float32).contiguous()
    ids = ["b", "a", "d", "c"]
    id_rank_by_id = {row_id: rank for rank, row_id in enumerate(sorted(ids))}
    id_ranks = torch.tensor([id_rank_by_id[row_id] for row_id in ids], device="cuda", dtype=torch.long)

    scores, indices = extension.topk(corpus, query, id_ranks, 3, 2)
    actual_ids = [ids[index] for index in indices.detach().cpu().tolist()]
    actual_scores = scores.detach().cpu().tolist()

    expected_order = sorted(range(len(ids)), key=lambda idx: (-float(torch.matmul(corpus[idx], query)), ids[idx]))[:3]

    assert actual_ids == [ids[index] for index in expected_order]
    assert actual_scores == [pytest.approx(float(torch.matmul(corpus[index], query))) for index in expected_order]
