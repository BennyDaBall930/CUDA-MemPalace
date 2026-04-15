#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;


void launch_mempalace_score_vector(
    const float* corpus,
    const float* query,
    float* scores,
    int64_t rows,
    int64_t cols);

void launch_mempalace_rank_topk(
    const float* scores,
    const int64_t* id_ranks,
    float* top_scores,
    int64_t* top_indices,
    int64_t rows,
    int64_t k);


static inline void check_cuda(const at::Tensor& tensor, const char* name) {
    if (!tensor.device().is_cuda()) {
        throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
    }
}


static inline void check_contiguous(const at::Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}


static inline void check_float32(const at::Tensor& tensor, const char* name) {
    if (tensor.scalar_type() != at::kFloat) {
        throw std::runtime_error(std::string(name) + " must be float32");
    }
}


static inline void check_int64(const at::Tensor& tensor, const char* name) {
    if (tensor.scalar_type() != at::kLong) {
        throw std::runtime_error(std::string(name) + " must be int64");
    }
}


static inline void check_score_inputs(const at::Tensor& corpus, const at::Tensor& query) {
    check_cuda(corpus, "corpus");
    check_cuda(query, "query");
    check_contiguous(corpus, "corpus");
    check_contiguous(query, "query");
    check_float32(corpus, "corpus");
    check_float32(query, "query");
    if (corpus.dim() != 2) {
        throw std::runtime_error("corpus must have shape [rows, dims]");
    }
    if (query.dim() != 1) {
        throw std::runtime_error("query must have shape [dims]");
    }
    if (corpus.size(1) != query.size(0)) {
        throw std::runtime_error("corpus dims must match query dims");
    }
}


at::Tensor score_vector(at::Tensor corpus, at::Tensor query, int64_t tile_size) {
    (void)tile_size;
    check_score_inputs(corpus, query);
    auto rows = corpus.size(0);
    auto scores = at::empty({rows}, corpus.options());
    if (rows == 0) {
        return scores;
    }

    launch_mempalace_score_vector(
        corpus.data_ptr<float>(),
        query.data_ptr<float>(),
        scores.data_ptr<float>(),
        rows,
        corpus.size(1));
    return scores;
}


std::vector<at::Tensor> topk(
    at::Tensor corpus,
    at::Tensor query,
    at::Tensor id_ranks,
    int64_t k,
    int64_t tile_size) {
    (void)tile_size;
    check_score_inputs(corpus, query);
    check_cuda(id_ranks, "id_ranks");
    check_contiguous(id_ranks, "id_ranks");
    check_int64(id_ranks, "id_ranks");
    if (id_ranks.dim() != 1 || id_ranks.size(0) != corpus.size(0)) {
        throw std::runtime_error("id_ranks must have shape [rows]");
    }
    if (k < 1) {
        throw std::runtime_error("k must be at least 1");
    }

    auto scores = score_vector(corpus, query, tile_size);
    auto rows = corpus.size(0);
    auto actual_k = std::min<int64_t>(k, rows);
    auto top_scores = at::empty({actual_k}, corpus.options());
    auto top_indices = at::empty({actual_k}, id_ranks.options());
    if (actual_k == 0) {
        return {top_scores, top_indices};
    }

    launch_mempalace_rank_topk(
        scores.data_ptr<float>(),
        id_ranks.data_ptr<int64_t>(),
        top_scores.data_ptr<float>(),
        top_indices.data_ptr<int64_t>(),
        rows,
        actual_k);
    return {top_scores, top_indices};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MemPalace CUDA exact scoring and deterministic top-k kernels";
    m.def("score_vector", &score_vector, "Compute exact cosine score vector");
    m.def("topk", &topk, "Compute deterministic top-k using score desc and ID-rank asc");
}
