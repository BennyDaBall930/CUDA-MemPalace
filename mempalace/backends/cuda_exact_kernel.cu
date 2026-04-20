#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <c10/cuda/CUDAException.h>


__global__ void mempalace_score_vector_kernel(
    const float* __restrict__ corpus,
    const float* __restrict__ query,
    float* __restrict__ scores,
    const int64_t rows,
    const int64_t cols)
{
    extern __shared__ float partials[];

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    float local_sum = 0.0f;
    const int64_t row_offset = row * cols;
    for (int64_t col = static_cast<int64_t>(threadIdx.x); col < cols; col += blockDim.x) {
        local_sum += corpus[row_offset + col] * query[col];
    }

    partials[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores[row] = partials[0];
    }
}


__device__ inline bool mempalace_precedes(
    const float lhs_score,
    const int64_t lhs_rank,
    const float rhs_score,
    const int64_t rhs_rank)
{
    return (lhs_score > rhs_score) || ((lhs_score == rhs_score) && (lhs_rank < rhs_rank));
}


__global__ void mempalace_rank_topk_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ id_ranks,
    float* __restrict__ top_scores,
    int64_t* __restrict__ top_indices,
    const int64_t rows,
    const int64_t k)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float my_score = scores[row];
    const int64_t my_rank = id_ranks[row];
    int64_t rank = 0;

    for (int64_t other = 0; other < rows; ++other) {
        if (other == row) {
            continue;
        }
        if (mempalace_precedes(scores[other], id_ranks[other], my_score, my_rank)) {
            ++rank;
            if (rank >= k) {
                return;
            }
        }
    }

    top_scores[rank] = my_score;
    top_indices[rank] = row;
}


void launch_mempalace_score_vector(
    const float* corpus,
    const float* query,
    float* scores,
    int64_t rows,
    int64_t cols)
{
    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(rows));
    const size_t shared_bytes = threads * sizeof(float);
    mempalace_score_vector_kernel<<<blocks, threads, shared_bytes>>>(
        corpus,
        query,
        scores,
        rows,
        cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void launch_mempalace_rank_topk(
    const float* scores,
    const int64_t* id_ranks,
    float* top_scores,
    int64_t* top_indices,
    int64_t rows,
    int64_t k)
{
    const int threads = 256;
    const dim3 blocks(static_cast<unsigned int>((rows + threads - 1) / threads));
    mempalace_rank_topk_kernel<<<blocks, threads>>>(
        scores,
        id_ranks,
        top_scores,
        top_indices,
        rows,
        k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
