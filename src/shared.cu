#include "shared.h"


namespace cumesh {


/**
 * Canonicalize an atomically filled CSR adjacency.
 *
 * The fill kernels claim slots with atomicAdd, so the order inside each
 * segment depends on warp scheduling. Consumers such as the QEM and skinny-cost
 * sums in simplify.cu accumulate floats in segment order, so an unordered
 * segment makes their results, and every decision derived from them, vary from
 * run to run. Sorting each segment by id gives the order a serial build would
 * produce, independent of scheduling.
 *
 * @param adjacency: the CSR values, shape (num_items), sorted in place
 * @param offsets: the CSR offsets, shape (num_segments + 1)
 */
void sort_adjacency_segments(
    int* adjacency,
    const int* offsets,
    size_t num_items,
    size_t num_segments,
    Buffer<char>& cub_temp_storage,
    cudaStream_t stream,
    int device_index
) {
    if (num_items == 0 || num_segments == 0) return;
    ScopedCudaTemporary<int> sorted_storage(num_items, stream, device_index);
    int* sorted = sorted_storage.get();
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(
        nullptr, temp_storage_bytes,
        adjacency, sorted,
        num_items, num_segments,
        offsets, offsets + 1,
        stream
    ));
    cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(
        cub_temp_storage.ptr, temp_storage_bytes,
        adjacency, sorted,
        num_items, num_segments,
        offsets, offsets + 1,
        stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        adjacency, sorted, num_items * sizeof(int),
        cudaMemcpyDeviceToDevice, stream
    ));
    sorted_storage.reset();
}


/**
 * Hook edges
 * @param adj: the buffer for adjacency, shape (M)
 * @param M: the number of adjacency
 * @param conn_comp_ids: the buffer for connected component ids, shape (F)
 * @param end_flag: flag to indicate if any union operation happened
 */
__global__ void hook_edges_kernel(
    const int2* adj,
    const int M,
    int* conn_comp_ids,
    int* end_flag
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    // get adjacent faces
    int f0 = adj[tid].x;
    int f1 = adj[tid].y;

    // union
    // find roots
    int root0 = conn_comp_ids[f0];
    while (root0 != conn_comp_ids[root0]) {
        root0 = conn_comp_ids[root0];
    }
    int root1 = conn_comp_ids[f1];
    while (root1 != conn_comp_ids[root1]) {
        root1 = conn_comp_ids[root1];
    }

    if (root0 == root1) return;

    int high = max(root0, root1);
    int low = min(root0, root1);
    atomicMin(&conn_comp_ids[high], low);
    *end_flag = 0;
}


/**
 * Compress connected components
 * @param conn_comp_ids: the buffer for connected component ids, shape (F)
 * @param F: the number of faces
 */
__global__ void compress_components_kernel(
    int* conn_comp_ids,
    const int F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int p = conn_comp_ids[tid];
    while (p != conn_comp_ids[p]) {
        p = conn_comp_ids[p];
    }
    conn_comp_ids[tid] = p;
}


} // namespace cumesh