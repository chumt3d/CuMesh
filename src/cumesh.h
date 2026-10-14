#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <map>
#include <vector>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

#include "utils.h"


#define BLOCK_SIZE 256


namespace cumesh {

class ReleaseCompletionToken {
public:
    ReleaseCompletionToken(
        int device_index,
        std::int64_t stream_id,
        std::uint64_t owner_id,
        std::uint64_t generation
    );
    ~ReleaseCompletionToken() noexcept;

    ReleaseCompletionToken(const ReleaseCompletionToken&) = delete;
    ReleaseCompletionToken& operator=(const ReleaseCompletionToken&) = delete;

    int device_index() const noexcept;
    std::int64_t stream_id() const noexcept;
    std::uint64_t owner_id() const noexcept;
    std::uint64_t generation() const noexcept;
    bool is_complete() const;
    std::string status() const;
    std::int64_t wait_on_current_stream() const;
    void synchronize() const;

    cudaError_t record(const c10::cuda::CUDAStream& stream) noexcept;
    cudaError_t wait(const c10::cuda::CUDAStream& stream) const noexcept;
    cudaError_t synchronize_noexcept() const noexcept;
    bool recorded() const noexcept;

private:
    cudaEvent_t event_;
    int device_index_;
    std::int64_t stream_id_;
    std::uint64_t owner_id_;
    std::uint64_t generation_;
    std::atomic<bool> recorded_{false};
};

struct ReleaseResult {
    std::string status;
    std::size_t buffers_released;
    int device_index;
    std::int64_t stream_id;
    int stream_synchronize_count;
    std::shared_ptr<ReleaseCompletionToken> completion_token;
    std::size_t quarantined_buffer_count;
    int error_code;
    std::string error_message;
    std::uint64_t owner_id = 0;
    std::uint64_t generation = 0;
    int cleanup_error_code = 0;
    std::string cleanup_error_message;
    int primary_error_code = 0;
    std::string primary_error_message;
};

class CuMesh;

class CuMeshOperationScope {
public:
    explicit CuMeshOperationScope(CuMesh& owner);
    ~CuMeshOperationScope() noexcept(false);

    cudaStream_t stream() const noexcept;
    int device_index() const noexcept;
    std::int64_t stream_id() const noexcept;
    c10::cuda::CUDAStream caller_stream() const noexcept;
    c10::cuda::CUDAStream execution_stream() const noexcept;
    void bridge_to_caller();

private:
    CuMesh& owner_;
    std::unique_lock<std::recursive_mutex> lock_;
    c10::cuda::CUDAStream canonical_stream_;
    c10::cuda::CUDAStream caller_stream_;
    at::cuda::CUDAEvent ingress_event_;
    std::optional<c10::cuda::CUDAStreamGuard> stream_guard_;
    bool bridged_to_caller_ = false;
};

class CuMesh {
public:
    static constexpr std::size_t buffer_slot_count = 48;

    Buffer<float3> vertices;
    Buffer<int3> faces;

    // Geometric properties
    Buffer<float> face_areas;
    Buffer<float3> face_normals;
    Buffer<float3> vertex_normals;

    // Connectivity
    Buffer<uint64_t> edges;
    Buffer<int> boundaries;
    Buffer<uint8_t> vert_is_boundary;
    Buffer<uint8_t> vert_is_manifold;
    Buffer<int> vert2edge;
    Buffer<int> vert2edge_cnt;
    Buffer<int> vert2edge_offset;
    Buffer<int> vert2bound;
    Buffer<int> vert2bound_cnt;
    Buffer<int> vert2bound_offset;
    Buffer<int> edge2face;
    Buffer<int> edge2face_cnt;
    Buffer<int> edge2face_offset;
    Buffer<int3> face2edge;
    Buffer<int> vert2face;
    Buffer<int> vert2face_cnt;
    Buffer<int> vert2face_offset;
    Buffer<int2> manifold_face_adj;
    Buffer<int2> manifold_bound_adj;
    Buffer<int> conn_comp_ids;
    Buffer<int> bound_conn_comp_ids;
    Buffer<int> loop_boundaries;
    Buffer<int> loop_boundaries_offset;
    int num_conn_comps = 0;
    int num_bound_conn_comps = 0;
    int num_bound_loops = 0;

    // Cleanup
    Buffer<int> vertices_map;
    Buffer<int> faces_map;

    // Simplification
    Buffer<float> edge_collapse_costs;
    Buffer<uint64_t> propagated_costs;

    // Atlasing
    int atlas_num_charts = 0;
    Buffer<int> atlas_chart_ids;
    Buffer<int> atlas_chart_vertex_map;
    Buffer<int3> atlas_chart_faces;
    Buffer<int> atlas_chart_faces_offset;
    Buffer<int> atlas_chart_vertex_offset;
    Buffer<float2> atlas_chart_uvs;

    Buffer<float4> atlas_chart_normal_cones;
    Buffer<uint64_t> atlas_chart_adj;
    Buffer<float> atlas_chart_adj_length;
    Buffer<float> atlas_chart_perims;
    Buffer<float> atlas_chart_areas;
    Buffer<int> atlas_chart2edge;
    Buffer<int> atlas_chart2edge_cnt;
    Buffer<int> atlas_chart2edge_offset;

    // Temporary storage
    Buffer<char> temp_storage;
    Buffer<char> cub_temp_storage;

    CuMesh();

    ~CuMesh() noexcept;

    CuMesh(const CuMesh&) = delete;
    CuMesh& operator=(const CuMesh&) = delete;
    CuMesh(CuMesh&&) = delete;
    CuMesh& operator=(CuMesh&&) = delete;

    void swap(CuMesh& other);

    ReleaseResult release_async(bool synchronize = false);

    std::size_t allocated_buffer_count() const noexcept;

    std::uint64_t owner_id() const noexcept;

    std::uint64_t generation() const noexcept;

    int device_index() const noexcept;

    static std::uint64_t destructor_release_error_count() noexcept;

    static std::uint64_t destructor_device_synchronize_count() noexcept;

    static std::size_t quarantined_cuda_allocation_count() noexcept;

    static void retry_quarantined_cuda_allocations();

#ifdef CUMESH_ENABLE_TEST_HOOKS
    static void
    test_fail_release_after(std::int64_t successful_enqueues) noexcept;
    static void test_fail_release_event_record_once() noexcept;
    static void test_fail_release_event_create_once() noexcept;
    static void test_fail_release_token_allocation_once() noexcept;
    static void test_fail_release_caller_wait_once() noexcept;
    static void test_fail_release_synchronize_once() noexcept;
    std::uint8_t test_release_lifecycle_fact_mask() const noexcept;
    static void test_fail_temporary_allocation_after(
        std::int64_t successful_allocations) noexcept;
    static void test_fail_output_allocation_after(
        std::int64_t successful_allocations) noexcept;
    static void
    test_fail_output_copy_after(std::int64_t successful_copies) noexcept;
    static void
    test_fail_output_map_after(std::int64_t successful_maps) noexcept;
    static void
    test_fail_temporary_cleanup(std::int64_t async_free_failures,
                                std::int64_t synchronize_failures,
                                std::int64_t fallback_free_failures) noexcept;
    static void test_fail_destructor_device_synchronize_once() noexcept;
    static void test_fail_destructor_legacy_free_once() noexcept;
    static std::uint64_t test_destructor_legacy_free_attempt_count() noexcept;
    static void test_fail_buffer_attributes_and_cleanup_once() noexcept;
    static void test_fail_buffer_attributes_once() noexcept;
    static void test_validate_stream_device(const c10::cuda::CUDAStream& stream, int expected_device);
    static void test_validate_completion_device_identity(int token_device, int expected_device);
#endif

    static std::uint64_t active_temporary_allocation_count() noexcept;

    static std::uint64_t total_temporary_allocation_count() noexcept;

    int num_vertices() const;

    int num_faces() const;

    int num_edges() const;

    int num_boundaries() const;

    int num_conneted_components() const;

    int num_boundary_conneted_components() const;

    int num_boundary_loops() const;

    void clear_cache();

    /**
     * Initialize mesh
     * 
     * @param vertices The vertex positions as an [V, 3] tensor.
     * @param faces The triangle faces as an [F, 3] tensor.
     */
    void init(const torch::Tensor& vertices, const torch::Tensor& faces);

    /**
     * Get the mesh.
     *
     * @return A tuple of the vertex positions and the triangle faces.
     */
    std::tuple<torch::Tensor, torch::Tensor> read();

    /**
     * Get the face normals.
     * 
     * @return The face normals as an [F, 3] tensor.
     */
    torch::Tensor read_face_normals();

    /**
     * Get the normals of the vertices.
     * 
     * @return The vertex normals as an [V, 3] tensor.
     */
    torch::Tensor read_vertex_normals();
    
    /**
     * Get the edges of the mesh.
     * 
     * @return The edges as an [E, 2] tensor.
     */
    torch::Tensor read_edges();

    /**
     * Get the boundaries of the mesh.
     * 
     * @return The boundaries as an [B] tensor.
     *         Each element is the index of a boundary edge.
     */
    torch::Tensor read_boundaries();

    /**
     * Get the manifold faces adjacency.
     * 
     * @return The manifold faces adjacency as an [M, 2] tensor.
     */
    torch::Tensor read_manifold_face_adjacency();

    /**
     * Get the manifold boundary adjacency.
     * 
     * @return The manifold boundary adjacency as an [M, 2] tensor.
     */
    torch::Tensor read_manifold_boundary_adjacency();

    /**
     * Get the connected components of the mesh.
     *
     * @return A tuple of:
     * - The number of connected components.
     * - The connected components ids as an [F] tensor.
     */
    std::tuple<int, torch::Tensor> read_connected_components();

    /**
     * Get the connected components of the mesh boundaries.
     *
     * @return A tuple of:
     * - The number of boundary connected components.
     * - The boundary connected components ids as an [B] tensor.
     */
    std::tuple<int, torch::Tensor> read_boundary_connected_components();

    /**
     * Get the boundary loops of the mesh.
     *
     * @return A tuple of:
     * - The number of boundary loops.
     * - The boundary loops as an [L] tensor.
     * - The boundary loops offsets as an [L+1] tensor.
     */
    std::tuple<int, torch::Tensor, torch::Tensor> read_boundary_loops();

    /**
     * Get all cached data.
     * 
     * @return A dictionary of all cached data.
     */
    std::unordered_map<std::string, torch::Tensor> read_all_cache();
    

    // Geometric functions

    /**
     * Compute face areas.
     * This function refreshes:
     * - face_areas
     */
    void compute_face_areas();

    /**
     * Compute face normals.
     * This function refreshes:
     * - face_normals
     */
    void compute_face_normals();

    /**
     * Compute vertex normals.
     * This function requires:
     * - vert2face
     * - vert2face_offset
     * This function refreshes:
     * - vertex_normals
     */
    void compute_vertex_normals();


    // Connectivity functions

    /**
     * Get the vertex to face adjacency.
     * This function refreshes:
     * - vert2face
     * - vert2face_cnt
     * - vert2face_offset
     */
    void get_vertex_face_adjacency();

    /**
     * Get the edges of the mesh.
     * This function refreshes:
     * - edges
     * - edge2face_cnt
     */
    void get_edges();

    /**
     * Get the edges of the mesh.
     * This function requires:
     * - edges
     * - edge2face_cnt
     * - vert2face
     * - vert2face_offset
     * This function refreshes:
     * - edge2face
     * - edge2face_offset
     * - face2edge
     */
    void get_edge_face_adjacency();

    /**
     * Get the vertex to edge adjacency.
     * This function requires:
     * - edges
     * This function refreshes:
     * - vert2edge
     * - vert2edge_cnt
     * - vert2edge_offset
     */
    void get_vertex_edge_adjacency();

    /**
     * Get boundary information.
     * This function requires:
     * - edges
     * - edge2face_cnt
     * This function refreshes:
     * - boundaries
     * - vert_is_boundary
     */
    void get_boundary_info();

    /**
     * Get the vertex to boundary adjacency.
     * This function requires:
     * - edges
     * - boundaries
     * This function refreshes:
     * - vert2bound
     * - vert2bound_cnt
     * - vert2bound_offset
     */
    void get_vertex_boundary_adjacency();

    /**
     * Get edge is manifold information.
     * This function requires:
     * - vert2edge
     * - vert2edge_offset
     * - edge2face_cnt
     * This function refreshes:
     * - vert_is_manifold
     */
    void get_vertex_is_manifold();

    /**
     * Get the face adjacency for manifold edges.
     * This function requires:
     * - edge2face
     * - edge2face_offset
     * This function refreshes:
     * - manifold_face_adj
     */
    void get_manifold_face_adjacency();

    /**
     * Get the face adjacency for manifold boundaries.
     * This function requires:
     * - vert_is_manifold
     * - vert2bound
     * - vert2bound_offset
     * This function refreshes:
     * - manifold_bound_adj
     */
    void get_manifold_boundary_adjacency();

    /**
     * Get the connected components of the mesh.
     * This function requires:
     * - manifold_face_adj
     * This function refreshes:
     * - conn_comp_ids
     */
    void get_connected_components();

    /**
     * Get the boundary connected components of the mesh.
     * This function requires:
     * - manifold_bound_adj
     * This function refreshes:
     * - bound_conn_comp_ids
     */
    void get_boundary_connected_components();

    /**
     * Get the boundary loops of the mesh.
     * This function requires:
     * - vert2bound
     * - vert2bound_offset
     * - vert_is_boundary
     * - bound_conn_comp_ids
     * This function refreshes:
     * - loop_boundaries
     * - loop_boundaries_offset
     */
    void get_boundary_loops();


    // Cleanup functions
    
    /**
     * Remove faces.
     */
    void remove_faces(torch::Tensor& face_mask);
    void _remove_faces(uint8_t* face_mask);

    /**
     * Remove unreferenced vertices.
     */
    void remove_unreferenced_vertices();

    /**
     * Remove duplicate faces.
     */
    void remove_duplicate_faces();

    /**
     * Remove degenerate faces.
     */
    void remove_degenerate_faces(float abs_thresh, float rel_thresh);

    /**
     * Fill holes.
     * This function requires:
     * - loop_boundaries
     * - loop_boundaries_offset
     * 
     * @param max_hole_perimeter The maximum perimeter of a hole to be filled.
     */
    void fill_holes(float max_hole_perimeter);

    /**
     * Repair Non-manifold edges by splitting edges.
     * This function requires:
     * - manifold_face_adj
     * This function refreshes:
     * - vertices
     * - faces
     * This function destroys:
     * - All connectivity information
     */
    void repair_non_manifold_edges();

    /**
     * Remove faces on non-manifold edges.
     * For each non-manifold edge (shared by >2 faces), only keep the first 2 faces.
     * This repairs non-manifold edges by deleting faces instead of splitting vertices.
     * This function requires:
     * - edge2face
     * - edge2face_offset
     * - edge2face_cnt
     * This function refreshes:
     * - vertices
     * - faces
     * This function destroys:
     * - All connectivity information
     */
    void remove_non_manifold_faces();

    /**
     * Remove small connected components.
     * This function requires:
     * - conn_comp_ids
     * This function refreshes:
     * - vertices
     * - faces
     * This function destroys:
     * - All connectivity information
     * 
     * @param min_area The minimum area of the connected components to be kept.
     */
    void remove_small_connected_components(float min_area);

    /**
     * Unify face orientations.
     * This function requires:
     * - manifold_face_adj
     * This function refreshes:
     * - faces
     */
    void unify_face_orientations();
    

    // Simplification functions

    /**
     * Run the edge collapse algorithm.
     * This function refreshes:
     * - vertices
     * - faces
     * This function destroys:
     * - All connectivity information
     * 
     * @param lambda_edge_length The weight for edge length term.
     * @param lambda_skinny The weight for skinny term.
     * @param threshold The threshold for edge collapse cost.
     * @return A tuple of the number of vertices and the number of faces after simplification.
     */
    std::tuple<int, int> simplify_step(float lambda_edge_length, float lambda_skinny, float threshold, bool timing=false,double max_step_seconds=INFINITY);
    torch::Tensor initial_simplification_quadrics();
    std::tuple<int,int,torch::Tensor> simplify_step_accumulated(const torch::Tensor& quadrics,float lambda_edge_length,float lambda_skinny,float threshold,double max_error_squared,bool timing=false,double max_step_seconds=INFINITY);
    std::atomic<bool> simplification_cancel_requested{false};
    void cancel_simplification() noexcept {simplification_cancel_requested.store(true,std::memory_order_release);}
    void clear_simplification_cancel() noexcept {simplification_cancel_requested.store(false,std::memory_order_release);}
    bool simplification_cancelled() const noexcept {return simplification_cancel_requested.load(std::memory_order_acquire);}
    std::map<std::string,uint64_t> last_simplification_admission;
    std::map<std::string,uint64_t> simplification_admission_statistics();


    // Atlasing functions

   /**
     * Compute charts for atlasing.
     * This function requires:
     * - manifold_face_adj
     * This function refreshes:
     * - atlas_face_chart_ids
     * - atlas_chart_vertex_map
     * - atlas_chart_faces
     * - atlas_chart_faces_offset
     *
     *  @param  threshold_cone_half_angle_rad The threshold for the cone half angle in radians.
     *  @param  refine_iterations             The number of refinement iterations.
     *  @param  global_iterations             The number of global iterations.
     *  @param  smooth_strength               The strength of the smoothing.
     *  @param  area_penalty_weight           Coefficient for chart size penalty. Cost += Area * weight.
     *                                        Prevents charts from becoming too large if > 0, 
     *                                        or encourages larger charts if < 0 (though usually used to penalize size variance).
     *  @param  perimeter_area_ratio_weight   Coefficient for shape irregularity (long-strip) penalty. 
     *                                        Cost += (Perimeter / Area) * weight.
     *                                        Higher values penalize long strips and encourage circular/compact shapes.
     */
    void compute_charts(
        float threshold_cone_half_angle_rad, 
        int refine_iterations, 
        int global_iterations, 
        float smooth_strength,
        float area_penalty_weight,
        float perimeter_area_ratio_weight
    );

    /**
     * Read the atlas charts.
     *
     * @return A tuple of:
     * - The number of charts.
     * - The chart ids as an [F] tensor.
     * - The chart vertex map as an [V] tensor.
     * - The chart faces as an [F, 3] tensor.
     * - The chart vertices offset as an [C+1] tensor.
     * - The chart faces offset as an [C+1] tensor.
     */
    std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> read_atlas_charts();

private:
    friend class CuMeshOperationScope;

    auto all_buffers() noexcept {
        return std::tie(
            vertices,
            faces,
            face_areas,
            face_normals,
            vertex_normals,
            edges,
            boundaries,
            vert_is_boundary,
            vert_is_manifold,
            vert2edge,
            vert2edge_cnt,
            vert2edge_offset,
            vert2bound,
            vert2bound_cnt,
            vert2bound_offset,
            edge2face,
            edge2face_cnt,
            edge2face_offset,
            face2edge,
            vert2face,
            vert2face_cnt,
            vert2face_offset,
            manifold_face_adj,
            manifold_bound_adj,
            conn_comp_ids,
            bound_conn_comp_ids,
            loop_boundaries,
            loop_boundaries_offset,
            vertices_map,
            faces_map,
            edge_collapse_costs,
            propagated_costs,
            atlas_chart_ids,
            atlas_chart_vertex_map,
            atlas_chart_faces,
            atlas_chart_faces_offset,
            atlas_chart_vertex_offset,
            atlas_chart_uvs,
            atlas_chart_normal_cones,
            atlas_chart_adj,
            atlas_chart_adj_length,
            atlas_chart_perims,
            atlas_chart_areas,
            atlas_chart2edge,
            atlas_chart2edge_cnt,
            atlas_chart2edge_offset,
            temp_storage,
            cub_temp_storage
        );
    }

    auto all_buffers() const noexcept {
        return std::tie(
            vertices,
            faces,
            face_areas,
            face_normals,
            vertex_normals,
            edges,
            boundaries,
            vert_is_boundary,
            vert_is_manifold,
            vert2edge,
            vert2edge_cnt,
            vert2edge_offset,
            vert2bound,
            vert2bound_cnt,
            vert2bound_offset,
            edge2face,
            edge2face_cnt,
            edge2face_offset,
            face2edge,
            vert2face,
            vert2face_cnt,
            vert2face_offset,
            manifold_face_adj,
            manifold_bound_adj,
            conn_comp_ids,
            bound_conn_comp_ids,
            loop_boundaries,
            loop_boundaries_offset,
            vertices_map,
            faces_map,
            edge_collapse_costs,
            propagated_costs,
            atlas_chart_ids,
            atlas_chart_vertex_map,
            atlas_chart_faces,
            atlas_chart_faces_offset,
            atlas_chart_vertex_offset,
            atlas_chart_uvs,
            atlas_chart_normal_cones,
            atlas_chart_adj,
            atlas_chart_adj_length,
            atlas_chart_perims,
            atlas_chart_areas,
            atlas_chart2edge,
            atlas_chart2edge_cnt,
            atlas_chart2edge_offset,
            temp_storage,
            cub_temp_storage
        );
    }

    auto cache_buffers() noexcept {
        return std::tie(
            face_areas,
            face_normals,
            vertex_normals,
            edges,
            boundaries,
            vert_is_boundary,
            vert_is_manifold,
            vert2edge,
            vert2edge_cnt,
            vert2edge_offset,
            vert2bound,
            vert2bound_cnt,
            vert2bound_offset,
            edge2face,
            edge2face_cnt,
            edge2face_offset,
            face2edge,
            vert2face,
            vert2face_cnt,
            vert2face_offset,
            manifold_face_adj,
            manifold_bound_adj,
            conn_comp_ids,
            bound_conn_comp_ids,
            loop_boundaries,
            loop_boundaries_offset,
            vertices_map,
            faces_map,
            edge_collapse_costs,
            propagated_costs,
            atlas_chart_ids,
            atlas_chart_vertex_map,
            atlas_chart_faces,
            atlas_chart_faces_offset,
            atlas_chart_vertex_offset,
            atlas_chart_uvs,
            atlas_chart_normal_cones,
            atlas_chart_adj,
            atlas_chart_adj_length,
            atlas_chart_perims,
            atlas_chart_areas,
            atlas_chart2edge,
            atlas_chart2edge_cnt,
            atlas_chart2edge_offset,
            temp_storage,
            cub_temp_storage
        );
    }

    void reset_metadata() noexcept;
    void release_fallback_noexcept() noexcept;
    void require_operable_locked() const;
    void establish_canonical_stream_locked(int device_index);
    c10::cuda::CUDAStream canonical_stream_locked() const;
    void reset_cache_metadata() noexcept;
    void quarantine_operation_scope_failure_noexcept(const char* reason) noexcept;

    mutable std::recursive_mutex operation_mutex_;
    const std::uint64_t owner_id_;
    std::uint64_t generation_ = 0;
    int canonical_device_index_ = -1;
    std::optional<c10::cuda::CUDAStream> canonical_stream_;
    // These are independent facts: a completion token can later prove local
    // async work while a different tuple owner remains durably held in global
    // Q.  Never collapse them into one poisoned/not-poisoned flag.
    bool completion_proof_pending_ = false;
    bool durable_global_quarantine_ = false;
    std::string quarantine_reason_;
    std::shared_ptr<ReleaseCompletionToken> pending_release_token_;
    std::size_t pending_release_count_ = 0;
    std::uint64_t pending_release_generation_ = 0;
};

} // namespace cumesh
