#include "cumesh.h"


namespace cumesh {


template <typename T>
struct TorchTypeMapping;

template <> struct TorchTypeMapping<int> {
    static constexpr auto scalar_type = torch::kInt32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 1;
};

template <> struct TorchTypeMapping<char> {
    static constexpr auto scalar_type = torch::kInt8;
    static constexpr int sizeof_scalar = 1;
    static constexpr int channels = 1;
};

template <> struct TorchTypeMapping<uint8_t> {
    static constexpr auto scalar_type = torch::kUInt8;
    static constexpr int sizeof_scalar = 1;
    static constexpr int channels = 1;
};

template <> struct TorchTypeMapping<float> {
    static constexpr auto scalar_type = torch::kFloat32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 1;
};

template <> struct TorchTypeMapping<uint64_t> {
    static constexpr auto scalar_type = torch::kInt32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 2;
};

template <> struct TorchTypeMapping<int2> {
    static constexpr auto scalar_type = torch::kInt32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 2;
};

template <> struct TorchTypeMapping<int3> {
    static constexpr auto scalar_type = torch::kInt32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 3;
};

template <> struct TorchTypeMapping<float2> {
    static constexpr auto scalar_type = torch::kFloat32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 2;
};

template <> struct TorchTypeMapping<float3> {
    static constexpr auto scalar_type = torch::kFloat32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 3;
};

template <> struct TorchTypeMapping<float4> {
    static constexpr auto scalar_type = torch::kFloat32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 4;
};


template<typename T>
static torch::Tensor buffer_to_tensor(
    const Buffer<T>& buffer,
    const c10::cuda::CUDAStream& execution_stream,
    int device_index,
    const c10::cuda::CUDAStream& output_stream
) {
    using Mapping = TorchTypeMapping<T>;

    int64_t count = static_cast<int64_t>(buffer.size);
    std::vector<int64_t> shape;
    if (Mapping::channels == 1) {
        shape = {count}; // 1D Tensor
    } else {
        shape = {count, Mapping::channels}; // 2D Tensor [N, C]
    }

    auto options = torch::dtype(Mapping::scalar_type).device(torch::Device(torch::kCUDA, device_index));
    torch::Tensor tensor;
    {
        c10::cuda::CUDAStreamGuard allocation_guard(output_stream);
#ifdef CUMESH_ENABLE_TEST_HOOKS
        test_maybe_fail(output_allocation_successes_before_failure, "Torch output allocation");
#endif
        tensor = torch::empty(shape, options);
    }
    if (output_stream != execution_stream) {
        record_tensor_on_stream(tensor, execution_stream);
    }

    static constexpr int dst_bytes = Mapping::sizeof_scalar * Mapping::channels;
    if (count == 0) {
        return tensor;
    }
    if (sizeof(T) == dst_bytes) {
        CUDA_CHECK(cudaMemcpyAsync(
            tensor.data_ptr(),
            buffer.ptr,
            count * sizeof(T),
            cudaMemcpyDeviceToDevice,
            execution_stream.stream()
        ));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(
            tensor.data_ptr(),
            dst_bytes,
            buffer.ptr,
            sizeof(T),
            dst_bytes,
            count,
            cudaMemcpyDeviceToDevice,
            execution_stream.stream()
        ));
    }
#ifdef CUMESH_ENABLE_TEST_HOOKS
    test_maybe_fail(output_copy_successes_before_failure, "Torch output copy completion");
#endif

    return tensor;
}


void CuMesh::init(const torch::Tensor& vertices, const torch::Tensor& faces) {
    TORCH_CHECK(vertices.is_cuda() && faces.is_cuda(), "[CuMesh] init requires CUDA tensors");
    TORCH_CHECK(vertices.get_device() == faces.get_device(), "[CuMesh] init tensors must share one CUDA device");
    TORCH_CHECK(
        vertices.dim() == 2 && vertices.size(1) == 3,
        "[CuMesh] init vertices must have shape [V, 3]"
    );
    TORCH_CHECK(
        faces.dim() == 2 && faces.size(1) == 3,
        "[CuMesh] init faces must have shape [F, 3]"
    );
    TORCH_CHECK(vertices.is_contiguous(), "[CuMesh] init vertices must be contiguous");
    TORCH_CHECK(faces.is_contiguous(), "[CuMesh] init faces must be contiguous");
    TORCH_CHECK(
        vertices.scalar_type() == torch::kFloat32,
        "[CuMesh] init vertices must have float32 dtype"
    );
    TORCH_CHECK(
        faces.scalar_type() == torch::kInt32,
        "[CuMesh] init faces must have int32 dtype"
    );
    const int input_device = vertices.get_device();
    std::unique_lock<std::recursive_mutex> lock(operation_mutex_);
    require_cuda_allocation_quarantine_clear();
    require_operable_locked();
    TORCH_CHECK(
        allocated_buffer_count() == 0,
        "[CuMesh] init cannot replace existing owned buffers"
    );
    establish_canonical_stream_locked(input_device);
    const c10::cuda::CUDAStream stream = canonical_stream_locked();
    c10::cuda::CUDAStreamGuard stream_guard(stream);

    const size_t num_vertices = static_cast<size_t>(vertices.size(0));
    const size_t num_faces = static_cast<size_t>(faces.size(0));
    try {
        this->vertices.init_async(num_vertices, stream.stream(), input_device);
        this->vertices.size = num_vertices;
        this->faces.init_async(num_faces, stream.stream(), input_device);
        this->faces.size = num_faces;
        if (num_vertices != 0) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                this->vertices.ptr,
                sizeof(float3),
                vertices.data_ptr<float>(),
                sizeof(float) * 3,
                sizeof(float) * 3,
                num_vertices,
                cudaMemcpyDeviceToDevice,
                stream.stream()
            ));
        }
        if (num_faces != 0) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                this->faces.ptr,
                sizeof(int3),
                faces.data_ptr<int>(),
                sizeof(int) * 3,
                sizeof(int) * 3,
                num_faces,
                cudaMemcpyDeviceToDevice,
                stream.stream()
            ));
        }
    } catch (const std::exception& primary_error) {
        cudaError_t cleanup_error = cudaSuccess;
        const cudaError_t vertices_cleanup = this->vertices.release_async(stream.stream(), input_device);
        if (vertices_cleanup != cudaSuccess) {
            cleanup_error = vertices_cleanup;
        }
        const cudaError_t faces_cleanup = this->faces.release_async(stream.stream(), input_device);
        if (cleanup_error == cudaSuccess && faces_cleanup != cudaSuccess) {
            cleanup_error = faces_cleanup;
        }
        if (cleanup_error != cudaSuccess) {
            durable_global_quarantine_ = true;
            quarantine_reason_ = std::string("init_cleanup_failed_after_primary_error: ")
                + cudaGetErrorString(cleanup_error);
        }
        TORCH_CHECK(
            false,
            "[CuMesh] init primary failure: ", primary_error.what(),
            "; cleanup: ",
            cleanup_error == cudaSuccess ? "release_enqueued" : cudaGetErrorString(cleanup_error)
        );
    }
    ++generation_;
}


std::tuple<torch::Tensor, torch::Tensor> CuMesh::read() {
    CuMeshOperationScope operation(*this);
    auto vertices = buffer_to_tensor(
        this->vertices,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    auto faces = buffer_to_tensor(
        this->faces,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return std::make_tuple(vertices, faces);
}


torch::Tensor CuMesh::read_face_normals() {
    CuMeshOperationScope operation(*this);
    auto face_normals = buffer_to_tensor(
        this->face_normals,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return face_normals;
}


torch::Tensor CuMesh::read_vertex_normals() {
    CuMeshOperationScope operation(*this);
    auto vertex_normals = buffer_to_tensor(
        this->vertex_normals,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return vertex_normals;
}


torch::Tensor CuMesh::read_edges() {
    CuMeshOperationScope operation(*this);
    auto edges = buffer_to_tensor(
        this->edges,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return edges;
}


torch::Tensor CuMesh::read_boundaries() {
    CuMeshOperationScope operation(*this);
    auto boundaries = buffer_to_tensor(
        this->boundaries,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return boundaries;
}


torch::Tensor CuMesh::read_manifold_face_adjacency() {
    CuMeshOperationScope operation(*this);
    auto manifold_face_adj = buffer_to_tensor(
        this->manifold_face_adj,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return manifold_face_adj;
}


torch::Tensor CuMesh::read_manifold_boundary_adjacency() {
    CuMeshOperationScope operation(*this);
    auto manifold_bound_adj = buffer_to_tensor(
        this->manifold_bound_adj,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return manifold_bound_adj;
}

  
std::tuple<int, torch::Tensor> CuMesh::read_connected_components() {
    CuMeshOperationScope operation(*this);
    auto conn_comp_ids_tensor = buffer_to_tensor(
        this->conn_comp_ids,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return std::make_tuple(this->num_conn_comps, conn_comp_ids_tensor);
}


std::tuple<int, torch::Tensor> CuMesh::read_boundary_connected_components() {
    CuMeshOperationScope operation(*this);
    auto bound_conn_comp_ids_tensor = buffer_to_tensor(
        this->bound_conn_comp_ids,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return std::make_tuple(this->num_bound_conn_comps, bound_conn_comp_ids_tensor);
}


std::tuple<int, torch::Tensor, torch::Tensor> CuMesh::read_boundary_loops() {
    CuMeshOperationScope operation(*this);
    auto loop_boundaries_tensor = buffer_to_tensor(
        this->loop_boundaries,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    auto loop_boundaries_offset_tensor = buffer_to_tensor(
        this->loop_boundaries_offset,
        operation.execution_stream(),
        operation.device_index(),
        operation.caller_stream()
    );
    operation.bridge_to_caller();
    return std::make_tuple(this->num_bound_loops, loop_boundaries_tensor, loop_boundaries_offset_tensor);
}


std::unordered_map<std::string, torch::Tensor> CuMesh::read_all_cache() {
    CuMeshOperationScope operation(*this);
    std::unordered_map<std::string, torch::Tensor> cache;
    auto read_buffer = [&](const auto& buffer) {
        auto tensor = buffer_to_tensor(
            buffer,
            operation.execution_stream(),
            operation.device_index(),
            operation.caller_stream()
        );
#ifdef CUMESH_ENABLE_TEST_HOOKS
        test_maybe_fail(output_map_successes_before_failure, "Torch output cache map insertion");
#endif
        return tensor;
    };
    cache["face_areas"] = read_buffer(this->face_areas);
    cache["face_normals"] = read_buffer(this->face_normals);
    cache["vertex_normals"] = read_buffer(this->vertex_normals);
    cache["edges"] = read_buffer(this->edges);
    cache["boundaries"] = read_buffer(this->boundaries);
    cache["vert_is_boundary"] = read_buffer(this->vert_is_boundary);
    cache["vert_is_manifold"] = read_buffer(this->vert_is_manifold);
    cache["vert2edge"] = read_buffer(this->vert2edge);
    cache["vert2edge_cnt"] = read_buffer(this->vert2edge_cnt);
    cache["vert2edge_offset"] = read_buffer(this->vert2edge_offset);
    cache["vert2bound"] = read_buffer(this->vert2bound);
    cache["vert2bound_cnt"] = read_buffer(this->vert2bound_cnt);
    cache["vert2bound_offset"] = read_buffer(this->vert2bound_offset);
    cache["edge2face"] = read_buffer(this->edge2face);
    cache["edge2face_cnt"] = read_buffer(this->edge2face_cnt);
    cache["edge2face_offset"] = read_buffer(this->edge2face_offset);
    cache["face2edge"] = read_buffer(this->face2edge);
    cache["vert2face"] = read_buffer(this->vert2face);
    cache["vert2face_cnt"] = read_buffer(this->vert2face_cnt);
    cache["vert2face_offset"] = read_buffer(this->vert2face_offset);
    cache["manifold_face_adj"] = read_buffer(this->manifold_face_adj);
    cache["manifold_bound_adj"] = read_buffer(this->manifold_bound_adj);
    cache["conn_comp_ids"] = read_buffer(this->conn_comp_ids);
    cache["bound_conn_comp_ids"] = read_buffer(this->bound_conn_comp_ids);
    cache["loop_boundaries"] = read_buffer(this->loop_boundaries);
    cache["loop_boundaries_offset"] = read_buffer(this->loop_boundaries_offset);
    cache["vertices_map"] = read_buffer(this->vertices_map);
    cache["faces_map"] = read_buffer(this->faces_map);
    cache["edge_collapse_costs"] = read_buffer(this->edge_collapse_costs);
    cache["propagated_costs"] = read_buffer(this->propagated_costs);
    cache["atlas_chart_ids"] = read_buffer(this->atlas_chart_ids);
    cache["atlas_chart_vertex_map"] = read_buffer(this->atlas_chart_vertex_map);
    cache["atlas_chart_faces"] = read_buffer(this->atlas_chart_faces);
    cache["atlas_chart_faces_offset"] = read_buffer(this->atlas_chart_faces_offset);
    cache["atlas_chart_vertex_offset"] = read_buffer(this->atlas_chart_vertex_offset);
    cache["atlas_chart_uvs"] = read_buffer(this->atlas_chart_uvs);
    cache["atlas_chart_normal_cones"] = read_buffer(this->atlas_chart_normal_cones);
    cache["atlas_chart_adj"] = read_buffer(this->atlas_chart_adj);
    cache["atlas_chart_adj_length"] = read_buffer(this->atlas_chart_adj_length);
    cache["atlas_chart_perims"] = read_buffer(this->atlas_chart_perims);
    cache["atlas_chart_areas"] = read_buffer(this->atlas_chart_areas);
    cache["atlas_chart2edge"] = read_buffer(this->atlas_chart2edge);
    cache["atlas_chart2edge_cnt"] = read_buffer(this->atlas_chart2edge_cnt);
    cache["atlas_chart2edge_offset"] = read_buffer(this->atlas_chart2edge_offset);
    cache["temp_storage"] = read_buffer(this->temp_storage);
    cache["cub_temp_storage"] = read_buffer(this->cub_temp_storage);
    operation.bridge_to_caller();
    return cache;
}


} // namespace cumesh
