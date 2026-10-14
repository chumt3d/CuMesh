#include <torch/extension.h>
#include "hash/api.h"
#include "cumesh.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<cumesh::ReleaseCompletionToken, std::shared_ptr<cumesh::ReleaseCompletionToken>>(
        m,
        "ReleaseCompletionToken"
    )
        .def_property_readonly("device_index", &cumesh::ReleaseCompletionToken::device_index)
        .def_property_readonly("stream_id", &cumesh::ReleaseCompletionToken::stream_id)
        .def_property_readonly("owner_id", &cumesh::ReleaseCompletionToken::owner_id)
        .def_property_readonly("generation", &cumesh::ReleaseCompletionToken::generation)
        .def_property_readonly("is_complete", &cumesh::ReleaseCompletionToken::is_complete)
        .def_property_readonly("status", &cumesh::ReleaseCompletionToken::status)
        .def("wait_on_current_stream", &cumesh::ReleaseCompletionToken::wait_on_current_stream)
        .def("synchronize", &cumesh::ReleaseCompletionToken::synchronize);

    py::class_<cumesh::ReleaseResult>(m, "ReleaseResult")
        .def_readonly("status", &cumesh::ReleaseResult::status)
        .def_readonly("buffers_released", &cumesh::ReleaseResult::buffers_released)
        .def_readonly("device_index", &cumesh::ReleaseResult::device_index)
        .def_readonly("stream_id", &cumesh::ReleaseResult::stream_id)
        .def_readonly("stream_synchronize_count", &cumesh::ReleaseResult::stream_synchronize_count)
        .def_readonly("completion_token", &cumesh::ReleaseResult::completion_token)
        .def_readonly("quarantined_buffer_count", &cumesh::ReleaseResult::quarantined_buffer_count)
        .def_readonly("error_code", &cumesh::ReleaseResult::error_code)
        .def_readonly("error_message", &cumesh::ReleaseResult::error_message)
        .def_readonly("owner_id", &cumesh::ReleaseResult::owner_id)
        .def_readonly("generation", &cumesh::ReleaseResult::generation)
        .def_readonly("cleanup_error_code", &cumesh::ReleaseResult::cleanup_error_code)
        .def_readonly("cleanup_error_message", &cumesh::ReleaseResult::cleanup_error_message)
        .def_readonly("primary_error_code", &cumesh::ReleaseResult::primary_error_code)
        .def_readonly("primary_error_message", &cumesh::ReleaseResult::primary_error_message);

    // Hash functions
    m.def("hashmap_insert_cuda", &cumesh::hashmap_insert_cuda);
    m.def("hashmap_lookup_cuda", &cumesh::hashmap_lookup_cuda);
    m.def("hashmap_insert_3d_cuda", &cumesh::hashmap_insert_3d_cuda);
    m.def("hashmap_lookup_3d_cuda", &cumesh::hashmap_lookup_3d_cuda);
    m.def("hashmap_insert_3d_idx_as_val_cuda", &cumesh::hashmap_insert_3d_idx_as_val_cuda);

    /* CUMESH */
    py::class_<cumesh::CuMesh>(m, "CuMesh")
        .def(py::init<>())
        .def("num_vertices", &cumesh::CuMesh::num_vertices)
        .def("num_faces", &cumesh::CuMesh::num_faces)
        .def("num_edges", &cumesh::CuMesh::num_edges)
        .def("num_boundaries", &cumesh::CuMesh::num_boundaries)
        .def("num_conneted_components", &cumesh::CuMesh::num_conneted_components)
        .def("num_boundary_conneted_components", &cumesh::CuMesh::num_boundary_conneted_components)
        .def("num_boundary_loops", &cumesh::CuMesh::num_boundary_loops)
        .def("swap", &cumesh::CuMesh::swap)
        .def("release_async", &cumesh::CuMesh::release_async, py::arg("synchronize") = false)
        .def("allocated_buffer_count", &cumesh::CuMesh::allocated_buffer_count)
        .def_property_readonly("owner_id", &cumesh::CuMesh::owner_id)
        .def_property_readonly("generation", &cumesh::CuMesh::generation)
        .def_property_readonly("device_index", &cumesh::CuMesh::device_index)
        .def_static("destructor_release_error_count", &cumesh::CuMesh::destructor_release_error_count)
        .def_static("destructor_device_synchronize_count", &cumesh::CuMesh::destructor_device_synchronize_count)
        .def_static("quarantined_cuda_allocation_count", &cumesh::CuMesh::quarantined_cuda_allocation_count)
        .def_static("retry_quarantined_cuda_allocations", &cumesh::CuMesh::retry_quarantined_cuda_allocations)
        .def("clear_cache", &cumesh::CuMesh::clear_cache)
        .def("init", &cumesh::CuMesh::init)
        .def("read", &cumesh::CuMesh::read)
        .def("read_face_normals", &cumesh::CuMesh::read_face_normals)
        .def("read_vertex_normals", &cumesh::CuMesh::read_vertex_normals)
        .def("read_edges", &cumesh::CuMesh::read_edges)
        .def("read_boundaries", &cumesh::CuMesh::read_boundaries)
        .def("read_manifold_face_adjacency", &cumesh::CuMesh::read_manifold_face_adjacency)
        .def("read_manifold_boundary_adjacency", &cumesh::CuMesh::read_manifold_boundary_adjacency)
        .def("read_connected_components", &cumesh::CuMesh::read_connected_components)
        .def("read_boundary_connected_components", &cumesh::CuMesh::read_boundary_connected_components)
        .def("read_boundary_loops", &cumesh::CuMesh::read_boundary_loops)
        .def("read_all_cache", &cumesh::CuMesh::read_all_cache)
        .def("compute_face_normals", &cumesh::CuMesh::compute_face_normals)
        .def("compute_vertex_normals", &cumesh::CuMesh::compute_vertex_normals)
        .def("get_vertex_face_adjacency", &cumesh::CuMesh::get_vertex_face_adjacency)
        .def("get_edges", &cumesh::CuMesh::get_edges)
        .def("get_edge_face_adjacency", &cumesh::CuMesh::get_edge_face_adjacency)
        .def("get_vertex_edge_adjacency", &cumesh::CuMesh::get_vertex_edge_adjacency)
        .def("get_boundary_info", &cumesh::CuMesh::get_boundary_info)
        .def("get_vertex_boundary_adjacency", &cumesh::CuMesh::get_vertex_boundary_adjacency)
        .def("get_vertex_is_manifold", &cumesh::CuMesh::get_vertex_is_manifold)
        .def("get_manifold_face_adjacency", &cumesh::CuMesh::get_manifold_face_adjacency)
        .def("get_manifold_boundary_adjacency", &cumesh::CuMesh::get_manifold_boundary_adjacency)
        .def("get_connected_components", &cumesh::CuMesh::get_connected_components)
        .def("get_boundary_connected_components", &cumesh::CuMesh::get_boundary_connected_components)
        .def("get_boundary_loops", &cumesh::CuMesh::get_boundary_loops)
        .def("remove_faces", &cumesh::CuMesh::remove_faces)
        .def("remove_unreferenced_vertices", &cumesh::CuMesh::remove_unreferenced_vertices)
        .def("remove_duplicate_faces", &cumesh::CuMesh::remove_duplicate_faces)
        .def("remove_degenerate_faces", &cumesh::CuMesh::remove_degenerate_faces)
        .def("fill_holes", &cumesh::CuMesh::fill_holes)
        .def("repair_non_manifold_edges", &cumesh::CuMesh::repair_non_manifold_edges)
        .def("remove_non_manifold_faces", &cumesh::CuMesh::remove_non_manifold_faces)
        .def("remove_small_connected_components", &cumesh::CuMesh::remove_small_connected_components)
        .def("unify_face_orientations", &cumesh::CuMesh::unify_face_orientations)
        .def("simplify_step", &cumesh::CuMesh::simplify_step)
        .def("initial_simplification_quadrics", &cumesh::CuMesh::initial_simplification_quadrics)
        .def("simplify_step_accumulated", &cumesh::CuMesh::simplify_step_accumulated)
        .def("simplification_admission_statistics", &cumesh::CuMesh::simplification_admission_statistics)
        .def("compute_charts", &cumesh::CuMesh::compute_charts)
        .def("read_atlas_charts", &cumesh::CuMesh::read_atlas_charts);

}
