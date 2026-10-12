// Copyright (c) 2025 Jianfeng XIANG. MIT; see ../../LICENSE.
// Narrow bindings from CuMesh 12289e1062f0603f2f0d0771b02e1395d247f26f/src/ext.cpp.
#include <torch/extension.h>
#include "hash/api.h"
#include "remesh/api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hashmap_lookup_3d_cuda", &cumesh::hashmap_lookup_3d_cuda);
    m.def("hashmap_insert_3d_idx_as_val_cuda", &cumesh::hashmap_insert_3d_idx_as_val_cuda);
    m.def("get_sparse_voxel_grid_active_vertices", &cumesh::get_sparse_voxel_grid_active_vertices);
    m.def("simple_dual_contour", &cumesh::simple_dual_contour);
}
