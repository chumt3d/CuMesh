#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
namespace cumesh {
struct CollapseSurface {
  const float3 *vertices;
  const int3 *faces;
  const uint64_t *edges;
  const int *adjacency;
  const int *offsets;
  float *costs;
  int face_count, edge_count;
};
// The caller supplies tracked, call-owned CuMesh allocation. This dedicated
// query translation unit depends on CUDA, CUB and the geometry contract only.
class ExactCollapseAdmission {
public:
  using Allocator = std::function<void *(size_t)>;
  ExactCollapseAdmission(CollapseSurface mesh, const float3 *proposals,
                         cudaStream_t stream, const Allocator &allocate,
                         double max_seconds,
                         const std::function<bool()> &cancelled);
  ~ExactCollapseAdmission();
  ExactCollapseAdmission(const ExactCollapseAdmission &) = delete;
  ExactCollapseAdmission &operator=(const ExactCollapseAdmission &) = delete;
  int filter_selected();
  const int *refused_edges() const;
  std::map<std::string, uint64_t> statistics() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace cumesh
