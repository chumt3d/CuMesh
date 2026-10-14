#include "admission.h"
#include "predicates.h"
#include <algorithm>
#include <climits>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <limits>
#include <stdexcept>
#include <string>

namespace cumesh {
namespace {
void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
void check_cuda(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("[CuMesh] simplify_spatial_cuda: ") +
                             cudaGetErrorString(error));
}
#define CUMESH_SPATIAL_CUDA_CHECK(expression) check_cuda(expression)
template <class T> class WorkspaceArray {
public:
  WorkspaceArray(size_t count,
                 const ExactCollapseAdmission::Allocator &allocate)
      : ptr_(nullptr) {
    require(count <= std::numeric_limits<size_t>::max() / sizeof(T),
            "[CuMesh] simplify_spatial_allocation_overflow");
    if (count)
      ptr_ = static_cast<T *>(allocate(count * sizeof(T)));
  }
  T *get() const { return ptr_; }

private:
  T *ptr_;
};
constexpr int query_tile =
    65536; // bounds per-launch CUDA local predicate storage
constexpr int query_threads = 128;
struct Box {
  float3 low, high;
};
struct MergeBox {
  __host__ __device__ Box operator()(Box a, Box b) const {
    return {make_float3(fminf(a.low.x, b.low.x), fminf(a.low.y, b.low.y),
                        fminf(a.low.z, b.low.z)),
            make_float3(fmaxf(a.high.x, b.high.x), fmaxf(a.high.y, b.high.y),
                        fmaxf(a.high.z, b.high.z))};
  }
};
struct Triangle {
  exact_geometry::Point p[3];
};
struct Surface {
  const float3 *vertices;
  const int3 *faces;
  const uint64_t *edges;
  const float3 *proposals;
  const int *owner;
  bool proposed;
};
__device__ Triangle triangle(Surface s, int face) {
  int3 f = s.faces[face];
  int ids[3] = {f.x, f.y, f.z};
  int edge = s.proposed ? s.owner[face] : -1;
  int e0 = -1, e1 = -1;
  if (edge >= 0) {
    uint64_t e = s.edges[edge];
    e0 = int(e >> 32);
    e1 = int(e & 0xffffffff);
  }
  Triangle out;
  for (int i = 0; i < 3; ++i) {
    float3 p =
        (ids[i] == e0 || ids[i] == e1) ? s.proposals[edge] : s.vertices[ids[i]];
    out.p[i] = {p.x, p.y, p.z};
  }
  return out;
}
__device__ Box bounds(Triangle t) {
  Box b{{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
  for (int i = 0; i < 3; ++i) {
    auto p = t.p[i];
    b = MergeBox{}(b, {make_float3(p.x, p.y, p.z), make_float3(p.x, p.y, p.z)});
  }
  b.low = {nextafterf(b.low.x, -INFINITY), nextafterf(b.low.y, -INFINITY),
           nextafterf(b.low.z, -INFINITY)};
  b.high = {nextafterf(b.high.x, INFINITY), nextafterf(b.high.y, INFINITY),
            nextafterf(b.high.z, INFINITY)};
  return b;
}
__device__ bool overlaps(Box a, Box b) {
  return a.low.x <= b.high.x && b.low.x <= a.high.x && a.low.y <= b.high.y &&
         b.low.y <= a.high.y && a.low.z <= b.high.z && b.low.z <= a.high.z;
}
__device__ int contact(Triangle a, Triangle b) {
  // Exact-position equivalence is the audit's canonical contact identity,
  // including proposed points coincident with another existing vertex.
  exact_geometry::Point p[6] = {a.p[0], a.p[1], a.p[2], b.p[0], b.p[1], b.p[2]};
  int g[6];
  for (int i = 0; i < 6; ++i) {
    g[i] = i;
    for (int j = 0; j < i; ++j)
      if (p[i].x == p[j].x && p[i].y == p[j].y && p[i].z == p[j].z) {
        g[i] = g[j];
        break;
      }
  }
  return exact_geometry::pair(p[0], p[1], p[2], p[3], p[4], p[5], g[0], g[1],
                              g[2], g[3], g[4], g[5]) &
         3;
}
__device__ unsigned spread(unsigned x) {
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  return (x | (x << 2)) & 0x09249249;
}
__device__ unsigned coordinate(double p, double low, double high) {
  double q = (p - low) / (high - low);
  // Infinite outward bounds or a flat axis only affect spatial ordering.
  // Closed AABB traversal remains complete even with identical Morton codes.
  if (!isfinite(q))
    q = 0.5;
  return unsigned(fmin(1023.0, fmax(0.0, q * 1024.0)));
}
__global__ void ids_kernel(int *ids, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    ids[i] = i;
}
__global__ void boxes_kernel(Surface s, const int *ids, Box *boxes, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    boxes[i] = bounds(triangle(s, ids[i]));
}
__global__ void keys_kernel(const Box *boxes, const Box *scene, const int *ids,
                            uint64_t *keys, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Box b = boxes[i], w = *scene;
  unsigned x = coordinate((double(b.low.x) + b.high.x) * 0.5, w.low.x,
                          w.high.x),
           y = coordinate((double(b.low.y) + b.high.y) * 0.5, w.low.y,
                          w.high.y),
           z = coordinate((double(b.low.z) + b.high.z) * 0.5, w.low.z,
                          w.high.z);
  keys[i] = (uint64_t(spread(x) | (spread(y) << 1) | (spread(z) << 2)) << 32) |
            unsigned(ids[i]);
}
struct TreeView {
  Box *boxes;
  int2 *children;
  int *parent;
  int *ready;
  int *faces;
  int n;
};
__device__ int prefix(const uint64_t *keys, int n, int i, int j) {
  return j < 0 || j >= n ? -1 : __clzll(keys[i] ^ keys[j]);
}
// Independent Karras radix topology on unique (Morton, face-ID) keys.
// https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees
__global__ void topology_kernel(TreeView tree, const uint64_t *keys) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tree.n - 1)
    return;
  int direction =
      prefix(keys, tree.n, i, i + 1) > prefix(keys, tree.n, i, i - 1) ? 1 : -1;
  int minimum = prefix(keys, tree.n, i, i - direction), maximum = 2;
  while (prefix(keys, tree.n, i, i + maximum * direction) > minimum)
    maximum *= 2;
  int length = 0;
  for (int step = maximum / 2; step >= 1; step /= 2)
    if (prefix(keys, tree.n, i, i + (length + step) * direction) > minimum)
      length += step;
  int j = i + length * direction, common = prefix(keys, tree.n, i, j),
      split = 0;
  for (int step = (length + 1) / 2; step >= 1; step = (step + 1) / 2) {
    if (split + step < length &&
        prefix(keys, tree.n, i, i + (split + step) * direction) > common)
      split += step;
    if (step == 1)
      break;
  }
  int pivot = i + split * direction + (direction < 0 ? -1 : 0),
      first = min(i, j), last = max(i, j);
  int left = pivot == first ? tree.n - 1 + pivot : pivot,
      right = pivot + 1 == last ? tree.n + pivot : pivot + 1;
  tree.children[i] = make_int2(left, right);
  tree.parent[left] = i;
  tree.parent[right] = i;
}
__global__ void refit_kernel(TreeView tree) {
  int leaf = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf >= tree.n)
    return;
  int node = tree.n - 1 + leaf;
  while (tree.parent[node] >= 0) {
    int parent = tree.parent[node];
    cuda::atomic_ref<int, cuda::thread_scope_device> arrival(
        tree.ready[parent]);
    if (arrival.fetch_add(1, cuda::memory_order_acq_rel) == 0)
      return;
    int2 kids = tree.children[parent];
    tree.boxes[parent] = MergeBox{}(tree.boxes[kids.x], tree.boxes[kids.y]);
    node = parent;
  }
}
__global__ void mark_owners_kernel(Surface s, const int *adjacency,
                                   const int *offsets, const float *costs,
                                   int ecount, int *owner, int *error) {
  int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= ecount || !isfinite(costs[edge]))
    return;
  uint64_t e = s.edges[edge];
  int ends[2] = {int(e >> 32), int(e & 0xffffffff)};
  for (int end : ends)
    for (int i = offsets[end]; i < offsets[end + 1]; ++i) {
      int old = atomicCAS(owner + adjacency[i], -1, edge);
      if (old != -1 && old != edge)
        atomicExch(error, 1);
    }
}
__global__ void fan_flags_kernel(Surface s, uint8_t *flags, int n) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= n)
    return;
  int owner = s.owner[face];
  flags[face] = 0;
  if (owner < 0)
    return;
  uint64_t e = s.edges[owner];
  int a = int(e >> 32), b = int(e & 0xffffffff);
  int3 f = s.faces[face];
  bool has_a = f.x == a || f.y == a || f.z == a,
       has_b = f.x == b || f.y == b || f.z == b;
  flags[face] = !(has_a && has_b);
}
enum QueryKind { audit_current, new_old, within_fan, competing_fans };
struct QueryCounts {
  unsigned long long nodes, leaves, queries;
  unsigned int max_nodes;
  unsigned long long undecided;
  int error;
};
__global__ void query_kernel(Surface source, const int *source_ids,
                             int source_count, Surface target, TreeView tree,
                             QueryKind kind, const float *costs,
                             const int *base_refused, int *refused,
                             QueryCounts *counts, int offset) {
  int query = offset + blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= source_count || query >= offset + query_tile)
    return;
  int face = source_ids[query],
      owner = kind == audit_current ? -1 : source.owner[face];
  Triangle t = triangle(source, face);
  Box box = bounds(t);
  int stack[66], size = 1;
  stack[0] = 0;
  unsigned long long visits = 0, leaves = 0;
  while (size) {
    int node = stack[--size];
    ++visits;
    if (!overlaps(box, tree.boxes[node]))
      continue;
    if (node < tree.n - 1) {
      if (size + 2 > 66) {
        atomicExch(&counts->error, 1);
        break;
      }
      int2 k = tree.children[node];
      stack[size++] = k.y;
      stack[size++] = k.x;
      continue;
    }
    int other = tree.faces[node - (tree.n - 1)];
    if (kind == audit_current && other <= face)
      continue;
    int other_owner = kind == audit_current ? -1 : source.owner[other];
    if (kind == new_old && other_owner == owner)
      continue;
    if (kind == within_fan && (other <= face || other_owner != owner))
      continue;
    if (kind == competing_fans &&
        (other <= face || other_owner == owner || base_refused[owner] ||
         base_refused[other_owner]))
      continue;
    ++leaves;
    Triangle target_triangle = triangle(target, other);
    int result = contact(t, target_triangle);
    if (kind == new_old || kind == competing_fans) {
      // A newly coincident contraction point could pinch an otherwise clean
      // vertex link after exact welding. Refuse it even when contact is legal.
      float3 proposal = source.proposals[owner];
      for (int i = 0; i < 3; ++i)
        if (proposal.x == target_triangle.p[i].x &&
            proposal.y == target_triangle.p[i].y &&
            proposal.z == target_triangle.p[i].z)
          result = 1;
    }
    if (result == 0)
      continue;
    if (result == 2)
      atomicAdd(&counts->undecided, 1ULL);
    if (kind == audit_current) {
      atomicExch(&counts->error, result == 2 ? 3 : 2);
      continue;
    }
    int loser = owner;
    if (kind == competing_fans) {
      float a = costs[owner], b = costs[other_owner];
      loser = (a > b || (a == b && owner > other_owner)) ? owner : other_owner;
    }
    atomicExch(refused + loser, 1);
  }
  atomicAdd(&counts->nodes, visits);
  atomicAdd(&counts->leaves, leaves);
  atomicAdd(&counts->queries, 1ULL);
  atomicMax(&counts->max_nodes, unsigned(visits));
}
__global__ void apply_refusal_kernel(float *costs, const int *refused,
                                     int *ever_refused, int n, int *accepted) {
  int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= n || !isfinite(costs[edge]))
    return;
  if (refused[edge]) {
    costs[edge] = INFINITY;
    ever_refused[edge] = 1;
  } else
    atomicAdd(accepted, 1);
}
struct Tree {
  WorkspaceArray<Box> boxes;
  WorkspaceArray<int2> children;
  WorkspaceArray<int> parent, ready, faces;
  Tree(int n, const ExactCollapseAdmission::Allocator &allocate)
      : boxes(2 * size_t(n) - 1, allocate), children(n - 1, allocate),
        parent(2 * size_t(n) - 1, allocate), ready(n - 1, allocate),
        faces(n, allocate) {}
  TreeView view(int n) {
    return {boxes.get(), children.get(), parent.get(),
            ready.get(), faces.get(),    n};
  }
};
} // namespace
struct ExactCollapseAdmission::Impl {
  std::map<std::string, uint64_t> statistics;
  CollapseSurface mesh;
  cudaStream_t stream;
  int F, E;
  Surface old, new_surface;
  Tree current, proposed;
  WorkspaceArray<int> ids, fan_ids, owner, refused, base_refused, ever_refused,
      count;
  WorkspaceArray<uint8_t> flags;
  WorkspaceArray<uint64_t> keys, sorted_keys;
  WorkspaceArray<Box> scene;
  WorkspaceArray<QueryCounts> counts;
  std::unique_ptr<WorkspaceArray<char>> workspace;
  size_t workspace_bytes = 0;
  Impl(CollapseSurface m, const float3 *p, cudaStream_t s,
       const ExactCollapseAdmission::Allocator &allocate)
      : mesh(m), stream(s), F(m.face_count), E(m.edge_count),
        old{m.vertices, m.faces, m.edges, p, nullptr, false}, new_surface(old),
        current(F, allocate), proposed(F, allocate), ids(F, allocate),
        fan_ids(F, allocate), owner(F, allocate), refused(E, allocate),
        base_refused(E, allocate), ever_refused(E, allocate),
        count(1, allocate), flags(F, allocate), keys(F, allocate),
        sorted_keys(F, allocate), scene(1, allocate), counts(1, allocate) {
    new_surface.owner = owner.get();
    new_surface.proposed = true;
    size_t bytes = 0;
    Box identity{{INFINITY, INFINITY, INFINITY},
                 {-INFINITY, -INFINITY, -INFINITY}};
    CUMESH_SPATIAL_CUDA_CHECK(cub::DeviceReduce::Reduce(
        nullptr, bytes, current.boxes.get(), scene.get(), F, MergeBox{},
        identity, stream));
    workspace_bytes = std::max(workspace_bytes, bytes);
    CUMESH_SPATIAL_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, keys.get(), sorted_keys.get(), ids.get(),
        current.faces.get(), F, 0, 64, stream));
    workspace_bytes = std::max(workspace_bytes, bytes);
    CUMESH_SPATIAL_CUDA_CHECK(
        cub::DeviceSelect::Flagged(nullptr, bytes, ids.get(), flags.get(),
                                   fan_ids.get(), count.get(), F, stream));
    workspace_bytes = std::max(workspace_bytes, bytes);
    workspace =
        std::make_unique<WorkspaceArray<char>>(workspace_bytes, allocate);
    CUMESH_SPATIAL_CUDA_CHECK(
        cudaMemsetAsync(ever_refused.get(), 0, E * sizeof(int), stream));
    ids_kernel<<<(F + 255) / 256, 256, 0, stream>>>(ids.get(), F);
    CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    build(current, F, ids.get(), old);
    CUMESH_SPATIAL_CUDA_CHECK(
        cudaMemsetAsync(counts.get(), 0, sizeof(QueryCounts), stream));
    query(old, ids.get(), F, old, current.view(F), audit_current, nullptr,
          nullptr);
    check("input");
  }
  void build(Tree &t, int n, const int *input, Surface surface) {
    TreeView v = t.view(n);
    Box identity{{INFINITY, INFINITY, INFINITY},
                 {-INFINITY, -INFINITY, -INFINITY}};
    CUMESH_SPATIAL_CUDA_CHECK(cudaMemsetAsync(
        v.parent, 0xff, (2 * size_t(n) - 1) * sizeof(int), stream));
    if (n > 1)
      CUMESH_SPATIAL_CUDA_CHECK(
          cudaMemsetAsync(v.ready, 0, (n - 1) * sizeof(int), stream));
    boxes_kernel<<<(n + 255) / 256, 256, 0, stream>>>(surface, input, v.boxes,
                                                      n);
    CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    size_t bytes = workspace_bytes;
    CUMESH_SPATIAL_CUDA_CHECK(
        cub::DeviceReduce::Reduce(workspace->get(), bytes, v.boxes, scene.get(),
                                  n, MergeBox{}, identity, stream));
    keys_kernel<<<(n + 255) / 256, 256, 0, stream>>>(v.boxes, scene.get(),
                                                     input, keys.get(), n);
    CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    bytes = workspace_bytes;
    CUMESH_SPATIAL_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        workspace->get(), bytes, keys.get(), sorted_keys.get(), input, v.faces,
        n, 0, 64, stream));
    boxes_kernel<<<(n + 255) / 256, 256, 0, stream>>>(surface, v.faces,
                                                      v.boxes + n - 1, n);
    CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    if (n > 1) {
      topology_kernel<<<(n + 254) / 256, 256, 0, stream>>>(v,
                                                           sorted_keys.get());
      CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
      refit_kernel<<<(n + 255) / 256, 256, 0, stream>>>(v);
      CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    }
  }
  void query(Surface source, const int *input, int n, Surface target,
             TreeView tree, QueryKind kind, const int *base, int *output) {
    for (int offset = 0; offset < n; offset += query_tile) {
      int tile = std::min(query_tile, n - offset);
      query_kernel<<<(tile + query_threads - 1) / query_threads, query_threads,
                     0, stream>>>(source, input, n, target, tree, kind,
                                  mesh.costs, base, output, counts.get(),
                                  offset);
      CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    }
  }
  void check(const char *stage) {
    QueryCounts c{};
    CUMESH_SPATIAL_CUDA_CHECK(cudaMemcpyAsync(&c, counts.get(), sizeof(c),
                                              cudaMemcpyDeviceToHost, stream));
    CUMESH_SPATIAL_CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::string prefix =
        std::string(stage) == "input" ? "input" : "proposal";
    statistics[prefix + "NodeVisits"] += c.nodes;
    statistics[prefix + "LeafTests"] += c.leaves;
    statistics[prefix + "Queries"] += c.queries;
    statistics["maximumNodeVisitsPerQuery"] = std::max(
        statistics["maximumNodeVisitsPerQuery"], uint64_t(c.max_nodes));
    statistics["undecidedRefusals"] += c.undecided;
    require(c.error != 1,
            std::string("[CuMesh] simplify_spatial_bvh_depth_overflow stage=") +
                stage);
    require(c.error != 2, "[CuMesh] simplify_spatial_input_intersection");
    require(c.error != 3, "[CuMesh] simplify_spatial_input_undecided");
  }
  int filter() {
    CUMESH_SPATIAL_CUDA_CHECK(
        cudaMemsetAsync(owner.get(), 0xff, F * sizeof(int), stream));
    CUMESH_SPATIAL_CUDA_CHECK(
        cudaMemsetAsync(count.get(), 0, sizeof(int), stream));
    mark_owners_kernel<<<(E + 255) / 256, 256, 0, stream>>>(
        old, mesh.adjacency, mesh.offsets, mesh.costs, E, owner.get(),
        count.get());
    CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    int bad = 0;
    CUMESH_SPATIAL_CUDA_CHECK(cudaMemcpyAsync(&bad, count.get(), sizeof(int),
                                              cudaMemcpyDeviceToHost, stream));
    CUMESH_SPATIAL_CUDA_CHECK(cudaStreamSynchronize(stream));
    require(!bad, "[CuMesh] simplify_spatial_ownership_conflict");
    fan_flags_kernel<<<(F + 255) / 256, 256, 0, stream>>>(new_surface,
                                                          flags.get(), F);
    CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    size_t bytes = workspace_bytes;
    CUMESH_SPATIAL_CUDA_CHECK(cub::DeviceSelect::Flagged(
        workspace->get(), bytes, ids.get(), flags.get(), fan_ids.get(),
        count.get(), F, stream));
    int fans = 0;
    CUMESH_SPATIAL_CUDA_CHECK(cudaMemcpyAsync(&fans, count.get(), sizeof(int),
                                              cudaMemcpyDeviceToHost, stream));
    CUMESH_SPATIAL_CUDA_CHECK(cudaStreamSynchronize(stream));
    CUMESH_SPATIAL_CUDA_CHECK(
        cudaMemsetAsync(base_refused.get(), 0, E * sizeof(int), stream));
    CUMESH_SPATIAL_CUDA_CHECK(
        cudaMemsetAsync(counts.get(), 0, sizeof(QueryCounts), stream));
    if (fans) {
      build(proposed, fans, fan_ids.get(), new_surface);
      query(new_surface, fan_ids.get(), fans, old, current.view(F), new_old,
            nullptr, base_refused.get());
      query(new_surface, fan_ids.get(), fans, new_surface, proposed.view(fans),
            within_fan, nullptr, base_refused.get());
      CUMESH_SPATIAL_CUDA_CHECK(
          cudaMemcpyAsync(refused.get(), base_refused.get(), E * sizeof(int),
                          cudaMemcpyDeviceToDevice, stream));
      query(new_surface, fan_ids.get(), fans, new_surface, proposed.view(fans),
            competing_fans, base_refused.get(), refused.get());
    } else
      CUMESH_SPATIAL_CUDA_CHECK(
          cudaMemsetAsync(refused.get(), 0, E * sizeof(int), stream));
    check("proposal");
    CUMESH_SPATIAL_CUDA_CHECK(
        cudaMemsetAsync(count.get(), 0, sizeof(int), stream));
    apply_refusal_kernel<<<(E + 255) / 256, 256, 0, stream>>>(
        mesh.costs, refused.get(), ever_refused.get(), E, count.get());
    CUMESH_SPATIAL_CUDA_CHECK(cudaGetLastError());
    int accepted = 0;
    CUMESH_SPATIAL_CUDA_CHECK(cudaMemcpyAsync(
        &accepted, count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUMESH_SPATIAL_CUDA_CHECK(cudaStreamSynchronize(stream));
    statistics["acceptedCollapses"] = accepted;
    ++statistics["selectionBatches"];
    return accepted;
  }
};
ExactCollapseAdmission::ExactCollapseAdmission(CollapseSurface m,
                                               const float3 *p, cudaStream_t s,
                                               const Allocator &allocate) {
  require(m.face_count > 0 && m.face_count <= INT_MAX / 2 && m.edge_count >= 0,
          "[CuMesh] simplify_spatial_index_overflow");
  impl_ = std::make_unique<Impl>(m, p, s, allocate);
}
ExactCollapseAdmission::~ExactCollapseAdmission() = default;
int ExactCollapseAdmission::filter_selected() { return impl_->filter(); }
const int *ExactCollapseAdmission::refused_edges() const {
  return impl_->ever_refused.get();
}
std::map<std::string, uint64_t> ExactCollapseAdmission::statistics() const {
  return impl_->statistics;
}
} // namespace cumesh
