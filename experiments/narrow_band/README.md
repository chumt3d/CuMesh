# CuMesh narrow-band Make-solid pilot

Experiment source only. This does not alter `cumesh`, the production setup,
submodule pins, or production dependencies. No CUDA compilation or geometry
acceptance has been performed locally. The integration owner runs the remote
pilot and accepts its finite-sheet, nested-shell, and close-gap controls.

## Build on the approved remote CUDA substrate

Start from this fork branch/commit, with existing PyTorch CUDA >=2.5 (unsigned
Torch integer dtypes), matching CUDA/NVCC headers, setuptools, and Ninja. Runtime
requires the controller's exact `warp-lang==1.13.0` and `tqdm`. Install these only
through the controller's approved dependency path and Socket Firewall; this
package deliberately does not choose or upgrade Torch/CUDA. No submodules are
required by the experiment. No cubvh, OptiX, xatlas, or nvdiffrast is built.

From `experiments/narrow_band`, build only this distribution:

```bash
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 sfw uv pip install --no-deps --no-build-isolation .
```

8.9 is the L4 architecture; select the actual approved GPU's architecture when
using another GPU. This compiles only `_src/bindings.cpp`, `_src/hash/hash.cu`,
`_src/remesh/simple_dual_contour.cu`, and `_src/remesh/svox2vert.cu`. Relative
headers close over `_src/hash/{api.h,hash.cuh}`, `_src/remesh/api.h`, `_src/utils.h`;
external headers come only from PyTorch/CUDA (including CUDA CUB). `dtypes.cuh`
and every production CuMesh/atlas/cleanup/simplification source are unnecessary.

## Interface and approved settings

```python
from cumesh_remesh_experiment import WarpUnsignedDistanceBVH, remesh_narrow_band_dc

# V: contiguous CUDA float32(N,3), F: contiguous same-device int32(M,3).
# Root performs and records the reversible longest-axis scaling to 100 mm.
# center is the float32 CUDA domain center; padded cubic domain_scale=105 mm.
# Use the exact padded domain scale if the controller computes another value.
resolution = 512
wall_mm = 1.0  # run 2.0 separately with the same source/domain; no hidden change.
band = (wall_mm / 2.0) * resolution / domain_scale
bvh = WarpUnsignedDistanceBVH(V, F, max_distance=4.0 * domain_scale)
diagnostics = {}
V_out, F_out = remesh_narrow_band_dc(
    V, F, center, domain_scale, resolution, band=band, project_back=0, bvh=bvh,
    diagnostics=diagnostics,
)
# Synchronize/read back under the controller's evidence boundary before releasing
# bvh or its V/F buffers; inverse-transform V_out to the input units afterwards.
```

Coordinates and `scale` must use the same units. `eps=band*scale/resolution`
is the source isosurface offset, so `wall_mm/2` sets the UDF offset. Whether the
returned shell has the useful requested wall thickness/topology remains an
experimental question. The UDF has no interior sign: a closed shell can produce
both inward and outward offsets. `project_back` is explicitly restricted to 0.
The query adapter supports optional source-compatible barycentrics independently.

All allocations/native kernels execute on the current Torch CUDA stream and
mesh device. Input tensors must already be ready there; a caller using producer
streams must establish `wait_stream` ordering. The BVH owns references to its
immutable input tensors. Output shape is `(N_new,3)` float32 vertices and
`(M_new,3)` int32 triangles; an empty isosurface returns empty tensors without
zero-block CUDA launches. Input validation rejects nonfinite vertices/centers,
out-of-range face indices, CPU/mixed-device buffers, and invalid scalar settings.

`unsigned_distance(positions, return_uvw=False)` preserves the query prefix and
returns float32 distances, int64 face IDs, and optional float32 `(...,3)` uvw.
Warp launches use caller-configurable `query_batch_size` (default 262144), preserving
query order. A face must exist strictly within explicit `max_distance`; misses
raise, so the remesher cannot silently consume truncated or invalid UDF values.
One scalar missed-query synchronization per query is intentional prototype cost.
The bound `4*domain_scale` exceeds the distance between all points and mesh faces
inside the declared cubic domain. The controller must ensure domain containment.

## Source mapping and deviations

Upstream authority: MIT CuMesh
`12289e1062f0603f2f0d0771b02e1395d247f26f`, starting owned-fork base
`a8d15fb8b2f504d86d56e8da9317633933a2119f`.

| Upstream | Experiment | Preserved behavior / change |
| --- | --- | --- |
| `src/hash/{hash.cu,hash.cuh,api.h}` | `_src/hash/` | Byte-identical hash kernels and APIs, including hash/sentinel/order semantics. |
| `src/remesh/simple_dual_contour.cu` | `_src/remesh/simple_dual_contour.cu` | Byte-identical edge traversal, interpolation, average intersection, fallback voxel center, flags. |
| `src/remesh/{api.h,svox2vert.cu}` | `_src/remesh/` | API/kernels identical; host wrapper adds actual CUB workspace admission before allocation. Original scan/order/synchronizations preserved. |
| `src/utils.h` | `_src/utils.h` | Byte-identical; includes upstream `CUDA_CHECK` and buffer helper. |
| `src/ext.cpp` | `_src/bindings.cpp` | Four required bindings only; no CuMesh class or unrelated exports. |
| `cumesh/remeshing.py` | `remeshing.py` | Refinement0.87, UDF-eps, resolution/base-grid ordering, hash sizing, contour and topology/split arithmetic preserved. Remove eager cuBVH import/default construction; use injected BVH. Device-local topology tables replace first-device globals. Empty-stage returns prevent zero-block launches. Per-triangle `torch.cross(dim=1)` makes the intended normal operation explicit when exactly 3 quads exist (upstream's unspecified dimension is ambiguous there). Resource guards fail before dangerous allocations. |
| `cumesh/bvh.py` | `warp_bvh.py` | Approved replacement with Warp Mesh/no-sign closest-point/eval-position, explicit float32/same-device/current-stream handling; original cuBVH CPU mesh transfer removed. No bitwise backend parity claim. |

Warp source authority was inspected from installed 1.13.0
`warp/{config.py,_src/torch.py,_src/types.py,_src/builtins.py,native/mesh.h}`.
The equivalent public source is [NVIDIA Warp v1.13.0](https://github.com/NVIDIA/warp/tree/v1.13.0).
`mesh_eval_position` is `p*u+q*v+r*(1-u-v)`, matching the adapter uvw.
Use LBVH explicitly: the 1.13 experimental `cubql` constructor supports ray queries
only and silently misses point queries. Warp ignores sliver/degenerate faces
when `length(cross(e0,e1))/(dot(e0,e0)+dot(e1,e1)+dot(e2,e2))<1e-6`.
Equal-distance tie selection, floating-point differences, and ignored slivers
can differ from cubvh; these are backend limitations requiring remote controls,
not asserted parity. This module does not copy Warp implementation code.

## Memory and evidence limits

The unchanged source stores all active voxel coordinates, corners, fields and
topology; batched distance launches do not bound those full arrays. Resource
admission uses counts A(active voxels), V(active vertices; bound 8A), and E(crossed
edges; bound 3A) with explicit dtype/shape byte sums in `resources.py`. It runs
before 8-child expansion, corner/contour allocation, and topology neighbors.
It counts driver-free bytes only, conservatively excluding Torch's cached blocks
because raw CUDA allocations cannot reuse them directly, and reserves 256 MiB
for workspace. Actual CUB workspace size is checked immediately before malloc.
Signed-int native corner/scan overflow is rejected. Admission failures include
counts/estimated bytes/free bytes and emit no partial mesh or downsampled output.
These estimates are safeguards, not measured peak-memory guarantees; runtime
OOM/quality/duration remain remotely unverified. Optional `diagnostics` records every refinement level and active voxel/vertex,
crossed-edge, and output counts; `bvh.query_sizes` records every distance query
size, including the corner query, so their sum gives cumulative query points.
Root records stage counts,
peak CUDA memory, synchronized elapsed time, input/output hashes and independent
artifact/topology controls, then releases T1 resources before running T2.
