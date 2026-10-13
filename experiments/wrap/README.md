# GPU exterior WRAP candidate

`cumesh_wrap.wrap_surface(V,F,settings=...,limits=...)` accepts contiguous CUDA
float32 positions **already in print millimetres**, and CUDA int32 triangles.
It returns those same canonical array types and a small evidence leaf. Scaling,
cleanup, fidelity, user disclosure, STL/3MF writers and final-byte qualification
belong to MAKE-SOLID. `WrapFailure` preserves typed cause/evidence and emits no
partial, pruned or reduced-resolution mesh. This is a candidate, not a released
Make-printable engine.

The FDM evaluation profile uses 0.5 mm one-sided offset, voxel <=0.25 mm,
and MAKE-SOLID's 0.4 mm minimum full wall. Padding is offset+two voxels per side.
Resolution is the smallest power of two >= padded extent/requested voxel, at
least upstream's 32. Actual voxels can be smaller. Projection ratio 0.1 is an
explicit **experimental setting**, not an adopted policy. Its movement is capped
by offset-minThickness/2-sqrt(3)*actualVoxel/2 and restricted to the original
lattice edge. That preserves the tetra cell containing each surface patch and
estimates distance from the source. The payload labels this ideal exact-UDF
estimate as uncertified for rounding and sliver omission; Warp's source
sliver omission means final product thickness/fidelity still must be measured.

## Borrowed source and deliberate changes

- CuMesh MIT upstream `12289e1062f0603f2f0d0771b02e1395d247f26f`:
  `cumesh/remeshing.py` octree refinement around UDF-offset, cube coordinates,
  power-of-two level expansion, optional barycentric closest-point projection.
  Preserve upstream rounded-up 0.87 refinement radius; Torch unique replaces the CUDA
  sparse hash/corner ordering. No output/order/numerical equivalence claim.
- Owned fork pilot `3d357b367d6b736baf7eb7cc702c0ac34029f345`:
  `experiments/narrow_band/.../warp_bvh.py` is copied unchanged. Warp1.13.0
  LBVH/no-sign distance remains the default. `cuBVH` can be injected using the
  same public unsigned_distance shape, device and barycentric contract.
  The pilot's CUDA hash/simple-dual-contour code was read but is not copied.
- Warp Apache-2.0 source `NVIDIA/warp` v1.13.0 supplies GPU BVH/traversal and
  atomics. No Warp implementation code is copied. Preserve its distribution
  notices with the dependency. Warp drops sufficiently sliver/degenerate faces;
  equal-distance ties/float arithmetic differ from cuBVH. No backend parity claim.
- Exterior reachability and six-tetra extraction are new Tiny3D code, using
  the Freudenthal subdivision (global 0->7 body diagonal). Sixteen tetra cases
  are constructed from sign partitions, not copied from another implementation.
  The flood graph includes the exact 14 tetra-lattice edge directions. Filled
  free regions become negative, eliminating internal UDF shells. A single output
  component is required by default; distant pieces fail instead of being erased.
- No CGAL/GPL source, library or executable is imported by this package. CGAL
  alpha-wrap may be measured separately as reference only.

The dense flood state+two queues use 12*(R+1)^3 bytes. Sparse refinement and
surface extraction add storage proportional to active band cells/corners and
output triangles. Tetra extraction is batched; global output edge dedup remains
proportional to the complete surface. Admission checks driver-free bytes plus a
256 MiB workspace reserve and a caller-specified CUDA envelope (default18 GiB on
L4). These safeguards are estimates, **not a measured peak bound**. No consumer
pricing is embedded. Timings/driver memory and final product acceptance are
separate evidence. The numerical sign epsilon is float32 epsilon*domain scale.

Every success checks edge incidence/direction, exact position coincidence,
connectivity and positive signed volume. Conforming tetra extraction with strictly
nonzero grid values supplies closed vertex links by construction; final exact
artifact validators must independently confirm manifoldness/self-intersection
and minimum wall. Projection never moves a crossing outside its shared grid edge.
Apertures below the wrapping scale and enclosed cavities may be filled and must
be disclosed. Open concavities reachable from the domain remain empty.

Install this isolated Python distribution through Socket Firewall with existing
Torch>=2.5 and `warp-lang==1.13.0`; the WRAP surface engine itself has no extension
build. The corrected native CuMesh reduction below requires a CUDA/NVCC build.
Fork source base is the exact parent gitlink31e62e1d1adcb4cb7132989137d88aa5042fae54;
production sees fork changes only after an explicit parent gitlink update.

## Bounded reduction pilot

`reduce_surface` proposes a QEM reduction using the CuMesh Python API. Its initial
pilot uses owned-fork base31e62e1's Python/defaults and records installed native
SHA, without claiming that binary's build provenance. Subsequent pilots use the
explicit source-built correction below. Candidate qualification does not infer
binary source correspondence from a package name.

The Standard pilot allocates one quarter of the existing product's source
tolerance (`8*0.001*source_bbox_diagonal_mm`) to sampled wrap/candidate deviation.
Standard uses one accumulated error-first QEM pass. Original face planes are
weighted by square root of double area and accumulated in float64; the metric
is weighted RMS distance to those planes in mm. An optimal LDL proposal (or
geometry-valid endpoint/segment fallback) rounds to actual stored float32 before
all native predicates; commit uses that identical proposal. The named error
cap never escalates to force a count. An optional advanced approximate count
stops further reduction, and the achieved count/error are disclosed.

Reduction v3 never normalizes/restores positions or recenters outputs. The caller
prepares its fixed source-to-print matrix before WRAP. The metric is an admission
proxy, not a sampled or Hausdorff certificate: qualification streams the actual
raw/candidate samples and keeps RAW if any final gate is failed or unresolved.
No density-derived target or corrective retry remains. Native state consists of
per-call CUDA float64 Tensor quadrics, merged/compacted each round; cache release
cannot discard the original planes. The native Python deadline acts between
steps and the integration caller retains its outer process deadline.

Each candidate checks exact edge incidence/direction, exact position coincidence,
duplicate/degenerate faces, components and positive volume per component, then
streams every referenced vertex and every face centroid through two-sided Warp
queries. Distances and centroids are float32. Receipts explicitly label sampled
maxima, exceedance fractions and completed counts. They establish no continuous
surface or Hausdorff bound. Final exact vertex links/self-intersection, wall,
source fidelity, cleanup, parts policy and actual STL/3MF bytes remain independent
product gates. Driver memory is sampled during raw+native+candidate/BVH overlap;
Torch allocation peaks alone omit native allocations.

## Root CuMesh collapse correction (controller371)

Policy v3 retains v2's a deterministic maximal batch over one immutable cost/CSR
snapshot. Each selection pass retains packed `(cost, edge ID)` priority, reserves
chosen complete stars, removes conflicting candidates, then repeats on the
remaining candidates. Geometry commits begin only after all selection passes.
Endpoint-minimum/gather propagation computes the same full-star minima with two
atomics per edge. One/two-face adjacency segments use an exact compare/swap;
nonmanifold segments retain the full deterministic CUB sort.

The initial three closed WRAP candidates became nonmanifold under both permitted
QEM attempts. `src/simplify.cu` now checks the full simplicial link: shared
neighbors must be the edge's one boundary or two interior opposites, and a common
opposite-pair link edge is forbidden (tetrahedron/duplicate-face case). An
interior edge joining two boundary vertices is rejected. A boundary contraction
cannot erase a last isolated triangle. Sorted unique native edge keys provide
binary-search membership, with O(degree*log E) work and no extra neighbor sets.
The conceptual source is [Dey et al., 1999](https://research-explorer.ista.ac.at/record/3582);
the implementation is independently written under the repository MIT license.

Cost-time validity rejects zero/nonfinite triangle area and nonpositive normal
dot products. Float32 QEM cancellation is clamped to positive zero; nonfinite
costs remain ineligible. Packed cost admission is finite and nonnegative. Propagation and
commit own every vertex in each complete endpoint one-ring using the existing
deterministic (cost,edge ID) ordering. This is stricter than legacy face-star
ownership; its throughput impact is measured separately. Eligibility runs in a
separate kernel over immutable faces before winners commit, preventing losing
eligibility readers from racing with a winner's face changes. Endpoint fans are
checked through existing face/edge adjacency: one oriented path or cycle, with
at most degree steps per vertex and O(F) aggregate walk work. Disconnected fans,
invalid incidence and inconsistent orientation cannot supply collapse endpoints.
The adjacency adds existing cache storage proportional to F/E; measured native
peak includes it. Its E+1 scan sentinel remains allocated for fully open inputs.
Face-to-edge lookup fills exactly three adjacency occurrences per triangle in
O(F log E) work. Existing opaque CUB workspace supplies E integer cursors after
the prefix scan is complete. This removes the high-valence star scan and remains
memory-safe on repeated-index triangles, which the fan guard refuses. Edge-face
pair ordering is unspecified; the oriented face-to-edge mapping is preserved.
Local link/normal
conditions do not certify global self-intersection or local wall thickness.

`simplify` preserves upstream threshold escalation but clamps it to float32's
largest finite value. A stalled step at that value emits `SimplificationFailure`
with code `simplify_no_legal_progress`, target and remaining counts; the current
mesh stays readable. Optional `deadline_monotonic` in the existing options dict
emits `simplify_deadline` between native steps. The root change applies to all
CuMesh callers without a native signature/ABI argument change. Consumers must
handle failure honestly; an unreachable target is never forced by infinity.
WRAP independently qualifies the readable result and removes only unreferenced
positions. Root source, Python, toolchain and native artifact hashes are recorded
by the dev CPU2 build; actual behavior is checked separately on L4.
Unchanged objects can be reused only with identical source/header hashes and
toolchain on the same immutable base image; build receipts bind reused object
hashes and their previous build manifest. Changed translation units are rebuilt.
