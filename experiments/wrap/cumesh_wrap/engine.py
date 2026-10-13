# Copyright (c) 2025 Jianfeng XIANG. MIT; see ../LICENSE.
# Sparse refinement adapted from CuMesh 12289e1062f0603f2f0d0771b02e1395d247f26f.
# Exterior traversal and conforming tetra extraction are Tiny3D additions.
"""GPU WRAP candidate. All geometry runs on the caller's CUDA stream.

Input/output positions have identical print-millimetre coordinates. The caller
owns scaling, cleanup, fidelity, disclosure, and writers. No production adoption
or equivalence to CGAL/cuBVH is claimed. See README for source mapping and limits.
"""

import math
import time
from dataclasses import asdict, dataclass

import torch
import warp as wp

from .warp_bvh import WarpUnsignedDistanceBVH, validate_mesh


class WrapFailure(RuntimeError):
    """No publishable mesh; preserve caller's input and diagnostics."""

    def __init__(self, code: str, evidence: dict):
        self.code = code
        self.evidence = evidence
        super().__init__(f"{code}: {evidence}")


@dataclass(frozen=True)
class WrapSettings:
    # Assignment's FDM wrap profile: 0.5 mm offset, approximately 1 mm wall.
    offset_mm: float = 0.5
    # Assignment permits <=offset/2; actual voxel is rounded DOWN by resolution.
    voxel_mm: float = 0.25
    # MAKE-SOLID minimum wall policy (full wall, not one-sided offset).
    min_thickness_mm: float = 0.4
    # Explicit bounded projection towards the original, constrained to grid edges.
    # 0.1 is an evaluation setting, not an adopted quality policy.
    project_back: float = 0.1
    require_one_part: bool = True


@dataclass(frozen=True)
class WrapLimits:
    # L4 operational admission: reserve 6 GiB of its 24 GiB for driver/runtime.
    max_cuda_bytes: int = 18 * 1024**3
    # Workspace allowance above explicit tensors; safeguard, not measured peak.
    workspace_bytes: int = 256 * 1024**2
    # Batch bounds transient closest-point and extraction allocations.
    query_batch_size: int = 262144
    cell_batch_size: int = 32768
    # Existing Tiny3D 900 s route budget, supplied/shortened by the executor.
    max_seconds: float = 900.0


# Binary cube corners; all cells share the same Freudenthal body diagonal 0->7.
CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
TETS = (
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
)
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
# Positive edges of that exact tetra lattice, plus their inverses. A six-neighbor
# flood would incorrectly fill passages crossing face/body diagonals.
LATTICE_STEPS = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
    (-1, -1, 0),
    (-1, 0, -1),
    (0, -1, -1),
    (-1, -1, -1),
)


@wp.kernel
def _seed_boundary(
    n: int,
    states: wp.array(dtype=wp.int32),
    queue: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    face = i // (n * n)
    a = (i // n) % n
    b = i % n
    x = a
    y = b
    z = int(0)
    if face == 1:
        z = n - 1
    elif face == 2:
        x = int(0)
        y = a
        z = b
    elif face == 3:
        x = n - 1
        y = a
        z = b
    elif face == 4:
        x = a
        y = int(0)
        z = b
    elif face == 5:
        x = a
        y = n - 1
        z = b
    key = (x * n + y) * n + z
    if wp.atomic_cas(states, key, 0, 1) == 0:
        slot = wp.atomic_add(count, 0, 1)
        queue[slot] = key


@wp.kernel
def _expand_frontier(
    n: int,
    states: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.vec3i),
):
    key = source[wp.tid()]
    x = key // (n * n)
    y = (key // n) % n
    z = key % n
    for j in range(14):
        p = wp.vec3i(x, y, z) + steps[j]
        if p[0] >= 0 and p[0] < n and p[1] >= 0 and p[1] < n and p[2] >= 0 and p[2] < n:
            other = (p[0] * n + p[1]) * n + p[2]
            if wp.atomic_cas(states, other, 0, 1) == 0:
                slot = wp.atomic_add(count, 0, 1)
                target[slot] = other


@wp.kernel
def _union_edges(edges: wp.array(dtype=wp.vec2i), parents: wp.array(dtype=wp.int32)):
    edge = edges[wp.tid()]
    a = edge[0]
    b = edge[1]
    while a != b:
        while parents[a] != a:
            a = parents[a]
        while parents[b] != b:
            b = parents[b]
        if a != b:
            hi = wp.max(a, b)
            lo = wp.min(a, b)
            if wp.atomic_cas(parents, hi, hi, lo) == hi:
                break


@wp.kernel
def _roots(parents: wp.array(dtype=wp.int32), roots: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = i
    while parents[p] != p:
        p = parents[p]
    roots[i] = p


def _triangle_table():
    """Construct the sixteen tetra sign cases; no borrowed lookup table."""
    rows = []
    for mask in range(16):
        inside = [i for i in range(4) if mask & (1 << i)]
        outside = [i for i in range(4) if not mask & (1 << i)]

        def edge(a, b):
            return TET_EDGES.index(tuple(sorted((a, b))))

        if len(inside) in (1, 3):
            one, others = (
                (inside[0], outside) if len(inside) == 1 else (outside[0], inside)
            )
            row = [edge(one, q) for q in others]
        elif len(inside) == 2:
            a, b = inside
            c, d = outside
            quad = [edge(a, c), edge(a, d), edge(b, d), edge(b, c)]
            row = [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]
        else:
            row = []
        rows.append(row + [-1] * (6 - len(row)))
    return rows


def _coords(keys, n):
    return torch.stack((keys // (n * n), (keys // n) % n, keys % n), dim=-1)


@torch.no_grad()
def wrap_surface(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    settings: WrapSettings = WrapSettings(),
    limits: WrapLimits = WrapLimits(),
    bvh=None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return canonical CUDA arrays and small evidence leaf, or WrapFailure.

    Publication requires the executor's final-byte validation and fidelity gates.
    A failure emits no reduced-resolution, cropped, or component-pruned output.
    """
    validate_mesh(vertices, faces)
    for value in (
        settings.offset_mm,
        settings.voxel_mm,
        settings.min_thickness_mm,
        limits.max_seconds,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("lengths and max_seconds must be positive and finite")
    if settings.voxel_mm > settings.offset_mm / 2:
        raise ValueError("voxel_mm must be <= offset_mm/2")
    if not math.isfinite(settings.project_back) or not 0 <= settings.project_back <= 1:
        raise ValueError("project_back must be in [0,1]")
    if min(limits.query_batch_size, limits.cell_batch_size, limits.max_cuda_bytes) <= 0:
        raise ValueError("memory/count limits must be positive")
    device = vertices.device
    evidence = {
        "schema": "tiny3d.gpu-wrap.v1",
        "settings": asdict(settings),
        "limits": asdict(limits),
        "status": "running",
        "backend": "warp_lbvh_1.13.0",
        "extraction": "freudenthal_marching_tetrahedra",
        "refinement": [],
    }
    started = time.perf_counter()
    wp.init()
    with torch.cuda.device(device):
        stream = wp.stream_from_torch(torch.cuda.current_stream(device))
        baseline = torch.cuda.memory_allocated(device)

        def check(stage, extra=0):
            free, total = torch.cuda.mem_get_info(device)
            live = torch.cuda.memory_allocated(device) - baseline
            needed = extra + limits.workspace_bytes
            if needed > free or live + needed > limits.max_cuda_bytes:
                raise WrapFailure(
                    "wrap_memory_limit",
                    {
                        **evidence,
                        "stage": stage,
                        "live_torch_bytes": live,
                        "extra_bytes": extra,
                        "driver_free_bytes": free,
                    },
                )
            if time.perf_counter() - started > limits.max_seconds:
                raise WrapFailure("wrap_deadline", {**evidence, "stage": stage})

        low, high = vertices.amin(dim=0), vertices.amax(dim=0)
        center = (low + high) / 2
        extent = float((high - low).max().item())
        # Padding: offset + two requested voxels on each side keeps all source
        # and offset material strictly away from the seeded domain boundary.
        scale = extent + 2 * (settings.offset_mm + 2 * settings.voxel_mm)
        resolution = max(32, 1 << math.ceil(math.log2(scale / settings.voxel_mm)))
        n = resolution + 1
        if n**3 >= 2**31:
            raise WrapFailure(
                "wrap_grid_index_limit", {**evidence, "resolution": resolution}
            )
        voxel = scale / resolution
        # UDF is 1-Lipschitz. Interpolation over a cube has error <=sqrt(3)*h/2
        # (weighted variance bound). Limit displacement so every output triangle
        # remains >=minThickness/2 away from source, not only its vertices.
        interpolation_error = math.sqrt(3) * voxel / 2
        max_projection = (
            settings.offset_mm - settings.min_thickness_mm / 2 - interpolation_error
        )
        if max_projection < 0:
            raise WrapFailure(
                "wrap_thickness_resolution",
                {
                    **evidence,
                    "voxel_mm": voxel,
                    "interpolation_error_mm": interpolation_error,
                },
            )
        evidence.update(
            {
                "resolution": resolution,
                "voxel_mm": voxel,
                "domain_scale_mm": scale,
                "domain_center_mm": center.tolist(),
                "interpolation_error_bound_mm": interpolation_error,
                "max_projection_mm": max_projection,
            }
        )
        check("grid admission", 12 * n**3)
        if bvh is None:
            bvh = WarpUnsignedDistanceBVH(
                vertices,
                faces,
                max_distance=4 * scale,
                query_batch_size=limits.query_batch_size,
            )
        if bvh.device != device:
            raise ValueError("BVH device mismatch")
        offsets = torch.tensor(CORNERS, dtype=torch.int64, device=device)
        base = 32
        coords = torch.stack(
            torch.meshgrid(*(torch.arange(base, device=device),) * 3, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        while True:
            check("refinement", coords.shape[0] * 96)
            values = bvh.unsigned_distance(
                ((coords.float() + 0.5) / base - 0.5) * scale + center
            )[0]
            # Upstream uses 0.87 (rounded sqrt(3)/2); use exact conservative radius.
            coords = coords[
                (values - settings.offset_mm).abs() <= math.sqrt(3) / 2 * scale / base
            ]
            evidence["refinement"].append(
                {"resolution": base, "active_cells": coords.shape[0]}
            )
            if base == resolution:
                break
            check("eight child expansion", coords.shape[0] * 8 * (24 + 96))
            coords = (coords[:, None, :] * 2 + offsets[None, :, :]).reshape(-1, 3)
            base *= 2
        if not coords.shape[0]:
            raise WrapFailure("wrap_empty_band", evidence)
        check("sparse corners", coords.shape[0] * 8 * 64 + 12 * n**3)
        corners = coords[:, None, :] + offsets[None, :, :]
        corner_keys = (corners[:, :, 0] * n + corners[:, :, 1]) * n + corners[:, :, 2]
        keys, inverse = torch.unique(
            corner_keys.reshape(-1), sorted=True, return_inverse=True
        )
        cell_corners = inverse.reshape(-1, 8)
        del inverse, corners, corner_keys
        points = (_coords(keys, n).float() / resolution - 0.5) * scale + center
        values = bvh.unsigned_distance(points)[0] - settings.offset_mm
        evidence.update(
            {"active_cells": coords.shape[0], "active_corners": keys.numel()}
        )
        del coords, points
        check("exterior flood", 12 * n**3)
        states = torch.zeros(n**3, dtype=torch.int32, device=device)
        states[keys[values <= 0]] = -1
        q0 = torch.empty_like(states)
        q1 = torch.empty_like(states)
        count = torch.zeros(1, dtype=torch.int32, device=device)
        steps = torch.tensor(LATTICE_STEPS, dtype=torch.int32, device=device)

        def array(t, dtype=None):
            return wp.from_torch(t, dtype=dtype, requires_grad=False)

        wp.launch(
            _seed_boundary,
            dim=6 * n * n,
            inputs=[n, array(states), array(q0), array(count)],
            stream=stream,
        )
        size = int(count.item())
        visited = size
        iterations = 0
        while size:
            check("flood frontier")
            count.zero_()
            wp.launch(
                _expand_frontier,
                dim=size,
                inputs=[
                    n,
                    array(states),
                    array(q0),
                    array(q1),
                    array(count),
                    array(steps, wp.vec3i),
                ],
                stream=stream,
            )
            size = int(count.item())
            visited += size
            iterations += 1
            q0, q1 = q1, q0
        evidence.update(
            {
                "exterior_vertices": visited,
                "flood_iterations": iterations,
                "filled_sparse_free_vertices": int(
                    ((states[keys] != 1) & (values > 0)).sum().item()
                ),
            }
        )
        exterior = states[keys] == 1
        # Strict sign separation avoids a zero set passing exactly through grid
        # vertices. Perturbation is one float32 epsilon in domain units.
        sign_epsilon = torch.finfo(torch.float32).eps * scale
        magnitudes = values.abs().clamp_min(sign_epsilon)
        phi = torch.where(exterior, magnitudes, -magnitudes)
        del states, q0, q1, count, exterior, magnitudes, steps, values
        lattice = _coords(keys, n).float()
        del keys
        tetra = torch.tensor(TETS, dtype=torch.int64, device=device)
        edges = torch.tensor(TET_EDGES, dtype=torch.int64, device=device)
        table = torch.tensor(_triangle_table(), dtype=torch.int64, device=device)
        masks = torch.tensor((1, 2, 4, 8), dtype=torch.int64, device=device)
        triangle_edges = []
        directions = []
        for begin in range(0, cell_corners.shape[0], limits.cell_batch_size):
            check("tetra extraction", limits.cell_batch_size * 6 * 512)
            ids = cell_corners[begin : begin + limits.cell_batch_size, tetra].reshape(
                -1, 4
            )
            field = phi[ids]
            negative = field < 0
            cases = (negative.long() * masks).sum(dim=1)
            active = (cases != 0) & (cases != 15)
            ids, negative, cases = ids[active], negative[active], cases[active]
            if not ids.shape[0]:
                continue
            pos = lattice[ids]
            away = (pos * (~negative)[:, :, None]).sum(dim=1) / (~negative).sum(dim=1)[
                :, None
            ] - (pos * negative[:, :, None]).sum(dim=1) / negative.sum(dim=1)[:, None]
            local = table[cases].reshape(-1, 2, 3)
            valid = local[:, :, 0] >= 0
            edge_ids = edges[local.clamp_min(0)]
            expanded = ids[:, None, None, :].expand(-1, 2, 3, -1)
            pairs = torch.gather(expanded, 3, edge_ids)[valid]
            triangle_edges.append(pairs.sort(dim=-1).values)
            directions.append(away[:, None, :].expand(-1, 2, -1)[valid])
        if not triangle_edges:
            raise WrapFailure("wrap_empty_surface", evidence)
        check("global edge dedup", sum(t.numel() * 8 for t in triangle_edges) * 4)
        pairs = torch.cat(triangle_edges)
        away = torch.cat(directions)
        del triangle_edges, directions, cell_corners
        unique_edges, inverse = torch.unique(
            pairs.reshape(-1, 2), sorted=True, dim=0, return_inverse=True
        )
        out_faces = inverse.reshape(-1, 3).int()
        del pairs, inverse
        endpoints = lattice[unique_edges]
        edge_phi = phi[unique_edges]
        t = (-edge_phi[:, 0] / (edge_phi[:, 1] - edge_phi[:, 0])).clamp(0, 1)
        original = (
            (endpoints[:, 0] + t[:, None] * (endpoints[:, 1] - endpoints[:, 0]))
            / resolution
            - 0.5
        ) * scale + center
        del lattice, phi, edge_phi
        # Orient in the original PL field. After projection the patch need not
        # be planar; comparing each triangle to a centroid vector could flip only
        # one triangle of a tetra disk and break shared-edge winding.
        triangles = original[out_faces]
        normals = torch.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1
        )
        flip = (normals * away).sum(dim=1) < 0
        out_faces[flip] = out_faces[flip][:, (0, 2, 1)]
        del triangles, normals, away, flip
        out_vertices = original
        if settings.project_back > 0:
            # Upstream closest-point/barycentric target; project only along the
            # shared lattice edge so every tetra disk stays inside its own cell.
            dist, face_id, uvw = bvh.unsigned_distance(original, return_uvw=True)
            closest = (vertices[faces[face_id]] * uvw[:, :, None]).sum(dim=1)
            edge_vector = (endpoints[:, 1] - endpoints[:, 0]) * voxel
            dt = ((closest - original) * edge_vector).sum(
                dim=1
            ) / edge_vector.square().sum(dim=1)
            dt *= settings.project_back
            bound = max_projection / edge_vector.norm(dim=1)
            dt = torch.maximum(torch.minimum(dt, bound), -bound)
            # Interior endpoints maintain the tetra sign topology; float32
            # epsilon is numerical separation, not a geometry quality threshold.
            # One normalized float32 epsilon is smaller than a world-coordinate
            # ULP on large grids. Account for resolution before interpolation.
            parameter_guard = torch.finfo(torch.float32).eps * n
            lower = torch.minimum(t, torch.full_like(t, parameter_guard))
            upper = torch.maximum(t, torch.full_like(t, 1 - parameter_guard))
            projected_t = torch.maximum(torch.minimum(t + dt, upper), lower)
            out_vertices = original + (projected_t - t)[:, None] * edge_vector
            evidence["projection_max_mm"] = float(
                (out_vertices - original).norm(dim=1).max().item()
            )
            del (
                dist,
                face_id,
                uvw,
                closest,
                edge_vector,
                dt,
                bound,
                projected_t,
                lower,
                upper,
            )
        else:
            evidence["projection_max_mm"] = 0.0
        del unique_edges, endpoints, t, original
        check("topology validation", out_faces.shape[0] * 256)
        directed = out_faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
        sorted_edges = directed.sort(dim=1).values
        undirected, mapping, counts = torch.unique(
            sorted_edges, sorted=True, dim=0, return_inverse=True, return_counts=True
        )
        winding = torch.zeros(counts.shape[0], dtype=torch.int32, device=device)
        winding.scatter_add_(
            0, mapping, torch.where(directed[:, 0] < directed[:, 1], 1, -1).int()
        )
        topology = {
            "boundary_edges": int((counts == 1).sum().item()),
            "nonmanifold_edges": int((counts > 2).sum().item()),
            "orientation_conflicts": int((winding != 0).sum().item()),
        }
        exact_positions = torch.unique(out_vertices, dim=0).shape[0]
        topology["coincident_vertices"] = out_vertices.shape[0] - exact_positions
        if any(topology.values()):
            raise WrapFailure(
                "wrap_topology_invalid", {**evidence, "topology": topology}
            )
        parents = torch.arange(out_vertices.shape[0], dtype=torch.int32, device=device)
        roots = torch.empty_like(parents)
        wp.launch(
            _union_edges,
            dim=undirected.shape[0],
            inputs=[array(undirected.int(), wp.vec2i), array(parents)],
            stream=stream,
        )
        wp.launch(
            _roots,
            dim=parents.numel(),
            inputs=[array(parents), array(roots)],
            stream=stream,
        )
        topology["parts"] = torch.unique(roots).numel()
        if settings.require_one_part and topology["parts"] != 1:
            raise WrapFailure("wrap_multiple_parts", {**evidence, "topology": topology})
        triangles = out_vertices[out_faces].double()
        volume = float(
            (triangles[:, 0] * torch.cross(triangles[:, 1], triangles[:, 2], dim=1))
            .sum()
            .item()
            / 6
        )
        if volume <= 0 or not math.isfinite(volume):
            raise WrapFailure(
                "wrap_orientation_invalid", {**evidence, "signed_volume_mm3": volume}
            )
        torch.cuda.synchronize(device)
        evidence.update(
            {
                "status": "candidate",
                "topology": topology,
                "signed_volume_mm3": volume,
                "output_vertices": out_vertices.shape[0],
                "output_faces": out_faces.shape[0],
                "seconds": time.perf_counter() - started,
                "wall_lower_bound_mm": 2
                * (
                    settings.offset_mm
                    - interpolation_error
                    - evidence["projection_max_mm"]
                ),
                "cuda_torch_peak_bytes": torch.cuda.max_memory_allocated(device),
                "final_product_validation": "not_run",
            }
        )
        return out_vertices.contiguous(), out_faces.contiguous(), evidence
