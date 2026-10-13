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

WRAP_POLICY_VERSION = "tiny3d.wrap.fdm-evaluation.v2"


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
def _hook_edges(edges: wp.array(dtype=wp.vec2i), parents: wp.array(dtype=wp.int32)):
    edge = edges[wp.tid()]
    a = parents[edge[0]]
    b = parents[edge[1]]
    if a != b:
        wp.atomic_min(parents, wp.max(a, b), wp.min(a, b))


def _component_labels(edges, vertex_count, stream, check):
    """Bounded hooks and pointer jumping, with deadlines between O(E+V) rounds.

    Parents decrease within the real component. No kernel walks a variable-length
    parent chain. Convergence requires stable roots and agreement along all edges.
    """
    parents = torch.arange(vertex_count, dtype=torch.int32, device=edges.device)
    rounds = 0
    edge_array = wp.from_torch(edges.int(), dtype=wp.vec2i, requires_grad=False)
    while True:
        check("component labeling")
        wp.launch(
            _hook_edges,
            dim=edges.shape[0],
            inputs=[edge_array, wp.from_torch(parents, requires_grad=False)],
            stream=stream,
        )
        jumped = parents[parents.long()]
        rounds += 1
        stable = torch.equal(jumped, parents)
        parents = jumped
        if stable and bool((parents[edges[:, 0]] == parents[edges[:, 1]]).all().item()):
            return parents, rounds


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


def _world_points(coords, resolution, scale, center):
    """One rounding to the positions actually queried/stored in this frame."""
    return ((coords.double() / resolution - 0.5) * scale + center.double()).float()


def _edge_world_points(endpoints, planes):
    return planes[endpoints.long(), torch.arange(3, device=planes.device)]


def _edge_interiors(world):
    """Parameter interval reaching representable, strictly interior f32 values.

    Each point's grid-plane/open-interval classification identifies its unique
    Freudenthal edge, provided the grid planes remain strictly increasing.
    This prevents coordinate coincidence without a second global sort.
    """
    start, finish = world[:, 0], world[:, 1]
    changing = start != finish
    delta = finish.double() - start.double()
    denominator = torch.where(changing, delta, torch.ones_like(delta))
    first = torch.nextafter(start, finish).double()
    last = torch.nextafter(finish, start).double()
    lower = torch.where(changing, (first - start.double()) / denominator, 0).amax(1)
    upper = torch.where(changing, (last - start.double()) / denominator, 1).amin(1)
    valid = changing.any(1) & (lower <= upper) & (lower > 0) & (upper < 1)
    return lower, upper, valid


def _edge_interpolate(world, parameter):
    return world[:, 0].double() + parameter[:, None] * (
        world[:, 1].double() - world[:, 0].double()
    )


def _edge_position_valid(world, positions):
    start, finish = world[:, 0], world[:, 1]
    changing = start != finish
    interior = (positions > torch.minimum(start, finish)) & (
        positions < torch.maximum(start, finish)
    )
    return torch.isfinite(positions).all(1) & torch.where(
        changing, interior, positions == start
    ).all(1)


def _orient_stored_faces(original, ideal, faces, away, batch_size, check):
    """Orient from the PL field; refuse rounded zero/inverted baseline faces."""
    for begin in range(0, len(faces), batch_size):
        check("stored extraction faces")
        batch = faces[begin : begin + batch_size]
        reference, stored = ideal[batch.long()], original[batch.long()].double()
        reference_normal = torch.cross(
            reference[:, 1] - reference[:, 0], reference[:, 2] - reference[:, 0], dim=1
        )
        normal = torch.cross(
            stored[:, 1] - stored[:, 0], stored[:, 2] - stored[:, 0], dim=1
        )
        valid = (
            torch.isfinite(normal).all(1)
            & (normal.square().sum(1) > 0)
            & ((normal * reference_normal).sum(1) > 0)
        )
        if not bool(valid.all().item()):
            raise WrapFailure(
                "wrap_extraction_precision_invalid",
                {"invalid_faces_in_batch": int((~valid).sum().item())},
            )
        flip = (reference_normal * away[begin : begin + len(batch)].double()).sum(1) < 0
        batch[flip] = batch[flip][:, (0, 2, 1)]


def _admit_projection(original, proposal, faces, rejected, batch_size, check):
    """All old/new face combinations protect any subset of refused proposals.

    Only one small face batch and combination are materialized at a time.
    This protects area/orientation, not global intersection or local wall checks.
    """
    for begin in range(0, len(faces), batch_size):
        check("stored projection faces")
        batch = faces[begin : begin + batch_size].long()
        old, new = original[batch].double(), proposal[batch].double()
        reference = torch.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0], dim=1)
        bad = torch.zeros(len(batch), dtype=torch.bool, device=faces.device)
        for combination in range(1, 8):
            triangle = torch.stack(
                [new[:, k] if combination & (1 << k) else old[:, k] for k in range(3)],
                dim=1,
            )
            normal = torch.cross(
                triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0], dim=1
            )
            bad |= (
                ~torch.isfinite(normal).all(1)
                | (normal.square().sum(1) <= 0)
                | ((normal * reference).sum(1) <= 0)
            )
        rejected[batch[bad].reshape(-1)] = True
    return torch.where(rejected[:, None], original, proposal)


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
        "policy_version": WRAP_POLICY_VERSION,
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
        center = (low.double() + high.double()) / 2
        extent = float((high.double() - low.double()).max().item())
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
        planes = _world_points(
            torch.arange(n, device=device)[:, None].expand(-1, 3),
            resolution,
            scale,
            center,
        )
        plane_steps = planes[1:] - planes[:-1]
        if not bool(torch.isfinite(planes).all().item()) or not bool(
            (plane_steps > 0).all().item()
        ):
            raise WrapFailure(
                "wrap_grid_precision_limit", {**evidence, "resolution": resolution}
            )
        infinity = torch.full_like(planes, math.inf)
        spacing = torch.maximum(
            torch.nextafter(planes, infinity).double() - planes.double(),
            planes.double() - torch.nextafter(planes, -infinity).double(),
        ).amax(0)
        # Half an ULP bounds each one-time coordinate rounding. Two such
        # displacements cover cell-center and corner positions in refinement.
        world_rounding = math.nextafter(float((spacing / 2).norm().item()), math.inf)
        # UDF is 1-Lipschitz. Interpolation over a cube has error <=sqrt(3)*h/2
        # (weighted variance bound). Limit displacement so every output triangle
        # remains >=minThickness/2 away from source, not only its vertices.
        interpolation_error = math.sqrt(3) * voxel / 2 + 2 * world_rounding
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
                "world_rounding_bound_mm": world_rounding,
                "position_policy": "shared_f32_grid_absolute_interpolation.v2",
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
                _world_points(coords.double() + 0.5, base, scale, center)
            )[0]
            # Preserve upstream's 0.87 rounded-up diagonal radius and margin.
            coords = coords[
                (values - settings.offset_mm).abs()
                <= 0.87 * scale / base + 2 * world_rounding
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
        points = _edge_world_points(_coords(keys, n), planes)
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
        lattice = _coords(keys, n).int()
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
            pos = _edge_world_points(lattice[ids], planes).double()
            away = (pos * (~negative)[:, :, None]).sum(dim=1) / (~negative).sum(dim=1)[
                :, None
            ] - (pos * negative[:, :, None]).sum(dim=1) / negative.sum(dim=1)[:, None]
            local = table[cases].reshape(-1, 2, 3)
            valid = local[:, :, 0] >= 0
            edge_ids = edges[local.clamp_min(0)]
            expanded = ids[:, None, None, :].expand(-1, 2, 3, -1)
            pairs = torch.gather(expanded, 3, edge_ids)[valid]
            triangle_edges.append(pairs.sort(dim=-1).values)
            directions.append(away[:, None, :].expand(-1, 2, -1)[valid].float())
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
        check("stored-coordinate interpolation", unique_edges.shape[0] * 256)
        world = _edge_world_points(endpoints, planes)
        lower, upper, interval_valid = _edge_interiors(world)
        if not bool(interval_valid.all().item()):
            raise WrapFailure(
                "wrap_grid_precision_limit",
                {
                    **evidence,
                    "invalid_edge_interiors": int((~interval_valid).sum().item()),
                },
            )
        t = (
            -edge_phi[:, 0].double()
            / (edge_phi[:, 1].double() - edge_phi[:, 0].double())
        ).clamp(0, 1)
        ideal = _edge_interpolate(world, t)
        t = torch.maximum(torch.minimum(t, upper), lower)
        original = _edge_interpolate(world, t).float()
        if not bool(_edge_position_valid(world, original).all().item()):
            raise WrapFailure("wrap_extraction_precision_invalid", evidence)
        extraction_adjust = float((original.double() - ideal).norm(dim=1).max().item())
        evidence["extraction_rounding_clamp_max_mm"] = extraction_adjust
        max_projection -= extraction_adjust
        evidence["max_projection_mm"] = max_projection
        if max_projection < 0:
            raise WrapFailure("wrap_thickness_resolution", evidence)
        del lattice, phi, edge_phi
        # Orient in the original PL field. After projection the patch need not
        # be planar; comparing each triangle to a centroid vector could flip only
        # one triangle of a tetra disk and break shared-edge winding.
        try:
            _orient_stored_faces(
                original, ideal, out_faces, away, limits.cell_batch_size, check
            )
        except WrapFailure as error:
            raise WrapFailure(error.code, {**evidence, **error.evidence}) from error
        del ideal, away
        out_vertices = original
        if settings.project_back > 0:
            check("stored-coordinate projection", original.shape[0] * 192)
            # Upstream closest-point/barycentric target; project only along the
            # shared lattice edge so every tetra disk stays inside its own cell.
            dist, face_id, uvw = bvh.unsigned_distance(original, return_uvw=True)
            closest = (
                vertices[faces[face_id]].double() * uvw[:, :, None].double()
            ).sum(dim=1)
            edge_vector = world[:, 1].double() - world[:, 0].double()
            dt = ((closest - original.double()) * edge_vector).sum(
                dim=1
            ) / edge_vector.square().sum(dim=1)
            dt *= settings.project_back
            bound = max_projection / edge_vector.norm(dim=1)
            dt = torch.maximum(torch.minimum(dt, bound), -bound)
            projected_t = torch.maximum(torch.minimum(t + dt, upper), lower)
            proposal = _edge_interpolate(world, projected_t).float()
            rejected = ~_edge_position_valid(world, proposal) | (
                (proposal.double() - original.double()).norm(dim=1) > max_projection
            )
            out_vertices = _admit_projection(
                original, proposal, out_faces, rejected, limits.cell_batch_size, check
            )
            evidence["projection_refused_vertices"] = int(rejected.sum().item())
            evidence["projection_max_mm"] = float(
                (out_vertices.double() - original.double()).norm(dim=1).max().item()
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
                proposal,
                rejected,
            )
        else:
            evidence["projection_max_mm"] = 0.0
        del unique_edges, endpoints, world, t, original, interval_valid, lower, upper
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
        roots, labeling_rounds = _component_labels(
            undirected, out_vertices.shape[0], stream, check
        )
        evidence["component_label_rounds"] = labeling_rounds
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
                "wall_lower_bound_estimate_mm": 2
                * (
                    settings.offset_mm
                    - interpolation_error
                    - extraction_adjust
                    - evidence["projection_max_mm"]
                ),
                "cuda_torch_peak_bytes": torch.cuda.max_memory_allocated(device),
                "wall_evidence_kind": "ideal_exact_udf_estimate_not_rounding_or_sliver_certified",
                "final_wall_measurement": "not_run",
                "final_product_validation": "not_run",
            }
        )
        return out_vertices.contiguous(), out_faces.contiguous(), evidence
