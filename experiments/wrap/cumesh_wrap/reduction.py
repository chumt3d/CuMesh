"""Bounded QEM proposal and sampled qualification in print millimetres.

The density model is a proposal, never an error certificate. The existing
product's source fidelity, exact geometry audit and wall gates remain required.
"""

import gc
import math
import time
from dataclasses import dataclass

import torch
import warp as wp

from .deviation import measure_bidirectional_deviation
from .engine import WrapFailure, _component_labels
from .warp_bvh import validate_mesh


@dataclass(frozen=True)
class ReductionSettings:
    # Controller #365/#366 allocates a fraction of the product source tolerance
    # to sampled raw-wrap/candidate deviation. This evaluation profile uses 1/4.
    deviation_fraction: float = 0.25
    approximate_target_face_count: int | None = None
    # CuMesh's checked Python API is normalized-coordinate QEM, not millimetres.
    normalization_mm: float = 100.0
    max_cuda_bytes: int = 18 * 1024**3


def inspect_surface(vertices, faces, *, deadline_monotonic, phase=lambda _: None):
    """Cheap exact-index/position gates; vertex links/intersections are external."""
    validate_mesh(vertices, faces)

    def check(label, _estimate=0):
        if time.monotonic() >= deadline_monotonic:
            raise WrapFailure("reduction_deadline", {"phase": label})

    phase("edge_topology")
    check("edge topology")
    directed = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges, inverse, counts = torch.unique(
        directed.sort(dim=1).values,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    winding = torch.zeros(len(counts), dtype=torch.int32, device=vertices.device)
    winding.scatter_add_(
        0, inverse, torch.where(directed[:, 0] < directed[:, 1], 1, -1).int()
    )
    result = {
        "boundary_edges": int((counts == 1).sum().item()),
        "nonmanifold_edges": int((counts > 2).sum().item()),
        "orientation_conflicts": int((winding != 0).sum().item()),
        "coincident_vertices": len(vertices) - len(torch.unique(vertices, dim=0)),
        "duplicate_faces": len(faces)
        - len(torch.unique(faces.sort(dim=1).values, dim=0)),
    }
    del directed, inverse, counts, winding
    wp.init()
    stream = wp.stream_from_torch(torch.cuda.current_stream(vertices.device))
    roots, rounds = _component_labels(edges, len(vertices), stream, check)
    del edges
    _, component = torch.unique(roots, sorted=True, return_inverse=True)
    volume = torch.zeros(
        int(component.max().item()) + 1, dtype=torch.float64, device=vertices.device
    )
    degenerate = 0
    for start in range(0, len(faces), 262144):
        check("face geometry")
        chunk = faces[start : start + 262144]
        triangles = vertices[chunk].double()
        normal = torch.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1
        )
        degenerate += int((normal.square().sum(dim=1) == 0).sum().item())
        signed = (
            triangles[:, 0] * torch.cross(triangles[:, 1], triangles[:, 2], dim=1)
        ).sum(dim=1) / 6
        volume.scatter_add_(0, component[chunk[:, 0].long()], signed)
    result.update(
        degenerate_faces=degenerate,
        parts=len(volume),
        component_label_rounds=rounds,
        nonpositive_volume_parts=int((volume <= 0).sum().item()),
        signed_volume_mm3=float(volume.sum().item()),
        vertex_links="not_checked_here",
        self_intersections="not_checked_here",
        min_wall="not_checked_here",
    )
    rejected = any(
        result[k]
        for k in (
            "boundary_edges",
            "nonmanifold_edges",
            "orientation_conflicts",
            "coincident_vertices",
            "duplicate_faces",
            "degenerate_faces",
            "nonpositive_volume_parts",
        )
    )
    return result, not rejected


def reduce_surface(
    raw_vertices,
    raw_faces,
    *,
    source_tolerance_mm,
    deadline_monotonic,
    settings=ReductionSettings(),
    phase=lambda _: None,
):
    """One QEM call, at most one fresh-raw corrective call; preserve last pass."""
    validate_mesh(raw_vertices, raw_faces)
    if not math.isfinite(source_tolerance_mm) or source_tolerance_mm <= 0:
        raise ValueError("source_tolerance_mm must be positive and finite")
    if not 0 < settings.deviation_fraction < 1:
        raise ValueError("deviation_fraction must be inside (0,1)")
    if settings.approximate_target_face_count is not None and (
        not isinstance(settings.approximate_target_face_count, int)
        or settings.approximate_target_face_count <= 0
    ):
        raise ValueError("approximate target must be a positive face count")
    import cumesh

    device = raw_vertices.device
    started = time.monotonic()

    def check(label):
        if time.monotonic() >= deadline_monotonic:
            raise WrapFailure("reduction_deadline", {"phase": label})
        free, total = torch.cuda.mem_get_info(device)
        if total - free > settings.max_cuda_bytes:
            raise WrapFailure(
                "reduction_cuda_envelope", {"phase": label, "used_bytes": total - free}
            )

    phase("density_model")
    area = 0.0
    for start in range(0, len(raw_faces), 262144):
        check("density model")
        triangles = raw_vertices[raw_faces[start : start + 262144]].double()
        area += (
            float(
                torch.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                    dim=1,
                )
                .norm(dim=1)
                .sum()
                .item()
            )
            / 2
        )
    del triangles
    budget = settings.deviation_fraction * source_tolerance_mm
    # Equilateral-triangle circumradius = budget, area=3*sqrt(3)/4*budget^2.
    # QEM output is not guaranteed equilateral; samples judge the actual proposal.
    triangle_area = 3 * math.sqrt(3) / 4 * budget**2
    density_target = max(4, math.ceil(area / triangle_area))
    first_target = min(
        len(raw_faces), settings.approximate_target_face_count or density_target
    )
    evidence = {
        "source_tolerance_mm": source_tolerance_mm,
        "sample_deviation_budget_mm": budget,
        "deviation_fraction": settings.deviation_fraction,
        "raw_area_mm2": area,
        "density_model": "equilateral_circumradius_equals_sample_budget",
        "model_triangle_area_mm2": triangle_area,
        "density_target_faces": density_target,
        "requested_approximate_target_faces": settings.approximate_target_face_count,
        "raw_vertices": len(raw_vertices),
        "raw_faces": len(raw_faces),
        "attempts": [],
        "selected": "raw_snapshot",
        "qualification": "candidate_only_final_product_gates_required",
    }
    selected_v, selected_f = raw_vertices, raw_faces
    center = (raw_vertices.amin(dim=0) + raw_vertices.amax(dim=0)) / 2
    for attempt in range(2):
        # Halving proposal spacing quadruples area-derived density. It is the
        # single permitted correction, never an iterative target-count search.
        target = min(len(raw_faces), first_target * (4 if attempt else 1))
        if target == len(raw_faces):
            evidence["attempts"].append(
                {"target_faces": target, "status": "raw_retained_no_reduction"}
            )
            break
        check("QEM start")
        phase(f"qem_{attempt + 1}")
        qem_started = time.monotonic()
        mesh = cumesh.CuMesh()
        mesh.init(
            ((raw_vertices - center) / settings.normalization_mm).contiguous(),
            raw_faces,
        )
        # Upstream Python/native defaults retained. The subprocess caller supplies
        # a hard deadline around this native API, which has no cancellation hook.
        qem_stop = None
        try:
            mesh.simplify(
                target,
                verbose=False,
                options={"deadline_monotonic": deadline_monotonic},
            )
        except cumesh.SimplificationFailure as error:
            qem_stop = {
                "code": error.code,
                "target_faces": error.target_faces,
                "remaining_faces": error.remaining_faces,
            }
        # Unreferenced positions are not mesh components and carry no geometry.
        mesh.remove_unreferenced_vertices()
        cv, cf = mesh.read()
        cv = (cv * settings.normalization_mm + center).contiguous()
        cf = cf.int().contiguous()
        torch.cuda.synchronize(device)
        del mesh
        gc.collect()
        row = {
            "attempt": attempt + 1,
            "target_faces": target,
            "vertices": len(cv),
            "faces": len(cf),
            "qem_seconds": time.monotonic() - qem_started,
            "qem_stop": qem_stop,
        }
        evidence["attempts"].append(row)
        try:
            check("QEM postcheck")
            topology, valid = inspect_surface(
                cv, cf, deadline_monotonic=deadline_monotonic, phase=phase
            )
            row["topology"] = topology
            if valid:
                phase(f"sample_deviation_{attempt + 1}")
                row["deviation"] = measure_bidirectional_deviation(
                    raw_vertices,
                    raw_faces,
                    cv,
                    cf,
                    max_deviation_mm=budget,
                    deadline_monotonic=deadline_monotonic,
                )
                valid = row["deviation"]["status"] == "PASSED"
            row["status"] = (
                "sample_and_edge_gates_passed" if valid else "postcheck_failed"
            )
            if valid:
                selected_v, selected_f = cv, cf
                evidence["selected"] = f"qem_{attempt + 1}"
                break
        except WrapFailure as error:
            row.update(
                status="postcheck_refused", code=error.code, detail=error.evidence
            )
            break
        del cv, cf
        gc.collect()
    torch.cuda.synchronize(device)
    evidence.update(
        final_vertices=len(selected_v),
        final_faces=len(selected_f),
        binary_stl_bytes=84 + 50 * len(selected_f),
        seconds=time.monotonic() - started,
    )
    phase("reduction_done")
    return selected_v, selected_f, evidence
