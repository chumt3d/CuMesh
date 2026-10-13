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

REDUCTION_POLICY_VERSION = "tiny3d.wrap.sampled-cover-error-first.v4"


@dataclass(frozen=True)
class ReductionSettings:
    # Controller #365/#366 allocates a fraction of the product source tolerance
    # to sampled raw-wrap/candidate deviation. This evaluation profile uses 1/4.
    deviation_fraction: float = 0.25
    approximate_target_face_count: int | None = None
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
    """One accumulated, error-first pass; qualify actual output samples."""
    validate_mesh(raw_vertices, raw_faces)
    if not math.isfinite(source_tolerance_mm) or source_tolerance_mm <= 0:
        raise ValueError("source_tolerance_mm must be positive and finite")
    if (
        not math.isfinite(settings.deviation_fraction)
        or not 0 < settings.deviation_fraction < 1
    ):
        raise ValueError("deviation_fraction must be inside (0,1)")
    target = settings.approximate_target_face_count
    if target is not None and (not isinstance(target, int) or target <= 0):
        raise ValueError("approximate target must be a positive face count")
    import cumesh

    started = time.monotonic()
    budget = settings.deviation_fraction * source_tolerance_mm
    evidence = {
        "policy_version": REDUCTION_POLICY_VERSION,
        "source_tolerance_mm": source_tolerance_mm,
        "sample_deviation_budget_mm": budget,
        "deviation_fraction": settings.deviation_fraction,
        "native_error_metric": "outward_rounded_original_vertex_cluster_cover_mm",
        "native_metric_is_sample_certificate": False,
        "requested_approximate_target_faces": target,
        "raw_vertices": len(raw_vertices),
        "raw_faces": len(raw_faces),
        "position_frame": "unchanged_input_print_mm",
        "position_dtype": "float32",
        "selected": "raw_snapshot",
        "qualification": "candidate_only_final_product_gates_required",
    }
    selected_v, selected_f = raw_vertices, raw_faces

    def check(label):
        if time.monotonic() >= deadline_monotonic:
            raise WrapFailure("reduction_deadline", {"phase": label})
        free, total = torch.cuda.mem_get_info(raw_vertices.device)
        if total - free > settings.max_cuda_bytes:
            raise WrapFailure(
                "reduction_cuda_envelope", {"phase": label, "used_bytes": total - free}
            )

    check("QEM start")
    phase("qem_error_first")
    mesh = cumesh.CuMesh()
    try:
        mesh.init(raw_vertices, raw_faces)
        begin = time.monotonic()
        try:
            mesh.simplify(
                target,
                options={"max_error": budget, "deadline_monotonic": deadline_monotonic},
            )
        except cumesh.SimplificationFailure as error:
            evidence["native_failure"] = {
                "code": error.code,
                "target_faces": error.target_faces,
                "remaining_faces": error.remaining_faces,
            }
        evidence["qem_seconds"] = time.monotonic() - begin
        evidence["simplification"] = mesh.last_simplification
        if "native_failure" not in evidence:
            mesh.remove_unreferenced_vertices()
            cv, cf = mesh.read()
            cv, cf = cv.contiguous(), cf.int().contiguous()
            torch.cuda.synchronize(raw_vertices.device)
        else:
            cv = cf = None
    finally:
        del mesh
        gc.collect()
    if cv is not None:
        try:
            check("QEM postcheck")
            evidence["topology"], valid = inspect_surface(
                cv, cf, deadline_monotonic=deadline_monotonic, phase=phase
            )
            if valid:
                phase("sample_deviation")
                evidence["deviation"] = measure_bidirectional_deviation(
                    raw_vertices,
                    raw_faces,
                    cv,
                    cf,
                    max_deviation_mm=budget,
                    deadline_monotonic=deadline_monotonic,
                )
                valid = evidence["deviation"]["status"] == "PASSED"
            if valid:
                selected_v, selected_f = cv, cf
                evidence.update(
                    selected="error_bounded_candidate",
                    target_met=target is None or len(cf) <= target,
                )
            else:
                evidence["status"] = "postcheck_failed_raw_retained"
        except WrapFailure as error:
            evidence.update(
                status="postcheck_unresolved_raw_retained",
                failure={"code": error.code, "evidence": error.evidence},
            )
    evidence.update(
        final_vertices=len(selected_v),
        final_faces=len(selected_f),
        binary_stl_bytes=84 + 50 * len(selected_f),
        seconds=time.monotonic() - started,
    )
    return selected_v, selected_f, evidence
