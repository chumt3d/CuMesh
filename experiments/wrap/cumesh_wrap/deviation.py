"""Bounded, two-sided SAMPLE deviation using the pilot's real Warp BVH.

Each direction samples every uniquely referenced source vertex once and every
source face centroid once. Centroids and Warp distances use float32 arithmetic.
This measures the declared samples only: it is not continuous surface coverage,
an exact nearest-distance proof, or a Hausdorff bound. Topology, intersections,
enclosure and wall thickness require their separate validation gates.
"""

import math
import time

import torch

from .warp_bvh import WarpUnsignedDistanceBVH


class _Unresolved(Exception):
    pass


def _identity(vertices, faces):
    return {
        "vertices": vertices.shape[0],
        "triangles": faces.shape[0],
        "vertex_data_ptr": vertices.data_ptr(),
        "face_data_ptr": faces.data_ptr(),
        "device": str(vertices.device),
        "coordinate_units": "mm",
    }


@torch.no_grad()
def measure_bidirectional_deviation(
    raw_vertices,
    raw_faces,
    candidate_vertices,
    candidate_faces,
    *,
    max_deviation_mm,
    deadline_monotonic,
    query_batch_size=262144,
    triangle_batch_size=65536,
    bvh_factory=None,
    raw_bvh=None,
):
    """Report PASSED/FAILED only for fully completed two-sided sample scope.

    FAILED means a declared sample's Warp float32 distance exceeds the budget;
    PASSED means every declared sample returned a distance <= the budget.
    Interrupted scope or backend failure is UNRESOLVED, preserving partial
    observations. Fractions weight each unique vertex and face centroid equally.
    Callers keep all arrays immutable and bind receipts to artifact hashes.

    ``raw_bvh`` reuses the immutable raw-wrap target across proposals. A factory
    receives (V,F,max_distance=...,query_batch_size=...) and its BVH must expose
    the original vertices/faces. Temporary referenced-vertex membership uses one
    CUDA boolean per source vertex; queries and triangle gathers are chunked.
    """
    started = time.monotonic()
    if not math.isfinite(max_deviation_mm) or max_deviation_mm <= 0:
        raise ValueError("max_deviation_mm must be positive and finite")
    if not math.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite")
    for name, count in (
        ("query_batch_size", query_batch_size),
        ("triangle_batch_size", triangle_batch_size),
    ):
        if type(count) is not int or not 0 < count < 2**31:
            raise ValueError(name + " must be a positive signed-int count")
    pairs = ((raw_vertices, raw_faces), (candidate_vertices, candidate_faces))
    for vertices, faces in pairs:
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.dtype != torch.float32
            or faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype != torch.int32
            or not vertices.is_cuda
            or faces.device != vertices.device
            or not vertices.is_contiguous()
            or not faces.is_contiguous()
            or not vertices.shape[0]
            or not faces.shape[0]
        ):
            raise ValueError("expected nonempty contiguous CUDA float32 V/int32 F")
    if candidate_vertices.device != raw_vertices.device:
        raise ValueError("both meshes must use the same CUDA device")
    evidence = {
        "status": "UNRESOLVED",
        "scope": "bidirectional_declared_samples_only",
        "raw": _identity(*pairs[0]),
        "candidate": _identity(*pairs[1]),
        "max_deviation_mm": float(max_deviation_mm),
        "sample_definition": {
            "vertices": "each uniquely referenced vertex once, ascending vertex index",
            "centroids": "one per full-list triangle, (v0+v1+v2)/3 in float32",
            "fraction_weighting": "one equal weight per vertex or centroid sample",
            "distance_backend": "Warp 1.13.0 float32 unsigned_distance",
            "continuous_or_hausdorff_bound": False,
        },
        "query_batch_size": query_batch_size,
        "triangle_batch_size": triangle_batch_size,
        "sample_count": 0,
        "samples_over_budget": 0,
        "fraction_over_budget": None,
        "max_sampled_deviation_mm": None,
        "directions": [],
    }

    def check():
        if time.monotonic() >= deadline_monotonic:
            raise _Unresolved("deadline")

    try:
        check()
        magnitude = 0.0
        for vertices, faces in pairs:
            for begin in range(0, vertices.shape[0], query_batch_size):
                check()
                block = vertices[begin : begin + query_batch_size]
                if not bool(torch.isfinite(block).all().item()):
                    raise ValueError("nonfinite input coordinates")
                magnitude = max(magnitude, float(block.abs().max().item()))
            for begin in range(0, faces.shape[0], triangle_batch_size):
                check()
                block = faces[begin : begin + triangle_batch_size]
                if bool(((block < 0) | (block >= vertices.shape[0])).any().item()):
                    raise ValueError("out-of-range triangle indices")
        # Strict slack beyond the largest possible vertex separation; honors
        # the actual Warp adapter's explicit max_distance capability.
        radius = max(4.0 * magnitude, 1.0)
        if radius >= 1e18:
            raise _Unresolved("backend_coordinate_range")
        factory = bvh_factory or WarpUnsignedDistanceBVH
        for direction, source, target in (
            ("raw_to_candidate", pairs[0], pairs[1]),
            ("candidate_to_raw", pairs[1], pairs[0]),
        ):
            check()
            source_vertices, source_faces = source
            target_vertices, target_faces = target
            row = {
                "direction": direction,
                "status": "UNRESOLVED",
                "required_referenced_vertices": None,
                "required_face_centroids": source_faces.shape[0],
                "sampled_referenced_vertices": 0,
                "sampled_face_centroids": 0,
                "sample_count": 0,
                "samples_over_budget": 0,
                "fraction_over_budget": None,
                "max_sampled_deviation_mm": None,
            }
            evidence["directions"].append(row)
            bvh = (
                raw_bvh
                if direction == "candidate_to_raw" and raw_bvh is not None
                else factory(
                    target_vertices,
                    target_faces,
                    max_distance=radius,
                    query_batch_size=query_batch_size,
                )
            )
            if bvh.vertices is not target_vertices or bvh.faces is not target_faces:
                raise _Unresolved("bvh_snapshot_mismatch")
            check()
            referenced = torch.zeros(
                source_vertices.shape[0],
                dtype=torch.bool,
                device=source_vertices.device,
            )
            for begin in range(0, source_faces.shape[0], triangle_batch_size):
                check()
                indices = source_faces[begin : begin + triangle_batch_size].reshape(-1)
                referenced[indices.long()] = True
            row["required_referenced_vertices"] = int(referenced.sum().item())

            def query(points, kind, *, witness_bvh=bvh):
                for begin in range(0, points.shape[0], query_batch_size):
                    check()
                    part = points[begin : begin + query_batch_size].contiguous()
                    distances, _, _ = witness_bvh.unsigned_distance(part)
                    if not bool(torch.isfinite(distances).all().item()):
                        raise _Unresolved("nonfinite_sample_distance")
                    count = part.shape[0]
                    maximum = float(distances.max().item())
                    # Promote only for comparison so a Python-float budget is not
                    # silently rounded to the distance tensor's float32 dtype.
                    over = int((distances.double() > max_deviation_mm).sum().item())
                    for item in (row, evidence):
                        item["sample_count"] += count
                        item["samples_over_budget"] += over
                        previous = item["max_sampled_deviation_mm"]
                        item["max_sampled_deviation_mm"] = (
                            maximum if previous is None else max(previous, maximum)
                        )
                        item["fraction_over_budget"] = (
                            item["samples_over_budget"] / item["sample_count"]
                        )
                    row[kind] += count
                    check()

            for begin in range(0, source_vertices.shape[0], query_batch_size):
                check()
                points = source_vertices[begin : begin + query_batch_size][
                    referenced[begin : begin + query_batch_size]
                ]
                query(points, "sampled_referenced_vertices")
            del referenced
            # Gather <=triangle_batch_size triangles; the inner query slices
            # before calling Warp, whose adapter allocates per supplied array.
            for begin in range(0, source_faces.shape[0], triangle_batch_size):
                check()
                triangles = source_vertices[
                    source_faces[begin : begin + triangle_batch_size].long()
                ]
                centroids = (
                    (triangles[:, 0] + triangles[:, 1]) + triangles[:, 2]
                ) / 3.0
                query(centroids, "sampled_face_centroids")
            row["status"] = "FAILED" if row["samples_over_budget"] else "PASSED"
            del query, bvh
        check()
        evidence["status"] = "FAILED" if evidence["samples_over_budget"] else "PASSED"
        evidence["reason"] = "both_declared_sample_sets_completed"
    except _Unresolved as exc:
        evidence["reason"] = str(exc)
    except torch.cuda.OutOfMemoryError:
        evidence["reason"] = "cuda_memory_exhaustion"
    except RuntimeError as exc:
        evidence["reason"] = "backend_runtime_error"
        evidence["error"] = str(exc)[:512]
    evidence["seconds"] = time.monotonic() - started
    return evidence
