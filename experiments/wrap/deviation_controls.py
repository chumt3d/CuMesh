"""Analytic controls of the real CUDA/Warp two-sided SAMPLE deviation path.

Expected values come from identity and independent parallel-plane geometry.
Run on the parent's approved GPU substrate; no mock or replacement BVH:
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python deviation_controls.py --out DIR

These checks establish sample behavior only. They do not validate topology,
intersections, wall thickness, continuous deviation or product acceptance.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import warp as wp

from cumesh_wrap.deviation import measure_bidirectional_deviation
from cumesh_wrap.warp_bvh import WarpUnsignedDistanceBVH


def _mesh(vertices, faces):
    return (
        torch.tensor(vertices, device="cuda", dtype=torch.float32),
        torch.tensor(faces, device="cuda", dtype=torch.int32),
    )


def _hash(mesh):
    return {
        "vertices_sha256": hashlib.sha256(mesh[0].cpu().numpy().tobytes()).hexdigest(),
        "faces_sha256": hashlib.sha256(mesh[1].cpu().numpy().tobytes()).hexdigest(),
    }


def execute(out):
    out.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(out / "warp-cache")
    torch.set_num_threads(2)
    triangle = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    faces = ((0, 1, 2),)
    raw = _mesh(triangle, faces)
    # The unused vertex must not become a sample. This guards the declared
    # referenced-vertex scope at the actual API boundary.
    same = _mesh(triangle + ((0, 0, 100),), faces)
    shifted = _mesh(tuple((x, y, 0.0625) for x, y, _ in triangle), faces)
    with_extra = _mesh(
        triangle + ((0, 0, 4), (0.25, 0, 4), (0, 0.25, 4)),
        ((0, 1, 2), (3, 4, 5)),
    )
    raw_bvh = WarpUnsignedDistanceBVH(*raw, max_distance=8.0)
    cases = (
        ("identity", same, 0.125, "PASSED"),
        ("parallel_shift_within", shifted, 0.125, "PASSED"),
        ("parallel_shift_excess", shifted, 0.03125, "FAILED"),
        ("asymmetric_extra_patch", with_extra, 0.125, "FAILED"),
        ("expired_deadline", same, 0.125, "UNRESOLVED"),
    )
    rows = []
    for name, candidate, budget, expected in cases:
        row = {"name": name, "raw": _hash(raw), "candidate": _hash(candidate)}
        try:
            evidence = measure_bidirectional_deviation(
                *raw,
                *candidate,
                max_deviation_mm=budget,
                deadline_monotonic=(
                    0.0 if name == "expired_deadline" else time.monotonic() + 30.0
                ),
                query_batch_size=2,
                triangle_batch_size=1,
                raw_bvh=raw_bvh,
            )
            row["evidence"] = evidence
            assert evidence["status"] == expected, evidence
            if expected == "UNRESOLVED":
                assert evidence["reason"] == "deadline", evidence
                assert evidence["sample_count"] == 0, evidence
                assert evidence["max_sampled_deviation_mm"] is None, evidence
            else:
                assert len(evidence["directions"]) == 2, evidence
                for direction in evidence["directions"]:
                    assert (
                        direction["sampled_referenced_vertices"]
                        == direction["required_referenced_vertices"]
                    ), evidence
                    assert (
                        direction["sampled_face_centroids"]
                        == direction["required_face_centroids"]
                    ), evidence
                if name == "identity":
                    assert evidence["sample_count"] == 8, evidence
                    assert evidence["max_sampled_deviation_mm"] <= 1e-6, evidence
                if name.startswith("parallel_shift"):
                    # Exact normal shift; float32 barycentric/centroid roundoff
                    # is much smaller than the independently chosen 1e-6 mm
                    # observation tolerance. No runtime acceptance margin.
                    assert abs(evidence["max_sampled_deviation_mm"] - 0.0625) <= 1e-6, (
                        evidence
                    )
                    assert evidence["fraction_over_budget"] == (
                        0.0 if expected == "PASSED" else 1.0
                    ), evidence
                if name == "asymmetric_extra_patch":
                    first, second = evidence["directions"]
                    assert first["status"] == "PASSED", evidence
                    assert second["status"] == "FAILED", evidence
                    assert abs(second["max_sampled_deviation_mm"] - 4.0) <= 1e-6, (
                        evidence
                    )
                    # 4 raw samples + 8 candidate samples; 4 extra-patch
                    # samples (3 vertices, 1 centroid) exceed the budget.
                    assert evidence["sample_count"] == 12, evidence
                    assert evidence["samples_over_budget"] == 4, evidence
                    assert evidence["fraction_over_budget"] == 1 / 3, evidence
            row["control_status"] = "passed"
        except Exception as exc:
            row["control_status"] = "failed"
            row["error"] = repr(exc)
        rows.append(row)
        (out / "deviation-controls.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = execute(args.out)
    print(json.dumps(results, indent=2))
    raise SystemExit(
        0 if all(row["control_status"] == "passed" for row in results) else 1
    )
