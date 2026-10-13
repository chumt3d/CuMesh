"""Bounded executable acceptance controls; mesh outputs and failures retained."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from cumesh_wrap import WrapFailure, WrapLimits, WrapSettings, wrap_surface
from cumesh_wrap.engine import _component_labels


def component_controls(out):
    """Long chain, two chains and isolated point expose traversal failures."""
    wp.init()
    stream = wp.stream_from_torch(torch.cuda.current_stream())
    count = 100000
    x = torch.arange(count - 1, device="cuda", dtype=torch.int32)
    edges = torch.stack((x, x + 1), dim=1)
    started = time.perf_counter()

    def check(stage):
        if time.perf_counter() - started > 20:
            raise RuntimeError("component_control_deadline at " + stage)

    labels, rounds = _component_labels(edges, count, stream, check)
    assert torch.unique(labels).numel() == 1
    cut = count // 2 - 1
    edges = torch.cat((edges[:cut], edges[cut + 1 :]))
    labels, split_rounds = _component_labels(edges, count + 1, stream, check)
    assert torch.unique(labels).numel() == 3
    assert labels[0] == labels[count // 2 - 1]
    assert labels[count // 2] == labels[count - 1]
    assert labels[0] != labels[count // 2]
    row = {
        "vertices": count + 1,
        "chain_rounds": rounds,
        "split_rounds": split_rounds,
        "seconds": time.perf_counter() - started,
        "status": "passed",
    }
    (out / "components.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def cube(size):
    v = np.array(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype=np.float32,
    )
    v = (v - 0.5) * size
    f = np.array(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
        dtype=np.int32,
    )
    return v, f


def execute(out, tiny=False):
    out.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(out / "warp-cache")
    wp.config.max_unroll = 16
    torch.set_num_threads(2)
    component_controls(out)
    size = 4.0 if tiny else 10.0
    v, f = cube(size)
    iv, iff = cube(size / 2)
    sheet = np.array(
        (
            (-size / 2, -size / 2, 0),
            (size / 2, -size / 2, 0),
            (size / 2, size / 2, 0),
            (-size / 2, size / 2, 0),
        ),
        dtype=np.float32,
    )
    cases = {
        "closed_cube": (v, f),
        "nested_shell": (np.concatenate((v, iv)), np.concatenate((f, iff + 8))),
        "open_sheet": (sheet, np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int32)),
    }
    results = []
    for name, (vertices, faces) in cases.items():
        destination = out / name
        destination.mkdir(exist_ok=True)
        start = time.perf_counter()
        row = {
            "name": name,
            "input_v_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
            "input_f_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
        }
        try:
            torch.cuda.reset_peak_memory_stats()
            vv, ff, evidence = wrap_surface(
                torch.tensor(vertices, device="cuda"),
                torch.tensor(faces, device="cuda"),
                settings=WrapSettings(),
                limits=WrapLimits(max_cuda_bytes=2 * 1024**3, max_seconds=90),
            )
            np.savez(
                destination / "mesh.npz",
                surface_vertices=vv.cpu().numpy(),
                surface_faces=ff.cpu().numpy(),
            )
            row.update(evidence)
            if name != "open_sheet":
                # A hollow UDF shell has volume O(size^2*offset), not size^3.
                assert size**3 <= evidence["signed_volume_mm3"] <= (size + 1) ** 3
            assert evidence["topology"]["parts"] == 1
            assert evidence["wall_lower_bound_estimate_mm"] >= 0.4 - 1e-6
            import trimesh

            mesh = trimesh.Trimesh(vv.cpu().numpy(), ff.cpu().numpy(), process=False)
            row["independent_trimesh"] = {
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "volume": float(mesh.volume),
            }
            assert mesh.is_watertight and mesh.is_winding_consistent
            if name == "open_sheet":
                origins = np.array(
                    ((0, 0, -size), (size / 4, 0, -size), (0, size / 4, -size))
                )
                rays = np.tile((0, 0, 1), (3, 1))
                locations, ray_ids, _ = mesh.ray.intersects_location(
                    origins, rays, multiple_hits=True
                )
                walls = []
                for ray in range(3):
                    zz = np.unique(np.round(locations[ray_ids == ray, 2], 6))
                    assert len(zz) == 2
                    walls.append(float(zz[1] - zz[0]))
                row["independent_sheet_ray_walls_mm"] = walls
                assert min(walls) >= 0.4
            row["status"] = "passed"
        except WrapFailure as exc:
            row.update(status="failed", code=exc.code, evidence=exc.evidence)
        except Exception as exc:
            import traceback

            row.update(
                status="failed", error=repr(exc), traceback=traceback.format_exc()
            )
        row["end_to_end_seconds"] = time.perf_counter() - start
        (destination / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        results.append(row)
        (out / "controls.json").write_text(json.dumps(results, indent=2) + "\n")
        torch.cuda.empty_cache()
        if row["status"] != "passed":
            break
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    rows = execute(args.out, args.tiny)
    print(json.dumps(rows, indent=2))
    raise SystemExit(
        0 if len(rows) == 3 and all(r["status"] == "passed" for r in rows) else 1
    )
