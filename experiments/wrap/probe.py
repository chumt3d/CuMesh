"""Offline engine measurement on immutable runtime-contract mesh files."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import warp as wp

from cumesh_wrap import WrapFailure, WrapLimits, WrapSettings, wrap_surface


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(record, output, max_seconds):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    row = {"input": record, "status": "running", "product_validation": "not_run"}
    source = Path(record["path"])
    try:
        if digest(source) != record["sha256"]:
            raise ValueError("input_hash_mismatch")
        import trimesh

        loaded = trimesh.load(source, process=False, skip_materials=True)
        mesh = (
            loaded.dump(concatenate=True)
            if isinstance(loaded, trimesh.Scene)
            else loaded
        )
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        # Same exact float32 first-occurrence weld as the corpus: no tolerance,
        # no face removal, no hidden source cleaning.
        _, first, inverse = np.unique(
            vertices, axis=0, return_index=True, return_inverse=True
        )
        order = np.argsort(first)
        remap = np.empty_like(order)
        remap[order] = np.arange(len(order))
        vertices = np.ascontiguousarray(vertices[first[order]])
        faces = np.ascontiguousarray(remap[inverse][faces], dtype=np.int32)
        longest = float(np.ptp(vertices, axis=0).max())
        scale = 100.0 / longest
        print_vertices = np.ascontiguousarray(vertices * scale, dtype=np.float32)
        row.update(
            {
                "printLongestSourceMm": 100.0,
                "sourceToMmScale": scale,
                "sourceVertices": len(vertices),
                "sourceFaces": len(faces),
                "canonical_v_f32_sha256": hashlib.sha256(
                    vertices.tobytes()
                ).hexdigest(),
                "canonical_f_i32_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
                "decodeSeconds": time.perf_counter() - started,
            }
        )
        np.savez(output / "source.npz", surface_vertices=vertices, surface_faces=faces)
        wp.config.kernel_cache_dir = str(output.parent / "warp-cache")
        torch.cuda.reset_peak_memory_stats()
        vv, ff, evidence = wrap_surface(
            torch.tensor(print_vertices, device="cuda"),
            torch.tensor(faces, device="cuda"),
            settings=WrapSettings(require_one_part=False),
            limits=WrapLimits(max_seconds=max_seconds),
        )
        row["engine"] = evidence
        np.savez(
            output / "wrap-print.npz",
            surface_vertices=vv.cpu().numpy(),
            surface_faces=ff.cpu().numpy(),
        )
        np.savez(
            output / "wrap-source-frame.npz",
            surface_vertices=vv.cpu().numpy().astype(np.float64) / scale,
            surface_faces=ff.cpu().numpy(),
        )
        row.update(
            status="candidate", outputSha256=digest(output / "wrap-source-frame.npz")
        )
    except WrapFailure as error:
        row.update(status="engine_refused", code=error.code, engine=error.evidence)
    except Exception as error:
        row.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    row["endToEndSeconds"] = time.perf_counter() - started
    (output / "result.json").write_text(
        json.dumps(row, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: row.get(k) for k in ("input", "status", "code", "endToEndSeconds")}
        ),
        flush=True,
    )
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=180)
    args = p.parse_args()
    run(json.loads(args.record), args.out, args.max_seconds)
