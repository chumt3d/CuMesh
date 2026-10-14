"""Real WRAP frame validity and zero-ended grid intervals; no equivalence claim."""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from cumesh_wrap import WrapFailure, WrapLimits, WrapSettings, wrap_surface
from cumesh_wrap.reduction import inspect_surface


def run(out):
    import cumesh_wrap.engine as engine

    out.mkdir(parents=True, exist_ok=False)
    vertices = np.array(
        [[x, y, z] for z in (-2, 2) for y in (-1.5, 1.5) for x in (-1, 1)],
        dtype=np.float64,
    )
    faces = torch.tensor(
        [
            [0, 2, 3],
            [0, 3, 1],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    frames = (
        ("origin", np.eye(3), [0, 0, 0]),
        ("origin_tilted", rz @ ry @ rx, [0, 0, 0]),
        ("print_offset_rotated", rz, [13, -7, 41.94254684448242]),
        ("far_offset_rotated", rz @ ry @ rx, [12345, -23456, 34567]),
    )
    rows = []
    for name, rotation, offset in frames:
        stored = np.ascontiguousarray(vertices @ rotation.T + offset, dtype=np.float32)
        row = {
            "name": name,
            "rotation": rotation.tolist(),
            "translationMm": offset,
            "inputSha256": hashlib.sha256(stored.tobytes()).hexdigest(),
            "runtime": {
                "wheelVersion": version("cumesh-wrap"),
                "engineSha256": hashlib.sha256(
                    Path(engine.__file__).read_bytes()
                ).hexdigest(),
            },
        }
        try:
            v, f, evidence = wrap_surface(
                torch.tensor(stored, device="cuda"),
                faces,
                settings=WrapSettings(),
                limits=WrapLimits(max_seconds=12),
            )
            topology, passed = inspect_surface(
                v, f, deadline_monotonic=time.monotonic() + 12
            )
            row.update(contract_passed=passed, topology=topology, evidence=evidence)
        except WrapFailure as error:
            row.update(contract_passed=False, code=error.code, evidence=error.evidence)
        rows.append(row)
    (out / "frame-controls.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        json.dumps(
            {"frames": len(rows), "passed": sum(r["contract_passed"] for r in rows)}
        )
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
