"""Immutable raw-WRAP reduction pilot; persist candidate bytes and phase VRAM."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import warp as wp

from cumesh_wrap.reduction import ReductionSettings, reduce_surface


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def backend_identity():
    import cumesh
    from cumesh import _C

    paths = {
        "python": Path(cumesh.__file__).parent / "cumesh.py",
        "native": Path(_C.__file__),
    }
    result = {
        key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()
    }
    result.update(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        warp=wp.__version__,
        binary_source_correspondence="unverified_untrusted_proposer",
    )
    try:
        dist = importlib.metadata.distribution("cumesh")
        result["package_version"] = dist.version
        result["direct_url"] = dist.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        result["package_version"] = "metadata_unavailable"
    return result


def run(raw_directory, output, max_seconds, source_identity):
    output.mkdir(parents=True, exist_ok=False)
    wp.config.kernel_cache_dir = str(output.parent / "warp-cache")
    started = time.monotonic()
    row = {
        "status": "running",
        "raw_directory": str(raw_directory),
        "wrap_source_identity": source_identity,
        "final_product_validation": "not_run",
    }
    phase_state = {"name": "load", "seconds": 0.0}
    samples = []
    stop = threading.Event()

    def phase(name):
        phase_state.update(name=name, seconds=time.monotonic() - started)
        temporary = output / "phase.tmp"
        temporary.write_text(json.dumps(phase_state))
        temporary.replace(output / "phase.json")

    def sample():
        while not stop.wait(0.1):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                samples.append(
                    {
                        "seconds": time.monotonic() - started,
                        "phase": phase_state["name"],
                        "used_bytes": int(result.stdout.strip()) * 1024**2,
                    }
                )

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        raw_leaf = json.loads((raw_directory / "result.json").read_text())
        if raw_leaf["status"] != "candidate":
            raise ValueError("raw_input_is_not_a_candidate")
        raw_file = raw_directory / "wrap-print.npz"
        row.update(
            raw_print_npz_sha256=digest(raw_file),
            raw_input=raw_leaf["input"],
            backend=backend_identity(),
        )
        row["backend"]["python_matches_checked_base"] = (
            row["backend"]["python"]["sha256"]
            == source_identity["checked_cumesh_python_sha256"]
        )
        raw = np.load(raw_file, allow_pickle=False)
        raw_v = torch.tensor(
            raw["surface_vertices"], dtype=torch.float32, device="cuda"
        )
        raw_f = torch.tensor(raw["surface_faces"], dtype=torch.int32, device="cuda")
        source = np.load(raw_directory / "source.npz", allow_pickle=False)
        scale = raw_leaf["sourceToMmScale"]
        source_span = np.ptp(
            source["surface_vertices"].astype(np.float64) * scale, axis=0
        )
        # Existing product fidelity tolerance: eight times eps_rel=0.001 times
        # original-source bounding-box diagonal, evaluated in print millimetres.
        tolerance = 8 * 0.001 * float(np.linalg.norm(source_span))
        row.update(sourceToMmScale=scale, source_tolerance_mm=tolerance)
        torch.cuda.reset_peak_memory_stats()
        vv, ff, evidence = reduce_surface(
            raw_v,
            raw_f,
            source_tolerance_mm=tolerance,
            deadline_monotonic=started + max_seconds,
            settings=ReductionSettings(),
            phase=phase,
        )
        row["reduction"] = evidence
        row["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        phase("serialize")
        vertices, faces = vv.cpu().numpy(), ff.cpu().numpy()
        np.savez(
            output / "candidate-print.npz",
            surface_vertices=vertices,
            surface_faces=faces,
        )
        np.savez(
            output / "candidate-source-frame.npz",
            surface_vertices=vertices.astype(np.float64) / scale,
            surface_faces=faces,
        )
        row.update(
            status="candidate",
            candidate_print_sha256=digest(output / "candidate-print.npz"),
            candidate_source_frame_sha256=digest(output / "candidate-source-frame.npz"),
            candidate_v_f32_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
            candidate_f_i32_sha256=hashlib.sha256(faces.tobytes()).hexdigest(),
        )
    except Exception as error:
        row.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        stop.set()
        sampler.join()
        phase("done")
        row.update(
            seconds=time.monotonic() - started,
            sampled_driver_peak_bytes=max(
                (s["used_bytes"] for s in samples), default=None
            ),
            sampled_driver_phase_peaks={
                name: max(s["used_bytes"] for s in samples if s["phase"] == name)
                for name in sorted({s["phase"] for s in samples})
            },
            driver_measurement="nvidia_smi_100ms_samples_not_exact_peak",
        )
        (output / "driver-samples.json").write_text(json.dumps(samples) + "\n")
        (output / "result.json").write_text(
            json.dumps(row, indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "status": row["status"],
                "seconds": row["seconds"],
                "driver_peak": row["sampled_driver_peak_bytes"],
            }
        ),
        flush=True,
    )
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=220)
    parser.add_argument("--source-identity", required=True)
    args = parser.parse_args()
    run(args.raw, args.out, args.max_seconds, json.loads(args.source_identity))
