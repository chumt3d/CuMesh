"""Explicitly authorized dev-only GPU controls, not deployment or product gate."""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("tiny3d-s4-wrap-controls-v1")
volume = modal.Volume.from_name("mesh-probe-artifacts", environment_name="dev")
image = modal.Image.from_id("im-J3CB2UHwcJ68K9lFkQbmnA").env(
    {
        **{
            key: "2"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        },
        "PYTHONPATH": "/opt/wrap",
    }
)
if modal.is_local():
    image = image.add_local_dir(
        HERE, "/opt/wrap", ignore=["**/__pycache__/**", "**/*.pyc"]
    )


@app.function(
    image=image,
    gpu="L4",
    cpu=(2, 2),
    memory=(4096, 8192),
    timeout=300,
    startup_timeout=60,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    volumes={"/evidence": volume},
)
def control(run_id: str):
    import hashlib
    import subprocess
    import time

    directory = Path("/evidence/make-solid-wrap") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    result = {"runId": run_id, "imageId": "im-J3CB2UHwcJ68K9lFkQbmnA"}
    result["files"] = {
        str(p.relative_to("/opt/wrap")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("/opt/wrap").rglob("*.py")
    }
    try:
        with (directory / "control.log").open("w") as handle:
            run = subprocess.run(
                [
                    "/opt/mesh-generation-service/.venv/bin/python",
                    "/opt/wrap/controls.py",
                    "--out",
                    str(directory / "controls"),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=270,
            )
        result["returncode"] = run.returncode
    except Exception as error:
        result.update(returncode=-1, error=repr(error))
    result["seconds"] = time.monotonic() - t0
    (directory / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return {
        "path": str(directory / "execution.json"),
        "returncode": result["returncode"],
        "seconds": result["seconds"],
    }


@app.local_entrypoint()
def main(run_id: str):
    call = control.spawn(run_id)
    print(
        json.dumps({"appId": app.app_id, "callId": call.object_id, "runId": run_id}),
        flush=True,
    )
    print(json.dumps(call.get()), flush=True)
