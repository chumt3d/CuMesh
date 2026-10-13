"""Dev-only CPU2 native build and L4 baseline/fixed controls; no publishing."""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parent.parent
app = modal.App("tiny3d-s4-wrap-native-fix-v1")
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
                "MAX_JOBS",
            )
        },
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "PYTHONPATH": "/opt/wrap",
    }
)
if modal.is_local():
    image = image.add_local_dir(
        HERE, "/opt/wrap", ignore=["**/__pycache__/**", "**/*.pyc"]
    )
    image = image.add_local_dir(SOURCE_ROOT / "src", "/opt/native-src/src")
    image = image.add_local_dir(
        SOURCE_ROOT / "cumesh",
        "/opt/native-src/cumesh",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=900,
    startup_timeout=60,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    volumes={"/evidence": volume},
)
def compile_native(build_id: str, identity: dict):
    import subprocess

    directory = Path("/evidence/make-solid-wrap/native") / build_id
    directory.parent.mkdir(parents=True, exist_ok=True)
    log_path = directory.parent / (build_id + ".log")
    try:
        with log_path.open("w") as log:
            process = subprocess.run(
                [
                    "/opt/mesh-generation-service/.venv/bin/python",
                    "/opt/wrap/native_build.py",
                    "--out",
                    str(directory),
                    "--identity",
                    json.dumps(identity),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=850,
            )
        return {"path": str(directory), "returncode": process.returncode}
    finally:
        volume.commit()


@app.function(
    image=image,
    gpu="L4",
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=850,
    startup_timeout=60,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    volumes={"/evidence": volume},
)
def execute_controls(run_id: str, build_id: str):
    import os
    import subprocess

    volume.reload()
    directory = Path("/evidence/make-solid-wrap") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    mode = "fixed" if build_id else "baseline"
    environment = dict(os.environ)
    if build_id:
        build_dir = Path("/evidence/make-solid-wrap/native") / build_id
        build = json.loads((build_dir / "build.json").read_text())
        if build["status"] != "built":
            raise ValueError("native_build_not_successful")
        environment["PYTHONPATH"] = build["package_path"] + ":/opt/wrap"
        environment["WRAP_NATIVE_BUILD_JSON"] = str(build_dir / "build.json")
    try:
        with (directory / "controls.log").open("w") as log:
            process = subprocess.run(
                [
                    "/opt/mesh-generation-service/.venv/bin/python",
                    "/opt/wrap/collapse_controls.py",
                    "--out",
                    str(directory / "controls"),
                    "--mode",
                    mode,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=200,
                env=environment,
            )
        return {"path": str(directory), "returncode": process.returncode, "mode": mode}
    finally:
        volume.commit()


@app.local_entrypoint()
def main(action: str, run_id: str, identity: str = "", build_id: str = ""):
    if action == "build":
        call = compile_native.spawn(run_id, json.loads(Path(identity).read_text()))
    elif action == "controls":
        call = execute_controls.spawn(run_id, build_id)
    else:
        raise ValueError("unknown action")
    print(
        json.dumps(
            {
                "appId": app.app_id,
                "callId": call.object_id,
                "runId": run_id,
                "action": action,
            }
        ),
        flush=True,
    )
    print(json.dumps(call.get()), flush=True)
