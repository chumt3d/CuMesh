"""Dev L4 pilot: sampled controls then three exact raw-WRAP reductions."""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("tiny3d-s4-wrap-reduction-pilot-v1")
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
    memory=(8192, 8192),
    timeout=850,
    startup_timeout=60,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    volumes={"/evidence": volume},
)
def pilot(run_id: str, source_identity: dict, build_id: str):
    import os
    import subprocess
    import time

    volume.reload()
    build_dir = Path("/evidence/make-solid-wrap/native") / build_id
    build = json.loads((build_dir / "build.json").read_text())
    if build["status"] != "built":
        raise ValueError("root_fixed_native_build_required")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = build["package_path"] + ":/opt/wrap"
    environment["WRAP_NATIVE_BUILD_JSON"] = str(build_dir / "build.json")
    directory = Path("/evidence/make-solid-wrap") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_identity = {
        **source_identity,
        "files": {
            str(path.relative_to("/opt/wrap")): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in Path("/opt/wrap").rglob("*.py")
        },
    }
    rows = []
    try:
        with (directory / "controls.log").open("w") as log:
            control = subprocess.run(
                [
                    "/opt/mesh-generation-service/.venv/bin/python",
                    "/opt/wrap/deviation_controls.py",
                    "--out",
                    str(directory / "controls"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
                env=environment,
            )
        rows.append({"name": "sample_controls", "exit_code": control.returncode})
        volume.commit()
        if control.returncode != 0:
            return {"rows": rows, "status": "controls_failed", "path": str(directory)}
        for name in ("palm", "elf", "stroller"):
            with (directory / (name + ".log")).open("w") as log:
                try:
                    process = subprocess.run(
                        [
                            "/opt/mesh-generation-service/.venv/bin/python",
                            "/opt/wrap/reduction_probe.py",
                            "--raw",
                            "/evidence/make-solid-wrap/priority20261013v2/" + name,
                            "--out",
                            str(directory / name),
                            "--max-seconds",
                            "220",
                            "--source-identity",
                            json.dumps(source_identity),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=250,
                        env=environment,
                    )
                    rows.append({"name": name, "exit_code": process.returncode})
                except subprocess.TimeoutExpired:
                    rows.append(
                        {"name": name, "status": "subprocess_deadline_raw_preserved"}
                    )
            (directory / "campaign.json").write_text(
                json.dumps(
                    {"rows": rows, "seconds": time.monotonic() - started}, indent=2
                )
                + "\n"
            )
            volume.commit()
    finally:
        (directory / "execution.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "source": source_identity,
                    "seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n"
        )
        volume.commit()
    return {"rows": rows, "seconds": time.monotonic() - started, "path": str(directory)}


@app.local_entrypoint()
def main(run_id: str, source_identity: str, build_id: str):
    identity = json.loads(Path(source_identity).read_text())
    call = pilot.spawn(run_id, identity, build_id)
    print(
        json.dumps({"appId": app.app_id, "callId": call.object_id, "runId": run_id}),
        flush=True,
    )
    print(json.dumps(call.get()), flush=True)
