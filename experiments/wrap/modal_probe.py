"""Offline dev L4 WRAP evaluation; every input and result is volume-persisted."""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("tiny3d-s4-wrap-corpus-v1")
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
def campaign(records: list[dict], run_id: str):
    import subprocess
    import time
    import threading
    import pynvml

    root = Path("/evidence/make-solid-wrap") / run_id
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []
    samples = []
    done = threading.Event()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sample():
        while not done.wait(0.1):
            samples.append(
                {
                    "seconds": time.monotonic() - started,
                    "used": pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                }
            )

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        for record in records:
            if time.monotonic() - started > 760:
                rows.append(
                    {"name": record["name"], "status": "not_run_campaign_budget"}
                )
                continue
            record = {**record, "path": "/evidence/" + record["volumePath"]}
            with (root / (record["name"] + ".log")).open("w") as log:
                try:
                    result = subprocess.run(
                        [
                            "/opt/mesh-generation-service/.venv/bin/python",
                            "/opt/wrap/probe.py",
                            "--record",
                            json.dumps(record),
                            "--out",
                            str(root / record["name"]),
                            "--max-seconds",
                            "220",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=250,
                    )
                    rows.append({"name": record["name"], "exitCode": result.returncode})
                except Exception as error:
                    rows.append({"name": record["name"], "error": repr(error)})
            (root / "campaign.json").write_text(
                json.dumps(
                    {"rows": rows, "seconds": time.monotonic() - started}, indent=2
                )
                + "\n"
            )
            volume.commit()
    finally:
        done.set()
        sampler.join()
        (root / "nvml.json").write_text(json.dumps(samples) + "\n")
        volume.commit()
    return {
        "path": str(root / "campaign.json"),
        "seconds": time.monotonic() - started,
        "rows": rows,
        "sampledVramPeakBytes": max((s["used"] for s in samples), default=None),
    }


@app.local_entrypoint()
def main(records: str, run_id: str):
    data = json.loads(Path(records).read_text())
    call = campaign.spawn(data, run_id)
    print(
        json.dumps({"appId": app.app_id, "callId": call.object_id, "runId": run_id}),
        flush=True,
    )
    print(json.dumps(call.get()), flush=True)
