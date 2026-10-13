"""Compile the exact owned CuMesh sources for L4, preserving build provenance."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out, identity):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path("/opt/native-src")
    files = sorted(
        path
        for folder in (root / "src", root / "cumesh")
        for path in folder.rglob("*")
        if path.is_file()
    )
    sources = [
        "hash/hash.cu",
        "atlas.cu",
        "clean_up.cu",
        "cumesh.cu",
        "connectivity.cu",
        "geometry.cu",
        "io.cu",
        "simplify.cu",
        "shared.cu",
        "ext.cpp",
    ]
    record = {
        "source": identity,
        "files": {str(p.relative_to(root)): sha(p) for p in files},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch": "8.9",
        "max_jobs": 2,
        "status": "building",
    }
    for tool in ("nvcc", "c++"):
        result = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, check=True, timeout=10
        )
        record[tool] = result.stdout[:2048]
    record_path = out / "build.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    build_dir = out / "build"
    build_dir.mkdir()
    # The unchanged xatlas extension is not rebuilt or used by reduction. Import
    # its package path without loading the old CuMesh _C (pybind type registry).
    installed = Path(importlib.util.find_spec("cumesh").origin).parent
    xatlas = next(installed.glob("_cumesh_xatlas*.so"))
    record["reused_xatlas"] = {"path": str(xatlas), "sha256": sha(xatlas)}
    try:
        load(
            name="_C",
            sources=[str(root / "src" / path) for path in sources],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            build_directory=str(build_dir),
            verbose=True,
            with_cuda=True,
            is_python_module=False,
        )
        package = out / "package" / "cumesh"
        shutil.copytree(root / "cumesh", package)
        binary = build_dir / "_C.so"
        shutil.copy2(binary, package / "_C.so")
        shutil.copy2(xatlas, package / xatlas.name)
        record.update(
            status="built",
            native_sha256=sha(binary),
            python_sha256=sha(package / "cumesh.py"),
            package_path=str(package.parent),
        )
    except Exception as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out, json.loads(args.identity))), flush=True)
