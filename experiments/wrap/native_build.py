"""Compile the exact owned CuMesh sources for L4, preserving build provenance."""

import argparse
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out, identity, reuse=None):
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
    record["reused_objects"] = []
    changed_objects = []
    manual_reuse = False
    if reuse:
        previous = json.loads((reuse / "build.json").read_text())
        headers = {
            key: value
            for key, value in record["files"].items()
            if key.startswith("src/") and not key.endswith((".cu", ".cpp"))
        }
        same_headers = all(
            previous["files"].get(key) == value for key, value in headers.items()
        )
        same_tools = all(
            previous.get(key) == record.get(key)
            for key in ("torch", "cuda", "arch", "nvcc", "c++")
        )
        if previous["status"] == "built" and same_headers and same_tools:
            # Mounted-source timestamps invalidate Ninja's dependency database
            # across containers. Reuse hash-qualified objects explicitly, compile
            # changed TUs with the recorded Ninja commands, then relink once.
            shutil.copy2(reuse / "build" / "build.ninja", build_dir / "build.ninja")
            manual_reuse = True
            for relative in sources:
                key = "src/" + relative
                object_name = Path(relative).stem + (
                    ".cuda.o" if relative.endswith(".cu") else ".o"
                )
                old_object = reuse / "build" / object_name
                if (
                    previous["files"].get(key) == record["files"][key]
                    and old_object.is_file()
                ):
                    target = build_dir / object_name
                    shutil.copy2(old_object, target)
                    record["reused_objects"].append(
                        {"source": key, "object": object_name, "sha256": sha(target)}
                    )
                else:
                    changed_objects.append(object_name)
            record["reuse_authority"] = {
                "build_json_sha256": sha(reuse / "build.json"),
                "path": str(reuse),
                "rule": "identical_source_headers_toolchain_immutable_base_image",
                "ninja_sha256": sha(build_dir / "build.ninja"),
            }
    # The unchanged xatlas extension is not rebuilt or used by reduction. Import
    # its package path without loading the old CuMesh _C (pybind type registry).
    installed = Path(importlib.util.find_spec("cumesh").origin).parent
    xatlas = next(installed.glob("_cumesh_xatlas*.so"))
    record["reused_xatlas"] = {"path": str(xatlas), "sha256": sha(xatlas)}
    try:
        if manual_reuse:
            record["executed_commands"] = []
            for target in changed_objects + ["_C.so"]:
                commands = subprocess.run(
                    ["ninja", "-t", "commands", target],
                    cwd=build_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.splitlines()
                # Object targets have one compiler command; the final dependency
                # tree ends with the link command. Execute argv, never a shell.
                arguments = shlex.split(commands[-1])
                if target == "_C.so" and "-shared" not in arguments:
                    raise ValueError("native_reuse_link_command_invalid")
                if (
                    "-o" not in arguments
                    or arguments[arguments.index("-o") + 1] != target
                ):
                    raise ValueError("native_reuse_output_command_invalid")
                print(json.dumps({"native_command": arguments}), flush=True)
                subprocess.run(arguments, cwd=build_dir, check=True)
                record["executed_commands"].append(arguments)
            torch.ops.load_library(str(build_dir / "_C.so"))
        else:
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
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(build(args.out, json.loads(args.identity), args.reuse)), flush=True
    )
