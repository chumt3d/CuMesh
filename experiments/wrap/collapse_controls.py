"""Real-CuMesh collapse regression controls, independent of native validators.

Run baseline and fixed in SEPARATE Python processes selecting the intended
installed extension via the parent's environment. Example:
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 180s python collapse_controls.py \
      --mode baseline --out BASELINE_DIR
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 180s python collapse_controls.py \
      --mode fixed --out FIXED_DIR

The public simplify API has no cancellation hook; the parent supplies the outer
process timeout. Baseline mode records observed contract failures without an
all-pass exit requirement. Fixed mode requires all selected contracts to pass.
CPU NumPy readback audits exact indices, links and float64 zero-area predicates;
this is a bounded diagnostic control, not a production CPU completion path.
No mocked proposer, cleanup/repair pass or native validator supplies the oracle.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import cumesh


def _arrays(vertices, faces):
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _orient_closed(vertices, faces):
    """Outward orientation of these convex analytic fixtures."""
    faces = faces.copy()
    points = vertices[faces].astype(np.float64)
    normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    away = points.mean(axis=1) - vertices.astype(np.float64).mean(axis=0)
    flip = np.einsum("ij,ij->i", normal, away) < 0
    faces[flip] = faces[flip][:, (0, 2, 1)]
    return vertices, faces


def cube_lattice(n=72):
    """6*n*n quads with exact integer-key shared surface vertices."""
    vertices, faces, lookup = [], [], {}

    def vertex(point):
        point = tuple(point)
        if point not in lookup:
            lookup[point] = len(vertices)
            vertices.append(point)
        return lookup[point]

    for axis in range(3):
        free = [dimension for dimension in range(3) if dimension != axis]
        natural_sign = -1 if axis == 1 else 1
        for side in (0, n):
            outward_sign = -1 if side == 0 else 1
            for i in range(n):
                for j in range(n):
                    quad = []
                    for u, v in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        point = [0, 0, 0]
                        point[axis] = side
                        point[free[0]], point[free[1]] = u, v
                        quad.append(vertex(point))
                    pair = ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3]))
                    if natural_sign != outward_sign:
                        pair = tuple((a, c, b) for a, b, c in pair)
                    faces.extend(pair)
    vertices, faces = _arrays(vertices, faces)
    return (vertices / np.float32(n) - np.float32(0.5)), faces


def _bipyramid(short_equator=False):
    equator = (
        ((-0.01, 0, 0), (0.01, 0, 0), (0, 1, 0))
        if short_equator
        else ((-0.5, -0.5, 0), (0.5, -0.5, 0), (0, 0.5, 0))
    )
    vertices, faces = _arrays(
        equator + ((0, 0, 1), (0, 0, -1)),
        ((3, 0, 1), (3, 1, 2), (3, 2, 0), (4, 1, 0), (4, 2, 1), (4, 0, 2)),
    )
    return _orient_closed(vertices, faces)


def fixture(name):
    if name == "cube72":
        return cube_lattice()
    if name in ("terminal_tetra", "terminal_public"):
        return _orient_closed(
            *_arrays(
                ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
                ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)),
            )
        )
    if name in ("invalid_equator", "safe_bipyramid", "nan_no_commit"):
        vertices, faces = _bipyramid(name == "invalid_equator")
        if name == "nan_no_commit":
            # Every face and every edge's incident geometry is nonfinite, while
            # safe apex-equator links still exist. A tetra would confound this
            # numeric guard with the independent terminal-topology guard.
            vertices[3:, 0] = np.nan
        return vertices, faces
    if name == "boundary_disk":
        # Only edge (0,1) has length²<0.01; its boundary contraction leaves one
        # positively oriented triangle. All face QEM planes are z=0.
        return _arrays(
            ((0, 0, 0), (0.02, 0, 0), (1, 1, 0), (0, 1, 0)),
            ((0, 1, 2), (0, 2, 3)),
        )
    if name == "boundary_chord":
        return _arrays(
            ((-0.01, 0, 0), (0, -1, 0), (0.01, 0, 0), (0, 1, 0)), ((0, 1, 2), (0, 2, 3))
        )
    if name == "last_triangle":
        return _arrays(((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),))
    if name == "offset_disk":
        vertices, faces = fixture("boundary_disk")
        vertices[:, 2] = vertices[:, 0] + vertices[:, 1]
        return vertices + np.float32(100000), faces
    if name == "disconnected_fan":
        # Two nonterminal convex bipyramids in opposite z half-spaces meet only
        # at vertex0. Only (0,1) is short; its link within the first fan is legal,
        # so refusing it specifically exercises the disconnected endpoint fan.
        first_v, first_f = _orient_closed(
            *_arrays(
                ((0, 0, 0), (0, 0, 0.02), (2, 0, 0.02), (0, 2, 0.02), (0.5, 0.5, 2)),
                ((4, 1, 2), (4, 2, 3), (4, 3, 1), (0, 2, 1), (0, 3, 2), (0, 1, 3)),
            )
        )
        second_v, second_f = _orient_closed(
            *_arrays(
                ((-2, -1, -2), (1, -2, -2), (1, 2, -2), (0, 0, 0), (0, 0, -4)),
                ((3, 0, 1), (3, 1, 2), (3, 2, 0), (4, 1, 0), (4, 2, 1), (4, 0, 2)),
            )
        )
        remap = np.array((5, 6, 7, 0, 8), dtype=np.int32)
        return np.concatenate((first_v, second_v[[0, 1, 2, 4]])), np.concatenate(
            (first_f, remap[second_f])
        )
    if name == "repeated_index":
        return _arrays(((0, 0, 0), (1, 0, 0)), ((0, 1, 1),))
    raise ValueError(name)


def _components(vertex_count, edges, referenced):
    parent = list(range(vertex_count))

    def root(index):
        index = int(index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    return len({root(index) for index in referenced})


def audit(vertices, faces):
    """Independent index topology and exact-zero float32-position diagnostics."""
    report = {
        "vertices": len(vertices),
        "faces": len(faces),
        "nonfinite_vertices": int((~np.isfinite(vertices).all(axis=1)).sum()),
        "invalid_indices": int(((faces < 0) | (faces >= len(vertices))).sum()),
        "boundary_edges": None,
        "nonmanifold_edges": None,
        "orientation_conflicts": None,
        "duplicate_faces": None,
        "zero_area_faces": None,
        "nonfinite_faces": None,
        "referenced_parts": None,
        "referenced_vertices": None,
        "invalid_vertex_links": None,
        "boundary_loops": None,
        "boundary_degree_violations": None,
        "euler_characteristic": None,
        "signed_volume": None,
        "geometry_predicate": "float32 positions promoted to float64, exact-zero cross-product test",
        "self_intersections": "not_checked_by_this_control",
    }
    if report["invalid_indices"] or not len(faces):
        return report
    directed = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    winding = np.bincount(
        inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)
    )
    boundary = edges[counts == 1]
    referenced = np.unique(faces)
    canonical = np.sort(faces, axis=1)
    points = vertices[faces].astype(np.float64)
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    finite_faces = np.isfinite(points).all(axis=(1, 2))
    links = {}
    for a, b, c in faces:
        for vertex, x, y in ((a, b, c), (b, c, a), (c, a, b)):
            links.setdefault(int(vertex), []).append((int(x), int(y)))
    boundary_vertices, degrees = np.unique(boundary, return_counts=True)
    boundary_vertex_set = set(map(int, boundary_vertices))
    bad_links = 0
    for vertex, pairs in links.items():
        neighbors, neighbor_degrees = np.unique(np.asarray(pairs), return_counts=True)
        local_index = {int(value): index for index, value in enumerate(neighbors)}
        local_edges = [(local_index[a], local_index[b]) for a, b in pairs]
        connected = _components(len(neighbors), local_edges, range(len(neighbors))) == 1
        if vertex in boundary_vertex_set:
            valid_degrees = np.count_nonzero(neighbor_degrees == 1) == 2 and np.all(
                (neighbor_degrees == 1) | (neighbor_degrees == 2)
            )
        else:
            valid_degrees = len(neighbors) >= 3 and np.all(neighbor_degrees == 2)
        bad_links += int(not (connected and valid_degrees))
    report.update(
        boundary_edges=int(len(boundary)),
        nonmanifold_edges=int((counts > 2).sum()),
        orientation_conflicts=int(((counts == 2) & (winding != 0)).sum()),
        duplicate_faces=int(len(faces) - len(np.unique(canonical, axis=0))),
        zero_area_faces=int((finite_faces & (normals == 0).all(axis=1)).sum()),
        nonfinite_faces=int((~finite_faces).sum()),
        referenced_parts=_components(len(vertices), edges, referenced),
        referenced_vertices=int(len(referenced)),
        invalid_vertex_links=bad_links,
        boundary_loops=(
            _components(len(vertices), boundary, boundary_vertices)
            if len(boundary)
            else 0
        ),
        boundary_degree_violations=int((degrees != 2).sum()),
        euler_characteristic=int(len(referenced) - len(edges) + len(faces)),
        signed_volume=(
            float(
                np.einsum(
                    "ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])
                ).sum()
                / 6
            )
            if bool(finite_faces.all())
            else None
        ),
        nonpositive_z_faces=int((finite_faces & (normals[:, 2] <= 0)).sum()),
    )
    return report


def valid_surface(report, closed):
    defects = (
        "invalid_indices",
        "nonfinite_vertices",
        "nonfinite_faces",
        "duplicate_faces",
        "zero_area_faces",
        "nonmanifold_edges",
        "orientation_conflicts",
        "invalid_vertex_links",
    )
    valid = report["faces"] > 0 and all(report[key] == 0 for key in defects)
    valid = valid and report["referenced_parts"] == 1
    if closed:
        return bool(
            valid
            and report["boundary_edges"] == 0
            and report["euler_characteristic"] == 2
            and report["signed_volume"] > 0
        )
    return bool(
        valid
        and report["boundary_edges"] > 0
        and report["boundary_loops"] == 1
        and report["boundary_degree_violations"] == 0
        and report["euler_characteristic"] == 1
    )


def _hash_arrays(vertices, faces):
    return {
        "vertices_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
        "faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
    }


def _runtime():
    extension = Path(cumesh._C.__file__)
    digest = hashlib.sha256()
    with extension.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cumesh_python_path": str(Path(cumesh.__file__).resolve()),
        "extension_path": str(extension.resolve()),
        "extension_sha256": digest.hexdigest(),
        "pid": os.getpid(),
    }


def execute(out, mode, selected):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    runtime = _runtime()
    names = (
        "cube72",
        "terminal_tetra",
        "invalid_equator",
        "safe_bipyramid",
        "boundary_disk",
        "nan_no_commit",
        "terminal_public",
        "boundary_chord",
        "last_triangle",
        "offset_disk",
        "disconnected_fan",
        "repeated_index",
    )
    if selected != "all":
        names = (selected,)
    rows = []
    for name in names:
        start = time.monotonic()
        destination = out / name
        destination.mkdir(exist_ok=True)
        vertices, faces = fixture(name)
        initial = audit(vertices, faces)
        np.savez(destination / "input.npz", vertices=vertices, faces=faces)
        row = {
            "name": name,
            "mode": mode,
            "runtime": runtime,
            "initial": initial,
            "input": _hash_arrays(vertices, faces),
            "contract_passed": False,
        }
        try:
            if name not in ("nan_no_commit", "disconnected_fan", "repeated_index"):
                assert valid_surface(
                    initial,
                    name
                    not in (
                        "boundary_disk",
                        "boundary_chord",
                        "last_triangle",
                        "offset_disk",
                    ),
                ), initial
            mesh = cumesh.CuMesh()
            mesh.init(
                torch.from_numpy(vertices.copy()).to("cuda"),
                torch.from_numpy(faces.copy()).to("cuda"),
            )
            operation_start = time.monotonic()
            typed_stop = None
            if name in ("cube72", "terminal_public", "offset_disk"):
                target = (
                    len(faces) // 5
                    if name == "cube72"
                    else (2 if name == "terminal_public" else 1)
                )
                row["operation"] = {
                    "api": "CuMesh.simplify",
                    "target_faces": target,
                    "reduction_factor": 5,
                }
                try:
                    mesh.simplify(target, verbose=False)
                except RuntimeError as error:
                    if getattr(error, "code", None) != "simplify_no_legal_progress":
                        raise
                    typed_stop = {
                        "code": error.code,
                        "remaining_faces": error.remaining_faces,
                        "target_faces": error.target_faces,
                    }
                row["typed_stop"] = typed_stop
                row["native_steps"] = getattr(mesh, "last_simplification", None)
            else:
                threshold = (
                    0.01
                    if name
                    in (
                        "invalid_equator",
                        "boundary_disk",
                        "boundary_chord",
                        "disconnected_fan",
                    )
                    else 1000.0
                )
                row["operation"] = {
                    "api": "CuMesh.cu_mesh.simplify_step",
                    "lambda_edge_length": 1.0,
                    "lambda_skinny": 0.0,
                    "threshold": threshold,
                    "steps": 1,
                }
                if name == "invalid_equator":
                    row["candidate_exposure"] = {
                        "only_threshold_eligible_edge": [0, 1],
                        "edge_length_squared": float(
                            np.square(
                                vertices[1].astype(np.float64) - vertices[0]
                            ).sum()
                        ),
                        "ideal_qem_upper": 4 * 0.01**2,
                        "other_edge_length_squared_lower": 1.0,
                        "reason": "short equator has an extra common link vertex; every other edge exceeds threshold by length cost",
                    }
                row["native_return"] = tuple(
                    mesh.cu_mesh.simplify_step(1.0, 0.0, threshold, False)
                )
            output_vertices, output_faces = mesh.read()
            torch.cuda.synchronize()
            row["native_seconds"] = time.monotonic() - operation_start
            output_vertices = output_vertices.cpu().numpy().copy()
            output_faces = output_faces.cpu().numpy().copy()
            np.savez(
                destination / "output.npz", vertices=output_vertices, faces=output_faces
            )
            final = audit(output_vertices, output_faces)
            row.update(final=final, output=_hash_arrays(output_vertices, output_faces))
            if name == "repeated_index":
                row["contract_passed"] = bool(
                    np.array_equal(output_vertices, vertices)
                    and np.array_equal(output_faces, faces)
                    and final["zero_area_faces"] == 1
                )
                row["invalid_geometry_disclosure"] = (
                    "repeated-index input retained without unsafe adjacency writes"
                )
            elif name == "disconnected_fan":
                row["contract_passed"] = bool(
                    initial["invalid_vertex_links"] == 1
                    and initial["nonmanifold_edges"] == 0
                    and np.array_equal(output_vertices, vertices)
                    and np.array_equal(output_faces, faces)
                    and final["invalid_vertex_links"] == 1
                )
                row["invalid_geometry_disclosure"] = (
                    "pinched input retained; never a valid printable surface"
                )
            elif name == "nan_no_commit":
                row["contract_passed"] = bool(
                    np.array_equal(output_vertices, vertices, equal_nan=True)
                    and np.array_equal(output_faces, faces)
                    and final["nonfinite_faces"] == len(faces)
                )
                row["invalid_geometry_disclosure"] = (
                    "input and retained output are nonfinite; never a valid printable surface"
                )
            else:
                valid = valid_surface(
                    final,
                    name
                    not in (
                        "boundary_disk",
                        "boundary_chord",
                        "last_triangle",
                        "offset_disk",
                    ),
                )
                if name == "cube72":
                    valid = valid and 4 <= len(output_faces) <= target
                elif name in ("terminal_tetra", "terminal_public"):
                    valid = valid and len(output_faces) == 4
                    if name == "terminal_public":
                        valid = valid and typed_stop is not None
                elif name == "invalid_equator":
                    valid = valid and len(output_faces) == 6
                elif name == "safe_bipyramid":
                    valid = valid and len(output_faces) == 4
                elif name in ("boundary_chord", "last_triangle"):
                    valid = (
                        valid
                        and np.array_equal(output_vertices, vertices)
                        and np.array_equal(output_faces, faces)
                    )
                else:
                    valid = (
                        valid
                        and len(output_faces) == 1
                        and final["nonpositive_z_faces"] == 0
                    )
                row["contract_passed"] = bool(valid)
            del mesh, output_vertices, output_faces
        except Exception as error:
            row["error"] = repr(error)
        row["end_to_end_seconds"] = time.monotonic() - start
        rows.append(row)
        (destination / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        (out / "collapse-controls.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "fixed"), required=True)
    parser.add_argument(
        "--case",
        choices=(
            "all",
            "cube72",
            "terminal_tetra",
            "invalid_equator",
            "safe_bipyramid",
            "boundary_disk",
            "nan_no_commit",
            "terminal_public",
            "boundary_chord",
            "last_triangle",
            "offset_disk",
            "disconnected_fan",
            "repeated_index",
        ),
        default="all",
    )
    args = parser.parse_args()
    results = execute(args.out, args.mode, args.case)
    print(json.dumps(results, indent=2))
    raise SystemExit(
        0
        if args.mode == "baseline" or all(row["contract_passed"] for row in results)
        else 1
    )
