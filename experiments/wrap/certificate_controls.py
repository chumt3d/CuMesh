"""Real public API regressions for certificate reuse, invalidation and recovery."""
import argparse
import ctypes
import json
import time
from pathlib import Path

import cumesh
import numpy as np
import torch

from admission_controls import arrays, mesh_for
from admission_geometry import exact_codes
from collapse_controls import cube_lattice


def advance(mesh, quadrics=None, seconds=10.0):
    if quadrics is None:
        quadrics = mesh.cu_mesh.initial_simplification_quadrics()
    value = mesh.cu_mesh.simplify_step_accumulated(quadrics, 1.0, 0.0, 1000.0, float('inf'), False, seconds)
    return value[2], mesh.cu_mesh.simplification_admission_statistics()


def cache_controls(out, bridge):
    library = ctypes.CDLL(str(bridge))
    library.exact_pair_codes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    vertices, faces = cube_lattice(8)
    rows = []

    def record(name, run):
        started = time.monotonic()
        data = run()
        rows.append({'name': name, 'seconds': time.monotonic()-started, **data})
        (out / 'certificate-controls.json').write_text(json.dumps(rows, indent=2) + '\n')
        assert data['passed'], rows[-1]

    def continuation():
        mesh = mesh_for(vertices, faces)
        try:
            q, first = advance(mesh)
            # Intentionally no additional sync between the two native steps.
            q, second = advance(mesh, q)
            v, f = arrays(mesh)
            exact = exact_codes(v, f, library)
            return {'passed': first.get('inputAuditReused', 0) == 0 and first['inputQueries'] == len(faces)
                    and second.get('inputAuditReused', 0) == 1 and second['inputQueries'] == 0
                    and second['acceptedCollapses'] > 0 and exact == {'intersections': 0, 'undecided': 0},
                    'first': first, 'second': second, 'exact': exact}
        finally:
            mesh.release_async(synchronize=True)
    record('consecutive_admitted_steps_reuse_after_existing_sync', continuation)

    def clear_cache():
        mesh = mesh_for(vertices, faces)
        try:
            q, _ = advance(mesh)
            mesh.clear_cache()
            _, result = advance(mesh, q)
            v, f = arrays(mesh)
            exact = exact_codes(v, f, library)
            return {'passed': result['inputAuditReused'] == 1 and result['inputQueries'] == 0
                    and exact == {'intersections': 0, 'undecided': 0}, 'statistics': result, 'exact': exact}
        finally:
            mesh.release_async(synchronize=True)
    record('successful_cache_only_clear_preserves_pure_proof', clear_cache)

    mutations = {
        'remove_faces': lambda mesh: mesh.remove_faces(torch.ones(mesh.num_faces, dtype=torch.bool, device='cuda')),
        'remove_unreferenced_vertices': lambda mesh: mesh.remove_unreferenced_vertices(),
        'remove_duplicate_faces': lambda mesh: mesh.remove_duplicate_faces(),
        'remove_degenerate_faces': lambda mesh: mesh.remove_degenerate_faces(),
        'fill_holes': lambda mesh: mesh.fill_holes(),
        'repair_non_manifold_edges': lambda mesh: mesh.repair_non_manifold_edges(),
        'remove_non_manifold_faces': lambda mesh: mesh.remove_non_manifold_faces(),
        'remove_small_connected_components': lambda mesh: mesh.remove_small_connected_components(0.0),
        'unify_face_orientations': lambda mesh: mesh.unify_face_orientations(),
    }
    for name, mutate in mutations.items():
        def mutation(mutate=mutate):
            mesh = mesh_for(vertices, faces)
            try:
                advance(mesh)
                mutate(mesh)
                current_faces = mesh.num_faces
                # A geometry edit requires newly initialized caller quadrics.
                _, result = advance(mesh)
                v, f = arrays(mesh)
                exact = exact_codes(v, f, library)
                return {'passed': result.get('selectionBatches', 0) > 0 and result['inputAuditReused'] == 0
                        and result['inputQueries'] == current_faces and exact == {'intersections': 0, 'undecided': 0},
                        'statistics': result, 'facesAudited': current_faces, 'exact': exact}
            finally:
                mesh.release_async(synchronize=True)
        record('geometry_edit_invalidates_' + name, mutation)

    def cancelled_then_cleared():
        mesh = mesh_for(vertices, faces)
        try:
            q, _ = advance(mesh)
            current_faces = mesh.num_faces
            mesh.cu_mesh.cancel_simplification()
            mesh.cu_mesh.clear_simplification_cancel()
            _, result = advance(mesh, q)
            v, f = arrays(mesh)
            exact = exact_codes(v, f, library)
            return {'passed': result['inputAuditReused'] == 0 and result['inputQueries'] == current_faces
                    and exact == {'intersections': 0, 'undecided': 0}, 'statistics': result, 'exact': exact}
        finally:
            mesh.release_async(synchronize=True)
    record('cancellation_notice_survives_public_clear', cancelled_then_cleared)

    dirty_v = np.concatenate([vertices, vertices + np.array([.05, .07, .03], dtype=np.float32)])
    dirty_f = np.concatenate([faces, faces + len(vertices)])
    def refuses_dirty(mesh):
        before_v, before_f = arrays(mesh)
        error = None
        try:
            advance(mesh)
        except RuntimeError as caught:
            error = str(caught)
        after_v, after_f = arrays(mesh)
        return {'passed': error is not None and 'simplify_spatial_input_intersection' in error
                and np.array_equal(before_v, after_v) and np.array_equal(before_f, after_f),
                'failure': error, 'geometryUnchanged': np.array_equal(before_v, after_v) and np.array_equal(before_f, after_f)}
    for clean_on_left in (True, False):
        def swapped(clean_on_left=clean_on_left):
            clean = mesh_for(vertices, faces)
            dirty = mesh_for(dirty_v, dirty_f)
            try:
                advance(clean)
                (clean if clean_on_left else dirty).swap(dirty if clean_on_left else clean)
                return refuses_dirty(clean)
            finally:
                clean.release_async(synchronize=True)
                dirty.release_async(synchronize=True)
        record('swap_invalidates_certified_' + ('left' if clean_on_left else 'right'), swapped)

    def reinitialized():
        mesh = mesh_for(vertices, faces)
        try:
            advance(mesh)
            mesh.release_async(synchronize=True)
            mesh.init(torch.from_numpy(dirty_v).cuda(), torch.from_numpy(dirty_f).cuda())
            return refuses_dirty(mesh)
        finally:
            mesh.release_async(synchronize=True)
    record('release_and_reinit_do_not_transfer_clean_proof', reinitialized)
    return {'status': 'passed', 'count': len(rows), 'rows': rows,
            'scope': 'Actual owned/Python method paths on CUDA; no untracked external C++ buffer writes'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    result = cache_controls(args.out, args.bridge)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'count': result['count']}), flush=True)
