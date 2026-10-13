"""Native cleanup must retain attached fan fragments without welding the mesh."""
import numpy as np
import unittest
import torch
from cumesh import CuMesh



def _mesh(vertices, faces):
    mesh = CuMesh()
    mesh.init(torch.tensor(vertices, dtype=torch.float32, device="cuda"),
              torch.tensor(faces, dtype=torch.int32, device="cuda").reshape(-1, 3))
    return mesh


def _exercise_fan(split_indices):
    vertices = [[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.],
                [0., -.1, 0.], [0., 1e-5, 0.],
                [10., 0., 0.], [10.1, 0., 0.], [10., 1e-5, 0.],
                [0., 0., .1], [0., 1e-5, .1]]
    faces = [[0, 1, 2], [1, 0, 3], [0, 1, 4], [5, 6, 7], [0, 8, 9]]
    if split_indices:
        # Exact numeric aliases include signed zero. Each fan face has its own indices.
        vertices += [[-0., 0., 0.], [.1, -0., 0.], [0., 0., -0.], [.1, 0., -0.]]
        faces[1] = [11, 10, 3]
        faces[2] = [12, 13, 4]
    source = np.asarray(vertices, dtype=np.float32)[faces]
    legacy = _mesh(vertices, faces)
    legacy.remove_small_connected_components(1e-5)
    assert legacy.num_faces == 2  # Reproduces the destructive pre-fix behavior.
    mesh = _mesh(vertices, faces)
    # Existing component cache must not override the requested geometric mode.
    mesh.get_connected_components()
    mesh.remove_small_connected_components(1e-5, connectivity="geometric_edge")
    v, f = mesh.read()
    np.testing.assert_array_equal(v.cpu().numpy()[f.cpu().numpy()], source[:3])
    assert mesh.num_faces == 3
    # Generic cleanup must not reuse the alternate labels after mutation.
    mesh.remove_small_connected_components(1e-5)
    assert mesh.num_faces == 2


@unittest.skipUnless(torch.cuda.is_available(), "requires native CUDA CuMesh")
class GeometricComponentCleanupTests(unittest.TestCase):
    def test_preserves_fan_but_removes_islands_and_point_contacts(self):
        for split_indices in [False, True]:
            with self.subTest(split_indices=split_indices):
                _exercise_fan(split_indices)

    def test_keeps_threshold_equal_area_and_removes_smaller_component(self):
        mesh = _mesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                      [2., 0., 0.], [2.5, 0., 0.], [2., .5, 0.]], [[0, 1, 2], [3, 4, 5]])
        mesh.remove_small_connected_components(.5, connectivity="geometric_edge")
        self.assertEqual(mesh.num_faces, 1)
        v, f = mesh.read()
        np.testing.assert_array_equal(v.cpu().numpy()[f.cpu().numpy()],
                                      [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])

    def test_empty_surface_and_invalid_parameters(self):
        empty = _mesh([[0., 0., 0.]], [])
        empty.remove_small_connected_components(1e-5, connectivity="geometric_edge")
        self.assertEqual(empty.num_faces, 0)
        mesh = _mesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]])
        for area in [float("nan"), float("inf"), -1.]:
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                mesh.remove_small_connected_components(area, connectivity="geometric_edge")
        with self.assertRaisesRegex(ValueError, "unsupported component connectivity"):
            mesh.remove_small_connected_components(1e-5, connectivity="vertex")
        mesh.remove_small_connected_components(1e-5, connectivity="geometric_edge")
        self.assertEqual(mesh.num_faces, 1)

    def test_can_remove_every_face_without_invalid_empty_launch(self):
        mesh = _mesh([[0., 0., 0.], [1., 0., 0.]], [[0, 0, 0], [1, 1, 1]])
        mesh.remove_small_connected_components(1e-5, connectivity="geometric_edge")
        self.assertEqual(mesh.num_faces, 0)
        self.assertEqual(mesh.num_vertices, 0)
        vertices, faces = mesh.read()
        self.assertEqual(tuple(vertices.shape), (0, 3))
        self.assertEqual(tuple(faces.shape), (0, 3))
        mesh.remove_small_connected_components(1e-5)
