"""Warp 1.13.0 unsigned closest-point adapter for the isolated CUDA pilot."""

import math

import torch
import warp as wp


@wp.kernel
def _unsigned_distance(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    max_distance: float,
    distances: wp.array(dtype=float),
    face_ids: wp.array(dtype=wp.int32),
    barycentrics: wp.array(dtype=wp.vec3),
    return_uvw: bool,
):
    i = wp.tid()
    point = positions[i]
    query = wp.mesh_query_point_no_sign(mesh, point, max_distance)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distances[i] = wp.length(point - closest)
        face_ids[i] = query.face
        if return_uvw:
            barycentrics[i] = wp.vec3(query.u, query.v, 1.0 - query.u - query.v)
    else:
        distances[i] = max_distance
        face_ids[i] = -1
        if return_uvw:
            barycentrics[i] = wp.vec3(0.0, 0.0, 0.0)


def validate_mesh(vertices: torch.Tensor, faces: torch.Tensor) -> None:
    """Validate once before handing pointers to Warp/native CUDA code."""
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype != torch.float32:
        raise ValueError("vertices must be (N, 3) float32")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.int32:
        raise ValueError("faces must be (M, 3) int32")
    if not vertices.is_cuda or faces.device != vertices.device:
        raise ValueError("vertices and faces must be on the same CUDA device")
    if not vertices.is_contiguous() or not faces.is_contiguous():
        raise ValueError("vertices and faces must be contiguous")
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("an unsigned-distance BVH requires a nonempty mesh")
    if not bool(torch.isfinite(vertices).all().item()):
        raise ValueError("vertices contain nonfinite positions")
    if bool(((faces < 0) | (faces >= vertices.shape[0])).any().item()):
        raise ValueError("faces contain out-of-range vertex indices")


class WarpUnsignedDistanceBVH:
    """Immutable CUDA mesh snapshot queried on the current Torch stream.

    max_distance is explicit: queries without a face strictly inside this radius
    raise instead of substituting a truncated UDF. Retain this object until its
    current CUDA stream is synchronized. Mesh tensors must not be mutated.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        *,
        max_distance: float,
        query_batch_size: int = 262144,
    ):
        validate_mesh(vertices, faces)
        if not math.isfinite(max_distance) or not 0 < max_distance < 1e18:
            raise ValueError("max_distance must be positive, finite and < 1e18")
        if wp.__version__ != "1.13.0":
            raise RuntimeError(
                f"pilot requires source-checked warp-lang 1.13.0, found {wp.__version__}"
            )
        if not isinstance(query_batch_size, int) or not 0 < query_batch_size < 2**31:
            raise ValueError("query_batch_size must be a positive signed-int count")
        self.query_batch_size = query_batch_size
        self.query_sizes = []
        wp.init()
        self.vertices = vertices
        self.faces = faces
        self.device = vertices.device
        self.max_distance = float(max_distance)
        with torch.cuda.device(self.device):
            stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
            with wp.ScopedStream(stream):
                self.mesh = wp.Mesh(
                    points=wp.from_torch(vertices, dtype=wp.vec3, requires_grad=False),
                    indices=wp.from_torch(
                        faces.view(-1), dtype=wp.int32, requires_grad=False
                    ),
                    support_winding_number=False,
                    bvh_constructor="lbvh",
                )

    def unsigned_distance(self, positions: torch.Tensor, return_uvw: bool = False):
        if (
            positions.ndim < 2
            or positions.shape[-1] != 3
            or positions.dtype != torch.float32
        ):
            raise ValueError("positions must have shape (..., 3) and dtype float32")
        if positions.device != self.device:
            raise ValueError("query positions must be on the mesh CUDA device")
        prefix = positions.shape[:-1]
        with torch.cuda.device(self.device):
            positions = positions.contiguous().view(-1, 3)
            n = positions.shape[0]
            self.query_sizes.append(n)
            distances = torch.empty(n, dtype=torch.float32, device=self.device)
            face_ids = torch.empty(n, dtype=torch.int32, device=self.device)
            # One unused element supplies a valid typed array when uvw is off.
            uvw = torch.empty(
                (n if return_uvw else 1, 3), dtype=torch.float32, device=self.device
            )
            if n:
                stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
                for start in range(0, n, self.query_batch_size):
                    stop = min(start + self.query_batch_size, n)
                    wp.launch(
                        kernel=_unsigned_distance,
                        dim=stop - start,
                        inputs=[
                            self.mesh.id,
                            wp.from_torch(
                                positions[start:stop],
                                dtype=wp.vec3,
                                requires_grad=False,
                            ),
                            self.max_distance,
                            wp.from_torch(distances[start:stop], requires_grad=False),
                            wp.from_torch(face_ids[start:stop], requires_grad=False),
                            wp.from_torch(
                                uvw[start:stop] if return_uvw else uvw,
                                dtype=wp.vec3,
                                requires_grad=False,
                            ),
                            return_uvw,
                        ],
                        stream=stream,
                    )
                # Prototype correctness check; includes a deliberate scalar sync.
                if bool((face_ids < 0).any().item()):
                    raise RuntimeError(
                        "Warp closest-point query missed a face inside max_distance"
                    )
            return (
                distances.view(prefix),
                face_ids.to(torch.int64).view(prefix),
                uvw.view(*prefix, 3) if return_uvw else None,
            )
