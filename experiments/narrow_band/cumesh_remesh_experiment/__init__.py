"""Experiment-only narrow-band remesh, isolated from production cumesh."""

import math

import torch

from .remeshing import _remesh_narrow_band_dc
from .warp_bvh import WarpUnsignedDistanceBVH, validate_mesh

__all__ = ["WarpUnsignedDistanceBVH", "remesh_narrow_band_dc"]


@torch.no_grad()
def remesh_narrow_band_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    resolution: int,
    band: float = 1,
    project_back: float = 0,
    verbose: bool = False,
    bvh=None,
    diagnostics: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the zero set of UDF - band*scale/resolution using injected BVH.

    Run on the current Torch CUDA stream. Input buffers must be ready on that
    stream (call wait_stream if another stream produced them). The experiment
    supports resolution powers of two >=32 and project_back=0 only.
    """
    validate_mesh(vertices, faces)
    if (
        center.shape != (3,)
        or center.dtype != torch.float32
        or center.device != vertices.device
    ):
        raise ValueError("center must be a float32 (3,) tensor on the mesh CUDA device")
    if not bool(torch.isfinite(center).all().item()):
        raise ValueError("center must be finite")
    if (
        not isinstance(resolution, int)
        or resolution < 32
        or resolution & (resolution - 1)
    ):
        raise ValueError("resolution must be a power of two >=32")
    if (resolution + 1) ** 3 >= 2**32:
        raise ValueError("pilot supports uint32 spatial keys only")
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(band) or band <= 0:
        raise ValueError("scale and band must be finite and positive")
    if project_back != 0:
        raise ValueError("the approved Make-solid pilot supports project_back=0 only")
    if bvh is None:
        raise ValueError(
            "pass an explicit WarpUnsignedDistanceBVH (or compatible UDF query object)"
        )
    if bvh.device != vertices.device:
        raise ValueError("BVH and mesh CUDA devices must match")
    with torch.cuda.device(vertices.device):
        return _remesh_narrow_band_dc(
            vertices,
            faces,
            center,
            scale,
            resolution,
            band,
            project_back,
            verbose,
            bvh,
            diagnostics,
        )
