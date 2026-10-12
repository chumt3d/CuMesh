"""Conservative pilot admission; does not reduce resolution or output scope."""

import torch

# Operational headroom for CUDA/Warp/CUB workspace beyond counted tensors.
# This is a conservative prototype admission estimate, not a proven peak bound.
WORKSPACE_RESERVE_BYTES = 256 * 1024**2


def require_memory(
    device: torch.device,
    stage: str,
    additional_bytes: int,
    *,
    active_voxels: int = 0,
    active_vertices: int = 0,
    crossed_edges: int = 0,
) -> None:
    free_driver, _ = torch.cuda.mem_get_info(device)
    needed = additional_bytes + WORKSPACE_RESERVE_BYTES
    # Count driver-free bytes only: upstream raw cudaMalloc cannot consume Torch
    # cached blocks. Conservative rejection is safer than admitting a raw OOM.
    if needed > free_driver:
        raise RuntimeError(
            f"pilot memory admission failed at {stage}: A={active_voxels} "
            f"V={active_vertices} E={crossed_edges} estimated additional={additional_bytes} "
            f"reserve={WORKSPACE_RESERVE_BYTES} driver_free={free_driver}; "
            "no resolution reduction or partial mesh emitted"
        )


def require_contour_memory(device: torch.device, a: int, v: int) -> None:
    # V<=8A. Voxel hash + corner coords/field/query/vertex hash + dual/flags.
    estimate = 16 * a + (12 + 4 + 12 + 16) * v + 24 * a
    require_memory(
        device,
        "sparse corners and contour",
        estimate,
        active_voxels=a,
        active_vertices=v,
    )


def require_topology_memory(device: torch.device, a: int, v: int, e: int) -> None:
    # E<=3A. Count upstream neighbors, connected coords, padded hash keys,
    # lookup+quads, both split candidates, normals, alignment, output triangles,
    # and conservative unique/sort/index workspace. Already-live tensors are
    # counted again deliberately, leaving headroom for eager Torch intermediates.
    contour = 16 * a + (12 + 4 + 12 + 16) * v + 24 * a
    topology = 144 * a + (48 + 64 + 16 + 16 + 96 + 48 + 8 + 24 + 96) * e + 16 * a
    require_memory(
        device,
        "topology generation",
        contour + topology,
        active_voxels=a,
        active_vertices=v,
        crossed_edges=e,
    )
