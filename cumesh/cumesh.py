from dataclasses import dataclass
from typing import Protocol

import math
import time
import torch
from tqdm import tqdm
from .xatlas import Atlas
from . import _C

SIMPLIFICATION_POLICY_VERSION = "cumesh.cumulative-cover-error-first.v4"


class SimplificationFailure(RuntimeError):
    """Target/deadline not reached; the current mesh remains owned and readable."""

    def __init__(self, code: str, target_faces: int | None, remaining_faces: int):
        self.code = code
        self.target_faces = target_faces
        self.remaining_faces = remaining_faces
        super().__init__(f"{code}: target={target_faces}, remaining={remaining_faces}")


class _NativeReleaseCompletion(Protocol):
    device_index: int
    stream_id: int
    owner_id: int
    generation: int
    is_complete: bool
    status: str

    def wait_on_current_stream(self) -> int: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class CuMeshReleaseCompletion:
    _token: _NativeReleaseCompletion

    @property
    def device_index(self) -> int:
        return int(self._token.device_index)

    @property
    def stream_id(self) -> int:
        return int(self._token.stream_id)

    @property
    def owner_id(self) -> int:
        return int(self._token.owner_id)

    @property
    def generation(self) -> int:
        return int(self._token.generation)

    @property
    def is_complete(self) -> bool:
        return bool(self._token.is_complete)

    @property
    def status(self) -> str:
        return str(self._token.status)

    def wait_on_current_stream(self) -> int:
        return int(self._token.wait_on_current_stream())

    def synchronize(self) -> None:
        self._token.synchronize()


@dataclass(frozen=True)
class CuMeshReleaseResult:
    status: str
    buffers_released: int
    device_index: int
    stream_id: int
    stream_synchronize_count: int
    completion_token: CuMeshReleaseCompletion | None
    quarantined_buffer_count: int
    error_code: int
    error_message: str
    owner_id: int
    generation: int
    cleanup_error_code: int
    cleanup_error_message: str
    primary_error_code: int
    primary_error_message: str

    @property
    def completion_status(self) -> str:
        if self.completion_token is None:
            return self.status
        return self.completion_token.status


class CuMesh:
    def __init__(self):
        self.cu_mesh = _C.CuMesh()

    def init(self, vertices: torch.Tensor, faces: torch.Tensor):
        """
        Initialize the CuMesh with vertices and faces.

        Args:
            vertices: a tensor of shape [V, 3] containing the vertex positions.
            faces: a tensor of shape [F, 3] containing the face indices.
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Input vertices must be of shape [V, 3]"
        assert faces.ndim == 2 and faces.shape[1] == 3, "Input faces must be of shape [F, 3]"
        assert vertices.is_contiguous() and faces.is_contiguous(), "Input tensors must be contiguous"
        assert vertices.is_cuda and faces.is_cuda and vertices.device == faces.device, "Input tensors must both be on the same CUDA device"
        self.cu_mesh.init(vertices, faces)
        
    @property
    def num_vertices(self) -> int:
        return self.cu_mesh.num_vertices()
    
    @property
    def num_faces(self) -> int:
        return self.cu_mesh.num_faces()
    
    @property
    def num_edges(self) -> int:
        return self.cu_mesh.num_edges()
    
    @property
    def num_boundaries(self) -> int:
        return self.cu_mesh.num_boundaries()
    
    @property
    def num_conneted_components(self) -> int:
        return self.cu_mesh.num_conneted_components()
    
    @property
    def num_boundary_conneted_components(self) -> int:
        return self.cu_mesh.num_boundary_conneted_components()
    
    @property
    def num_boundary_loops(self) -> int:
        return self.cu_mesh.num_boundary_loops()

    def clear_cache(self):
        """
        Clear the cached data.
        """
        self.cu_mesh.clear_cache()

    def swap(self, other: "CuMesh") -> None:
        if not isinstance(other, CuMesh):
            raise TypeError("other must be a CuMesh")
        self.cu_mesh.swap(other.cu_mesh)

    def release_async(self, synchronize: bool = False) -> CuMeshReleaseResult:
        """Release all owned CUDA storage in canonical stream order."""
        result = self.cu_mesh.release_async(synchronize)
        native_completion = result.completion_token
        return CuMeshReleaseResult(
            status=str(result.status),
            buffers_released=int(result.buffers_released),
            device_index=int(result.device_index),
            stream_id=int(result.stream_id),
            stream_synchronize_count=int(result.stream_synchronize_count),
            completion_token=(
                None if native_completion is None else CuMeshReleaseCompletion(native_completion)
            ),
            quarantined_buffer_count=int(result.quarantined_buffer_count),
            error_code=int(result.error_code),
            error_message=str(result.error_message),
            owner_id=int(result.owner_id),
            generation=int(result.generation),
            cleanup_error_code=int(result.cleanup_error_code),
            cleanup_error_message=str(result.cleanup_error_message),
            primary_error_code=int(result.primary_error_code),
            primary_error_message=str(result.primary_error_message),
        )

    @property
    def allocated_buffer_count(self) -> int:
        return int(self.cu_mesh.allocated_buffer_count())

    @property
    def owner_id(self) -> int:
        return int(self.cu_mesh.owner_id)

    @property
    def generation(self) -> int:
        return int(self.cu_mesh.generation)

    @property
    def device_index(self) -> int:
        return int(self.cu_mesh.device_index)

    @staticmethod
    def destructor_release_error_count() -> int:
        return int(_C.CuMesh.destructor_release_error_count())

    @staticmethod
    def destructor_device_synchronize_count() -> int:
        return int(_C.CuMesh.destructor_device_synchronize_count())

    @staticmethod
    def quarantined_cuda_allocation_count() -> int:
        return int(_C.CuMesh.quarantined_cuda_allocation_count())

    @staticmethod
    def retry_quarantined_cuda_allocations() -> None:
        _C.CuMesh.retry_quarantined_cuda_allocations()

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read the current vertices and faces from the CuMesh.

        Returns:
            A tuple of two tensors: the vertex positions and the face indices.
        """
        return self.cu_mesh.read()
    
    def read_face_normals(self) -> torch.Tensor:
        """
        Read the normals of the faces from the CuMesh.

        Returns:
            The face normals as an [F, 3] tensor.
        """
        return self.cu_mesh.read_face_normals()
    
    def read_vertex_normals(self) -> torch.Tensor:
        """
        Read the normals of the vertices from the CuMesh.

        Returns:
            The vertex normals as an [V, 3] tensor.
        """
        return self.cu_mesh.read_vertex_normals()
    
    def read_edges(self) -> torch.Tensor:
        """
        Read the edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [E, 2] containing the edge indices.
        """
        return self.cu_mesh.read_edges()
    
    def read_boundaries(self) -> torch.Tensor:
        """
        Read the boundary edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [B] containing the boundary edge indices.
        """
        return self.cu_mesh.read_boundaries()
    
    
    def read_manifold_face_adjacency(self) -> torch.Tensor:
        """
        Read the manifold face adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold face adjacency.
        """
        return self.cu_mesh.read_manifold_face_adjacency()
    
    def read_manifold_boundary_adjacency(self) -> torch.Tensor:
        """
        Read the manifold boundary adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold boundary adjacency.
        """
        return self.cu_mesh.read_manifold_boundary_adjacency()
    
    def read_connected_components(self) -> tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each face.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [F] containing the connected component ID for each face.
        """
        return self.cu_mesh.read_connected_components()
    
    def read_boundary_connected_components(self) -> tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each boundary edge.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [E] containing the connected component ID for each boundary edge.
        """
        return self.cu_mesh.read_boundary_connected_components()
    
    def read_boundary_loops(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Read the boundary loops of the mesh.

        Returns:
            A tuple of three values:
                - the number of boundary loops
                - a tensor of shape [L] containing the indices of the boundary edges in each loop.
                - a tensor of shape [N_loops + 1] containing the offsets of the boundary edges in each loop.
        """
        return self.cu_mesh.read_boundary_loops()
    
    def read_all_cache(self) -> dict[str, torch.Tensor]:
        """
        Read all cached data.

        Returns:
            A dictionary of cached data.
        """
        return self.cu_mesh.read_all_cache()
    
    def compute_face_normals(self):
        """
        Compute the normals of the faces.
        """
        self.cu_mesh.compute_face_normals()
    
    def compute_vertex_normals(self):
        """
        Compute the normals of the vertices.
        """
        self.cu_mesh.compute_vertex_normals()
        
    def get_vertex_face_adjacency(self):
        """
        Compute the vertex to face adjacency.
        """
        self.cu_mesh.get_vertex_face_adjacency()
        
    def get_edges(self):
        """
        Compute the edges of the mesh.
        """
        self.cu_mesh.get_edges()
        
    def get_edge_face_adjacency(self):
        """
        Compute the edge to face adjacency.
        """
        self.cu_mesh.get_edge_face_adjacency()
        
    def get_vertex_edge_adjacency(self):
        """
        Compute the vertex to edge adjacency.
        """
        self.cu_mesh.get_vertex_edge_adjacency()
        
    def get_boundary_info(self):
        """
        Compute the boundary information of the mesh.
        """
        self.cu_mesh.get_boundary_info()
        
    def get_vertex_boundary_adjacency(self):
        """
        Compute the vertex to boundary adjacency.
        """
        self.cu_mesh.get_vertex_boundary_adjacency()
        
    def get_manifold_face_adjacency(self):
        """
        Compute the manifold face adjacency.
        """
        self.cu_mesh.get_manifold_face_adjacency()
        
    def get_manifold_boundary_adjacency(self):
        """
        Compute the manifold boundary adjacency.
        """
        self.cu_mesh.get_manifold_boundary_adjacency()
        
    def get_connected_components(self):
        """
        Compute the connected components of the mesh.
        """
        self.cu_mesh.get_connected_components()
        
    def get_boundary_connected_components(self):
        """
        Compute the connected components of the boundary of the mesh.
        """
        self.cu_mesh.get_boundary_connected_components()
        
    def get_boundary_loops(self):
        """
        Compute the boundary loops of the mesh.
        """
        self.cu_mesh.get_boundary_loops()
        
    def remove_faces(self, face_mask: torch.Tensor):
        """
        Remove faces from the mesh.

        Args:
            face_mask: a boolean tensor of shape [F] indicating which faces to remove.
        """
        assert face_mask.ndim == 1 and face_mask.shape[0] == self.num_faces, "face_mask must be a boolean tensor of shape [F]"
        assert face_mask.is_contiguous() and face_mask.is_cuda, "face_mask must be a CUDA tensor"
        assert face_mask.dtype == torch.bool, "face_mask must be a boolean tensor"
        self.cu_mesh.remove_faces(face_mask)
    
    def remove_unreferenced_vertices(self):
        """
        Remove unreferenced vertices from the mesh.
        """
        self.cu_mesh.remove_unreferenced_vertices()
        
    def remove_duplicate_faces(self):
        """
        Remove duplicate faces from the mesh.
        """
        self.cu_mesh.remove_duplicate_faces()
        
    def remove_degenerate_faces(self, abs_thresh: float=1e-24, rel_thresh: float=1e-12):
        """
        Remove degenerate faces from the mesh.

        Args:
            abs_thresh: absolute area threshold below which a face is considered degenerate.
            rel_thresh: relative area to square of the longest edge threshold below which a face is considered degenerate.
                Note that a face is considered degenerate if both the absolute and relative conditions are met.
        """
        self.cu_mesh.remove_degenerate_faces(abs_thresh, rel_thresh)
        
    def fill_holes(self, max_hole_perimeter: float=3e-2):
        """
        Fill holes in the mesh.

        Args:
            max_hole_perimeter: the maximum perimeter of a hole to fill.
        """
        self.cu_mesh.fill_holes(max_hole_perimeter)
        
    def repair_non_manifold_edges(self):
        """
        Repair Non-manifold edges by splitting vertices.
        This creates duplicate vertices with the same coordinates.
        """
        self.cu_mesh.repair_non_manifold_edges()

    def remove_non_manifold_faces(self):
        """
        Remove faces on non-manifold edges.
        For each non-manifold edge (shared by >2 faces), only keep the first 2 faces.
        This repairs non-manifold edges by deleting faces instead of splitting vertices.
        """
        self.cu_mesh.remove_non_manifold_faces()
        
    def remove_small_connected_components(self, min_area: float):
        """
        Repair Non-manifold edges by splitting edges
        
        Args:
            min_area: the minimum area of a connected component to keep.
        """
        self.cu_mesh.remove_small_connected_components(min_area)
        
    def unify_face_orientations(self):
        """
        Unify the orientations of the faces.
        """
        self.cu_mesh.unify_face_orientations()
    
    def simplify(self, target_num_faces: int | None = None, verbose: bool=False, options: dict={}):
        """Accumulate original planes with optional error-first admission.

        max_error bounds the cumulative original-vertex cluster cover radius.
        Weighted RMS original-plane distance ranks proposals; callers still
        qualify the final actual surface samples.
        Target count is an extra stop. At an error/topology floor, the readable
        result may exceed the target; last_simplification reports that state.
        """
        error_limit=options.get('max_error')
        if target_num_faces is not None and (not isinstance(target_num_faces,int) or target_num_faces<=0):
            raise ValueError("target_num_faces must be a positive integer or None")
        if target_num_faces is None and error_limit is None:
            raise ValueError("max_error or target_num_faces is required")
        maximum=torch.finfo(torch.float32).max
        if error_limit is not None:
            if not math.isfinite(error_limit) or error_limit<=0 or error_limit>math.sqrt(maximum):
                raise ValueError("max_error must be positive finite with a float32-representable square")
            max_error_squared=error_limit**2
            threshold=max_error_squared
        else:
            max_error_squared=math.inf
            threshold=options.get('thresh',1e-8)
            if not math.isfinite(threshold) or not 0<=threshold<=maximum:
                raise ValueError("thresh must be finite, nonnegative and representable as float32")
        for name,default in [('lambda_edge_length',1e-2),('lambda_skinny',1e-3)]:
            value=options.get(name,default)
            if not math.isfinite(value) or not 0<=value<=maximum:
                raise ValueError(f'{name} must be finite, nonnegative and representable as float32')
        deadline=options.get('deadline_monotonic')
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline_monotonic must be finite")
        faces=self.cu_mesh.num_faces()
        self.last_simplification={"target_faces":target_num_faces,"max_error":error_limit,"metric":"outward_rounded_original_vertex_cluster_cover","ranking_metric":"weighted_RMS_original_face_planes","steps":[],"status":"running"}
        if target_num_faces is not None and faces<=target_num_faces:
            self.last_simplification.update(status="target_reached",remaining_faces=faces)
            return
        quadrics=None
        pbar=tqdm(total=faces,desc="Simplifying",disable=not verbose)
        try:
            quadrics=self.cu_mesh.initial_simplification_quadrics()
            while True:
                if deadline is not None and time.monotonic()>=deadline:
                    raise SimplificationFailure("simplify_deadline",target_num_faces,faces)
                started=time.monotonic()
                _,next_faces,next_quadrics=self.cu_mesh.simplify_step_accumulated(
                    quadrics,options.get('lambda_edge_length',1e-2),options.get('lambda_skinny',1e-3),
                    threshold,max_error_squared,options.get('timing',False))
                quadrics=next_quadrics
                self.last_simplification['steps'].append({"faces_before":faces,"faces_after":next_faces,"threshold":threshold,"seconds":time.monotonic()-started})
                pbar.update(faces-next_faces)
                if target_num_faces is not None and next_faces<=target_num_faces:
                    self.last_simplification.update(status="target_reached",remaining_faces=next_faces)
                    break
                deleted=faces-next_faces
                if deleted==0 and error_limit is not None:
                    self.last_simplification.update(status="error_or_topology_limit_reached",remaining_faces=next_faces)
                    break
                if deleted==0 and threshold==maximum:
                    raise SimplificationFailure("simplify_no_legal_progress",target_num_faces,next_faces)
                if error_limit is None and deleted/faces<1e-2:
                    threshold=min(maximum,max(threshold*10,torch.finfo(torch.float32).tiny))
                faces=next_faces
            self.last_simplification["max_cluster_radius"]=float(quadrics[:,11].max().item())
        except SimplificationFailure as error:
            self.last_simplification.update(status=error.code,remaining_faces=error.remaining_faces)
            raise
        except Exception:
            self.last_simplification['status']='failed'
            try: self.last_simplification['remaining_faces']=self.cu_mesh.num_faces()
            except Exception: self.last_simplification['remaining_faces']=None
            raise
        finally:
            del quadrics
            pbar.close()

    def compute_charts(
        self,
        threshold_cone_half_angle_rad: float=math.radians(90),
        refine_iterations: int=100,
        global_iterations: int=3,
        smooth_strength: float=1,
        area_penalty_weight: float=0.1,
        perimeter_area_ratio_weight: float=0.0001,
    ):
        """
        Compute the atlas charts.

        Args:
            threshold_cone_half_angle_rad: The threshold for the cone half angle in radians.
            refine_iterations: The number of refinement iterations.
            smooth_strength: The strength of chart boundary smoothing.
            area_penalty_weight: Coefficient for chart size penalty. Cost += Area * weight.
                                 Prevents charts from becoming too large if > 0, 
                                 or encourages larger charts if < 0 (though usually used to penalize size variance).
            perimeter_area_ratio_weight: Coefficient for shape irregularity (long-strip) penalty. 
                                         Cost += (Perimeter / Area) * weight.
                                         Higher values penalize long strips and encourage circular/compact shapes.
        """
        self.cu_mesh.compute_charts(
            threshold_cone_half_angle_rad,
            refine_iterations,
            global_iterations,
            smooth_strength,
            area_penalty_weight,
            perimeter_area_ratio_weight
        )
        
    def read_atlas_charts(self) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Read the atlas chart IDs for each face.
        
        Returns:
            A tuple of two values:
                - the number of charts
                - a tensor of shape [F] containing the chart ID for each face.
                - a tensor of shape [V] containing the vertex map
                - a tensor of shape [F, 3] containing the chart faces
                - a tensor of shape [C+1] containing the offsets of the chart vertices in the vertices tensor.
                - a tensor of shape [C+1] containing the offsets of the chart faces in the faces tensor.
        """
        return self.cu_mesh.read_atlas_charts()
    
    def uv_unwrap(
        self,
        compute_charts_kwargs: dict={},
        xatlas_compute_charts_kwargs: dict={},
        xatlas_pack_charts_kwargs: dict={},
        return_vmaps: bool=False,
        verbose: bool=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Parameterize the mesh using the accelerated mesh clustering and Xatlas
        
        Args:
            compute_charts_kwargs: a dictionary of options for the compute_charts function.
            xatlas_compute_charts_kwargs: a dictionary of options for the xatlas compute_charts function.
            xatlas_pack_charts_kwargs: a dictionary of options for the xatlas pack_charts function.
            return_vmaps: whether to return the vertex maps.
            verbose: whether to print the progress.
            
        Returns:
            A tuple of:
                - the vertex positions
                - the face indices
                - the uv coordinates
                - (optional) the map from the new vertex indices to the old vertex indices
        """
        xatlas_compute_charts_kwargs['verbose'] = verbose
        xatlas_pack_charts_kwargs['verbose'] = verbose
        
        self.remove_degenerate_faces()
        
        # 1. Fast mesh clustering
        self.compute_charts(**compute_charts_kwargs)
        new_vertices, new_faces = self.read()
        num_charts, charts_id, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = self.read_atlas_charts()
        chart_vertices = new_vertices[chart_vmap].cpu()
        chart_faces = chart_faces.cpu()
        chart_vertex_offset = chart_vertex_offset.cpu()
        chart_face_offset = chart_face_offset.cpu()
        chart_vmap = chart_vmap.cpu()
        if verbose:
            print(f"Get {num_charts} clusters after fast clustering")
        
        # 2. Xatlas packing
        xatlas = Atlas()
        chart_vmaps = []
        for i in tqdm(range(num_charts), desc="Adding clusters to xatlas", disable=not verbose):
            chart_faces_i = chart_faces[chart_face_offset[i]:chart_face_offset[i+1]] - chart_vertex_offset[i]
            chart_vertices_i = chart_vertices[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
            chart_vmap_i = chart_vmap[chart_vertex_offset[i]:chart_vertex_offset[i+1]]
            chart_vmaps.append(chart_vmap_i)
            xatlas.add_mesh(chart_vertices_i, chart_faces_i)
        xatlas.compute_charts(**xatlas_compute_charts_kwargs)
        xatlas.pack_charts(**xatlas_pack_charts_kwargs)
        vmaps = []
        faces = []
        uvs = []
        cnt = 0
        for i in tqdm(range(num_charts), desc="Gathering results from xatlas", disable=not verbose):
            vmap, x_faces, x_uvs = xatlas.get_mesh(i)
            vmaps.append(chart_vmaps[i][vmap])
            faces.append(x_faces + cnt)
            uvs.append(x_uvs)
            cnt += vmap.shape[0]
        vmaps = torch.cat(vmaps, dim=0)
        vertices = new_vertices.cpu()[vmaps]
        faces = torch.cat(faces, dim=0)
        uvs = torch.cat(uvs, dim=0)
        
        out = [vertices, faces, uvs]
        if return_vmaps:
            out.append(vmaps)
        
        return tuple(out)
