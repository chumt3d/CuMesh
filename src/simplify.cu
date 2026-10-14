#include "cumesh.h"
#include "dtypes.cuh"
#include <cub/cub.cuh>
#include <c10/cuda/CUDAStream.h>
#include "shared.h"
#include "exact_geometry/admission.h"
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>

// Tiny3D modification: topology-preserving contractions and independent stars.
// MIT (repository LICENSE). Link condition, not copied implementation:
// Dey, Edelsbrunner, Guha, Nekhayev, Topology preserving edge contraction (1999).
// https://research-explorer.ista.ac.at/record/3582


namespace cumesh {
struct SimplificationControl {
    const CuMesh& mesh;
    double max_seconds;
    std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    double remaining()const{return max_seconds-std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();}
    void check()const {
        TORCH_CHECK(!mesh.simplification_cancelled(),"[CuMesh] simplify_spatial_cancelled");
        TORCH_CHECK(remaining()>0.0,"[CuMesh] simplify_spatial_deadline");
    }
};


__device__ inline uint64_t pack_key_value_positive(int key, float value) {
    unsigned int v = __float_as_uint(value);
    return (static_cast<uint64_t>(v) << 32) |
           static_cast<unsigned int>(key);
}


__device__ inline void unpack_key_value_positive(uint64_t key_value, int& key, float& value) {
    key = static_cast<int>(key_value & 0xffffffffu);
    value = __uint_as_float(static_cast<unsigned int>(key_value >> 32));
}


/**
 * Get the QEM for each vertex
 *
 * @param vertices: the vertices of the mesh, shape (V)
 * @param faces: the faces of the mesh, shape (F)
 * @param vert2face: the buffer for neighbor face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighbor face ids offset, shape (V+1)
 * @param V: the number of vertices
 * @param F: the number of faces
 * @param qems: the buffer for QEMs, shape (V)
 */
static __global__ void get_qem_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset,
    const int V,
    const int F,
    QEM* qems
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    // compute QEM
    QEM v_qem;
    for (int f = vert2face_offset[tid]; f < vert2face_offset[tid+1]; f++) {
        int3 f_vids = faces[vert2face[f]];
        float3 a=vertices[f_vids.x],b=vertices[f_vids.y],c=vertices[f_vids.z];
        double ux=double(b.x)-a.x,uy=double(b.y)-a.y,uz=double(b.z)-a.z;
        double vx=double(c.x)-a.x,vy=double(c.y)-a.y,vz=double(c.z)-a.z;
        double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
        double double_area=sqrt(nx*nx+ny*ny+nz*nz);
        if (!(double_area>0.0)||!isfinite(double_area)) {v_qem.weight=NAN;break;}
        nx/=double_area;ny/=double_area;nz/=double_area;
        double d=-(nx*double(a.x)+ny*double(a.y)+nz*double(a.z));
        // meshoptimizer b233043: sqrt(double-area) scales weight linearly.
        v_qem.add_plane(make_double4(nx,ny,nz,d),sqrt(double_area));
    }
    qems[tid] = v_qem;
}


/**
 * Get the QEM for each vertex
 */
void get_qem(
    CuMesh& ctx,
    cudaStream_t stream
) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    ctx.temp_storage.resize(V * sizeof(QEM));
    get_qem_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        V, F,
        reinterpret_cast<QEM*>(ctx.temp_storage.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
}


inline __device__ bool process_incident_tri(
    int tri_idx,
    int collapse_keep_vert, // the vertex we keep (e0 or e1)
    int collapse_other_vert, // the other one (the one removed)
    const float3* vertices,
    const int3* faces,
    const Vec3f& v_new, // actual stored fp32 proposal
    float& skinny_cost,
    int& num_tri
) {
    int3 f_vids = faces[tri_idx];

    // If this triangle contains the other vertex (the edge), it will be removed, skip it
    if (f_vids.x == collapse_other_vert || f_vids.y == collapse_other_vert || f_vids.z == collapse_other_vert)
        return true; // skip, not an error

    // get old positions
    Vec3f a(vertices[f_vids.x]);
    Vec3f b(vertices[f_vids.y]);
    Vec3f c(vertices[f_vids.z]);

    // build new positions: replace occurrences of collapse_keep_vert with v_new
    Vec3f na = (f_vids.x == collapse_keep_vert) ? v_new : a;
    Vec3f nb = (f_vids.y == collapse_keep_vert) ? v_new : b;
    Vec3f nc = (f_vids.z == collapse_keep_vert) ? v_new : c;

    // Check actual stored fp32 positions, promoted to fp64 like the audit.
    double ux=double(b.x)-a.x,uy=double(b.y)-a.y,uz=double(b.z)-a.z;
    double vx=double(c.x)-a.x,vy=double(c.y)-a.y,vz=double(c.z)-a.z;
    double ox=uy*vz-uz*vy,oy=uz*vx-ux*vz,oz=ux*vy-uy*vx;
    double px=double(nb.x)-na.x,py=double(nb.y)-na.y,pz=double(nb.z)-na.z;
    double qx=double(nc.x)-na.x,qy=double(nc.y)-na.y,qz=double(nc.z)-na.z;
    double nx=py*qz-pz*qy,ny=pz*qx-px*qz,nz=px*qy-py*qx;
    double old_area=.5*sqrt(ox*ox+oy*oy+oz*oz),new_area=.5*sqrt(nx*nx+ny*ny+nz*nz);
    if (!(old_area>0.0)||!(new_area>0.0)||!isfinite(old_area)||!isfinite(new_area)||!(ox*nx+oy*ny+oz*nz>0.0)) return false;
    double ex=double(nc.x)-nb.x,ey=double(nc.y)-nb.y,ez=double(nc.z)-nb.z;
    double denom=ex*ex+ey*ey+ez*ez+px*px+py*py+pz*pz+qx*qx+qy*qy+qz*qz;
    double metric=4.0*sqrt(3.0)*new_area/denom;
    float term=float(1.0-fmin(fmax(metric,0.0),1.0));
    skinny_cost += term;
    num_tri += 1;
    return true;
}


// get_edges() supplies sorted unique uint64 keys. Membership costs O(log E),
// avoiding a quadratic intersection of two high-valence vertex stars.
inline __device__ bool has_edge(const uint64_t* edges, int E, int a, int b) {
    uint64_t key = (uint64_t(min(a, b)) << 32) | uint64_t(max(a, b));
    int lo = 0, hi = E;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (edges[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo < E && edges[lo] == key;
}


inline __device__ bool collapse_link_valid(
    int a, int b, const int3* faces, const int* vert2face,
    const int* offsets, const uint64_t* edges, int E,
    const uint8_t* boundary
) {
    if (offsets[b + 1] - offsets[b] < offsets[a + 1] - offsets[a]) {
        int tmp = a; a = b; b = tmp;
    }
    int opposite[2] = {-1, -1};
    int edge_faces = 0;
    for (int i = offsets[a]; i < offsets[a + 1]; ++i) {
        int3 f = faces[vert2face[i]];
        if (f.x != b && f.y != b && f.z != b) continue;
        if (edge_faces == 2) return false; // nonmanifold edge
        int v = (f.x != a && f.x != b) ? f.x :
                ((f.y != a && f.y != b) ? f.y : f.z);
        if (v == a || v == b || (edge_faces && opposite[0] == v)) return false;
        opposite[edge_faces++] = v;
    }
    if (edge_faces != 1 && edge_faces != 2) return false;
    // An interior chord between two boundary vertices merges/pinches boundary.
    if (edge_faces == 2 && boundary[a] && boundary[b]) return false;
    // Do not contract away the last triangle of a disk component.
    if (edge_faces == 1 && offsets[a + 1] - offsets[a] == 1 &&
        offsets[b + 1] - offsets[b] == 1) return false;

    bool opposite_edge_in_a_link = false;
    for (int i = offsets[a]; i < offsets[a + 1]; ++i) {
        int3 f = faces[vert2face[i]];
        int v[3] = {f.x, f.y, f.z};
        for (int k = 0; k < 3; ++k) {
            if (v[k] == a || v[k] == b || v[k] == opposite[0] ||
                v[k] == opposite[1]) continue;
            if (has_edge(edges, E, b, v[k])) return false;
        }
        if (edge_faces == 2 && f.x != b && f.y != b && f.z != b) {
            bool has_first = f.x == opposite[0] || f.y == opposite[0] || f.z == opposite[0];
            bool has_second = f.x == opposite[1] || f.y == opposite[1] || f.z == opposite[1];
            opposite_edge_in_a_link |= has_first && has_second;
        }
    }
    // The vertex-only condition is insufficient for a tetrahedron boundary:
    // a common link edge would create duplicate triangles after contraction.
    if (opposite_edge_in_a_link) {
        for (int i = offsets[b]; i < offsets[b + 1]; ++i) {
            int3 f = faces[vert2face[i]];
            if (f.x == a || f.y == a || f.z == a) continue;
            bool has_first = f.x == opposite[0] || f.y == opposite[0] || f.z == opposite[0];
            bool has_second = f.x == opposite[1] || f.y == opposite[1] || f.z == opposite[1];
            if (has_first && has_second) return false;
        }
    }
    return true;
}


inline __device__ int face_corner(int3 face, int vertex) {
    return face.x == vertex ? 0 : (face.y == vertex ? 1 : (face.z == vertex ? 2 : -1));
}


inline __device__ int corner_value(int3 values, int corner) {
    return corner == 0 ? values.x : (corner == 1 ? values.y : values.z);
}


// Each oriented manifold vertex link is one path (boundary) or one cycle.
// Walk its incident faces through precomputed edge adjacency, at most degree
// steps. Total walk work is O(F); no per-vertex visited set/parent-chain walk.
static __global__ void get_collapse_vertex_fans_kernel(
    const int3* faces, const int* vert2face, const int* offsets,
    const int3* face2edge, const int* edge2face, const int* edge_offsets,
    const int* edge_counts, int V, uint8_t* manifold
) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= V) return;
    manifold[vertex] = 0;
    int degree = offsets[vertex + 1] - offsets[vertex];
    if (degree == 0) return;
    int start = vert2face[offsets[vertex]], boundaries = 0, incoming_boundary = -1;
    for (int i = offsets[vertex]; i < offsets[vertex + 1]; ++i) {
        int fid = vert2face[i];
        int3 face = faces[fid];
        if (face.x == face.y || face.x == face.z || face.y == face.z) return;
        int corner = face_corner(face, vertex);
        if (corner < 0) return;
        int3 eids = face2edge[fid];
        int outgoing = edge_counts[corner_value(eids, corner)];
        int incoming = edge_counts[corner_value(eids, (corner + 2) % 3)];
        if ((outgoing != 1 && outgoing != 2) || (incoming != 1 && incoming != 2)) return;
        boundaries += (outgoing == 1) + (incoming == 1);
        if (incoming == 1) incoming_boundary = fid;
    }
    if (boundaries != 0 && boundaries != 2) return;
    if (boundaries == 2) {
        if (incoming_boundary < 0) return;
        start = incoming_boundary;
    }
    int current = start;
    for (int visited = 1; visited <= degree; ++visited) {
        int3 face = faces[current];
        int corner = face_corner(face, vertex);
        if (corner < 0) return;
        int eid = corner_value(face2edge[current], corner);
        if (edge_counts[eid] == 1) {
            manifold[vertex] = boundaries == 2 && visited == degree;
            return;
        }
        int offset = edge_offsets[eid];
        int first = edge2face[offset], second = edge2face[offset + 1];
        int next = first == current ? second : (second == current ? first : -1);
        if (next < 0 || next == current) return;
        int next_corner = face_corner(faces[next], vertex);
        if (next_corner < 0 ||
            corner_value(faces[next], (next_corner + 2) % 3) != corner_value(face, (corner + 1) % 3)) return;
        if (next == start) {
            manifold[vertex] = boundaries == 0 && visited == degree;
            return;
        }
        current = next;
    }
}


void get_collapse_vertex_fans(CuMesh& ctx, cudaStream_t stream) {
    ctx.get_edge_face_adjacency();
    int V = ctx.vertices.size;
    ctx.vert_is_manifold.resize(V);
    get_collapse_vertex_fans_kernel<<<(V + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.faces.ptr, ctx.vert2face.ptr, ctx.vert2face_offset.ptr,
        ctx.face2edge.ptr, ctx.edge2face.ptr, ctx.edge2face_offset.ptr,
        ctx.edge2face_cnt.ptr, V, ctx.vert_is_manifold.ptr);
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Get the cost for each edge collapse
 *
 * @param vertices: the vertices of the mesh, shape (V)
 * @param faces: the faces of the mesh, shape (F)
 * @param vert2face: the buffer for neighbor face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighbor face ids offset, shape (V+1)
 * @param edges: the buffer for edges, shape (E)
 * @param vert_is_boundary: the buffer for boundary vertex indicator, shape (V)
 * @param qems: the buffer for QEMs, shape (V)
 * @param V: the number of vertices
 * @param F: the number of faces
 * @param E: the number of edges
 * @param edge_collapse_costs: the buffer for edge collapse costs, shape (E)
 */
// Outward-rounded cluster cover bounds actual original-vertex displacement.
// A legal contraction maps every original simplex to a surviving simplex.
// Barycentric interpolation then bounds both surface directions by this radius.
__device__ inline double displacement_upper(float3 a,float3 b) {
    double dx=__dsub_ru(fmax(double(a.x),double(b.x)),fmin(double(a.x),double(b.x)));
    double dy=__dsub_ru(fmax(double(a.y),double(b.y)),fmin(double(a.y),double(b.y)));
    double dz=__dsub_ru(fmax(double(a.z),double(b.z)),fmin(double(a.z),double(b.z)));
    return __dsqrt_ru(__dadd_ru(__dmul_ru(dx,dx),__dadd_ru(__dmul_ru(dy,dy),__dmul_ru(dz,dz))));
}

__device__ inline double cluster_cover_upper(const QEM& a,const QEM& b,float3 pa,float3 pb,float3 proposed) {
    return fmax(__dadd_ru(a.radius,displacement_upper(pa,proposed)),
                __dadd_ru(b.radius,displacement_upper(pb,proposed)));
}


static __global__ void get_edge_collapse_cost_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset,
    const uint64_t* edges,
    const uint8_t * vert_is_boundary,
    const uint8_t * vert_is_manifold,
    const QEM* qems,
    const int V,
    const int F,
    const int E,
    const float lambda_edge_length,
    const float lambda_skinny,
    float* edge_collapse_costs,
    float3* proposed_positions, double max_error_squared
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    if (!vert_is_manifold[e0] || !vert_is_manifold[e1] ||
        !collapse_link_valid(e0, e1, faces, vert2face, vert2face_offset,
                             edges, E, vert_is_boundary)) {
        edge_collapse_costs[tid] = INFINITY;
        return;
    }

    float3 p0=vertices[e0],p1=vertices[e1];
    QEM q=qems[e0]+qems[e1];
    if (!(q.weight>0.0)||!isfinite(q.weight)||!(qems[e0].radius>=0.0)||!(qems[e1].radius>=0.0)) {edge_collapse_costs[tid]=INFINITY;return;}
    float3 candidates[5];int count_candidates=0;
    bool b0=vert_is_boundary[e0],b1=vert_is_boundary[e1];
    if (b0!=b1) candidates[count_candidates++]=b0?p0:p1;
    else {
        float3 optimal;double error;
        // Boundary edges stay on their segment; interior edges may use optimum.
        if (!b0 && q.solve_optimal(optimal,error)) candidates[count_candidates++]=optimal;
        candidates[count_candidates++]=q.solve_segment(p0,p1);
        candidates[count_candidates++]=p0;candidates[count_candidates++]=p1;
        candidates[count_candidates++]=make_float3(.5f*p0.x+.5f*p1.x,.5f*p0.y+.5f*p1.y,.5f*p0.z+.5f*p1.z);
    }
    double dx=double(p1.x)-p0.x,dy=double(p1.y)-p0.y,dz=double(p1.z)-p0.z;
    double length2=dx*dx+dy*dy+dz*dz,best_error=INFINITY,best_cost=INFINITY;
    float3 best=make_float3(0,0,0);
    for (int k=0;k<count_candidates;++k) {
        Vec3f v(candidates[k]);
        if (!isfinite(v.x)||!isfinite(v.y)||!isfinite(v.z)) continue;
        double error=q.normalized_error(v);
        if (!isfinite(error)||error>max_error_squared) continue;
        double radius=cluster_cover_upper(qems[e0],qems[e1],p0,p1,candidates[k]);
        if (!isfinite(radius)||__dmul_ru(radius,radius)>max_error_squared) continue;
        float skinny=0;int count=0;bool valid=true;
        for (int f=vert2face_offset[e0];f<vert2face_offset[e0+1];++f)
            if (!process_incident_tri(vert2face[f],e0,e1,vertices,faces,v,skinny,count)) {valid=false;break;}
        if (!valid) continue;
        for (int f=vert2face_offset[e1];f<vert2face_offset[e1+1];++f)
            if (!process_incident_tri(vert2face[f],e1,e0,vertices,faces,v,skinny,count)) {valid=false;break;}
        if (!valid) continue;
        // Error-limited callers rank and admit by the original-plane metric.
        // Legacy count-only calls may still use their requested regularizers.
        double cost=error;
        if (!isfinite(max_error_squared)) {
            cost+=lambda_edge_length*length2;
            if (count) cost+=lambda_skinny*(skinny/count)*length2;
        }
        if (isfinite(cost)&&(cost<best_cost||(cost==best_cost&&error<best_error))) {best_error=error;best_cost=cost;best=candidates[k];}
    }
    edge_collapse_costs[tid]=isfinite(best_cost)?float(best_cost):INFINITY;
    proposed_positions[tid]=best;

}


/**
 * Get the cost for each edge collapse
 */
void get_edge_collapse_cost(
    CuMesh& ctx,
    float lambda_edge_length,
    float lambda_skinny,
    cudaStream_t stream,
    const QEM* quadrics,float3* proposed_positions,double max_error_squared
) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;
    ctx.edge_collapse_costs.resize(E);
    get_edge_collapse_cost_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edges.ptr,
        ctx.vert_is_boundary.ptr,
        ctx.vert_is_manifold.ptr,
        quadrics,
        V, F, E,
        lambda_edge_length, lambda_skinny,
        ctx.edge_collapse_costs.ptr,proposed_positions,max_error_squared
    );
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Propagate cost to neighboring faces
 *
 * @param edges: the buffer for edges, shape (E)
 * @param vert2face: the buffer for neighboring face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighboring face ids offset, shape (V+1)
 * @param edge_collapse_costs: the buffer for edge collapse costs, shape (E)
 * @param V: the number of vertices
 * @param F: the number of faces
 * @param E: the number of edges
 * @param propagated_costs: the buffer for one-ring vertex costs, shape (V)
 */
static __global__ void endpoint_min_cost_kernel(
    const uint64_t* edges,
    const float* edge_collapse_costs,
    const uint8_t* active_edges,
    const int E,
    uint64_t* endpoint_min_costs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    if (!isfinite(edge_collapse_costs[tid]) || (active_edges && !active_edges[tid])) return;
    uint64_t cost = pack_key_value_positive(tid, edge_collapse_costs[tid]);

    atomicMin(reinterpret_cast<unsigned long long*>(&endpoint_min_costs[e0]), static_cast<unsigned long long>(cost));
    atomicMin(reinterpret_cast<unsigned long long*>(&endpoint_min_costs[e1]), static_cast<unsigned long long>(cost));
}


static __global__ void gather_star_min_cost_kernel(
    const int3* faces, const int* vert2face, const int* vert2face_offset,
    const uint64_t* endpoint_min_costs, int V, uint64_t* propagated_costs
) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= V) return;
    uint64_t minimum = std::numeric_limits<uint64_t>::max();
    for (int f = vert2face_offset[vertex]; f < vert2face_offset[vertex+1]; ++f) {
        int3 ids = faces[vert2face[f]];
        uint64_t a = endpoint_min_costs[ids.x];
        uint64_t b = endpoint_min_costs[ids.y];
        uint64_t c = endpoint_min_costs[ids.z];
        if (a < minimum) minimum = a;
        if (b < minimum) minimum = b;
        if (c < minimum) minimum = c;
    }
    // Exactly the same minimum over all edges whose endpoint stars contain
    // this vertex. Integer minima are order independent; isolated vertices
    // retain MAX just as in the edge-scatter implementation.
    propagated_costs[vertex] = minimum;
}


/**
 * Propagate cost to neighboring faces
 */
void propagate_cost(
    CuMesh& ctx,
    cudaStream_t stream,
    const uint8_t* active_edges = nullptr
) {
    size_t V = ctx.vertices.size;
    size_t E = ctx.edges.size;
    ctx.propagated_costs.resize(V);
    // QEMs in temp_storage are dead after the cost kernel on this same stream.
    // Reuse 8V bytes of its 96V-byte workspace; no persistent allocation.
    ctx.temp_storage.resize(V * sizeof(uint64_t));
    auto* endpoint_min_costs = reinterpret_cast<uint64_t*>(ctx.temp_storage.ptr);
    fill_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        endpoint_min_costs,
        V,
        std::numeric_limits<uint64_t>::max()
    );
    CUDA_CHECK(cudaGetLastError());
    endpoint_min_cost_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.edges.ptr,
        ctx.edge_collapse_costs.ptr,
        active_edges,
        E, endpoint_min_costs
    );
    CUDA_CHECK(cudaGetLastError());
    gather_star_min_cost_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.faces.ptr, ctx.vert2face.ptr, ctx.vert2face_offset.ptr,
        endpoint_min_costs, V,
        ctx.propagated_costs.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Collapse edges parallelly
 *
 * @param vertices: the vertices of the mesh, shape (V)
 * @param faces: the faces of the mesh, shape (F)
 * @param edges: the buffer for edges, shape (E)
 * @param vert2face: the buffer for neighboring face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighboring face ids offset, shape (V+1)
 * @param edge_collapse_costs: the buffer for edge collapse costs, shape (E)
 * @param propagated_costs: the buffer for one-ring vertex costs, shape (V)
 * @param vert_is_boundary: the buffer for boundary vertex indicator, shape (V)
 * @param V: the number of vertices
 * @param F: the number of faces
 * @param E: the number of edges
 * @param collapse_thresh: the threshold for cost collapse
 * @param vertices_kept: the flag for vertices kept, shape (V)
 * @param faces_kept: the flag for faces kept, shape (F)
 */
static __global__ void select_collapse_edges_kernel(
    const int3* faces,
    const uint64_t* edges,
    const int* vert2face,
    const int* vert2face_offset,
    const float* edge_collapse_costs,
    const uint64_t* propagated_costs,
    uint8_t* active_edges,
    uint8_t* chosen_edges,
    uint8_t* reserved_vertices,
    int* progress,
    const int E
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    float cost = edge_collapse_costs[tid];
    if (!active_edges[tid]) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    uint64_t pack = pack_key_value_positive(tid, cost);

    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int3 ids = faces[vert2face[f]];
        if (propagated_costs[ids.x] != pack || propagated_costs[ids.y] != pack ||
            propagated_costs[ids.z] != pack) {
            return;
        }
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        int3 ids = faces[vert2face[f]];
        if (propagated_costs[ids.x] != pack || propagated_costs[ids.y] != pack ||
            propagated_costs[ids.z] != pack) {
            return;
        }
    }
    // All claim checks used the same immutable snapshot. Winners have disjoint
    // complete stars, so these byte stores have one writer per vertex.
    chosen_edges[tid] = 1;
    active_edges[tid] = 0;
    atomicAdd(progress + 1, 1);
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; ++f) {
        int3 ids = faces[vert2face[f]];
        reserved_vertices[ids.x] = reserved_vertices[ids.y] = reserved_vertices[ids.z] = 1;
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; ++f) {
        int3 ids = faces[vert2face[f]];
        reserved_vertices[ids.x] = reserved_vertices[ids.y] = reserved_vertices[ids.z] = 1;
    }
}


static __global__ void initialize_active_edges_kernel(
    const float* costs, float threshold, int E, uint8_t* active
) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < E) active[edge] = isfinite(costs[edge]) && costs[edge] <= threshold;
}


static __global__ void prune_collapse_edges_kernel(
    const int3* faces, const uint64_t* edges,
    const int* vert2face, const int* offsets,
    const uint8_t* reserved, int E, uint8_t* active, int* progress
) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= E || !active[edge]) return;
    uint64_t key = edges[edge];
    int endpoints[2] = {int(key >> 32), int(key & 0xFFFFFFFF)};
    for (int i = 0; i < 2; ++i) {
        int vertex = endpoints[i];
        for (int f = offsets[vertex]; f < offsets[vertex+1]; ++f) {
            int3 ids = faces[vert2face[f]];
            if (reserved[ids.x] || reserved[ids.y] || reserved[ids.z]) {
                active[edge] = 0;
                return;
            }
        }
    }
    atomicAdd(progress, 1);
}


static __global__ void retain_chosen_costs_kernel(
    const uint8_t* chosen, int E, float* costs
) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < E && !chosen[edge]) costs[edge] = INFINITY;
}


static int select_maximal_collapse_batch(CuMesh& ctx, float threshold, cudaStream_t stream, bool timing,const SimplificationControl& control) {
    size_t V = ctx.vertices.size, E = ctx.edges.size;
    // This CSR/CUB workspace is dead until compression. Keep active/chosen
    // edges and reserved vertices separate from the 8V endpoint-min workspace.
    size_t state_bytes = 2 * E + V;
    size_t progress_offset = (state_bytes + sizeof(int) - 1) & ~(sizeof(int) - 1);
    ctx.cub_temp_storage.resize(progress_offset + 2 * sizeof(int));
    auto* active = reinterpret_cast<uint8_t*>(ctx.cub_temp_storage.ptr);
    auto* chosen = active + E;
    auto* reserved = chosen + E;
    auto* progress = reinterpret_cast<int*>(active + progress_offset);
    CUDA_CHECK(cudaMemsetAsync(active, 0, progress_offset + 2 * sizeof(int), stream));
    initialize_active_edges_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.edge_collapse_costs.ptr, threshold, E, active);
    CUDA_CHECK(cudaGetLastError());

    int counts[2], waves = 0, total_chosen = 0;
    do {
        control.check();
        CUDA_CHECK(cudaMemsetAsync(progress, 0, 2 * sizeof(int), stream));
        select_collapse_edges_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            ctx.faces.ptr, ctx.edges.ptr, ctx.vert2face.ptr, ctx.vert2face_offset.ptr,
            ctx.edge_collapse_costs.ptr, ctx.propagated_costs.ptr,
            active, chosen, reserved, progress, E);
        CUDA_CHECK(cudaGetLastError());
        prune_collapse_edges_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            ctx.faces.ptr, ctx.edges.ptr, ctx.vert2face.ptr, ctx.vert2face_offset.ptr,
            reserved, E, active, progress);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(counts, progress, sizeof(counts), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++waves;
        total_chosen += counts[1];
        // A lowest packed key among remaining candidates always wins. No
        // arbitrary wave cap silently substitutes a non-maximal batch.
        TORCH_CHECK(counts[0] == 0 || counts[1] > 0, "[CuMesh] simplify_selection_no_progress");
        if (counts[0]) propagate_cost(ctx, stream, active);
    } while (counts[0]);
    retain_chosen_costs_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        chosen, E, ctx.edge_collapse_costs.ptr);
    CUDA_CHECK(cudaGetLastError());
    if (timing) std::cout << "collapse_selection: waves=" << waves << " chosen=" << total_chosen << std::endl;
    return total_chosen;
}

static __global__ void restore_spatial_candidates_kernel(
    const float* original, const int* refused, int E, float* costs
) {
    int edge=blockIdx.x*blockDim.x+threadIdx.x;
    if(edge<E) costs[edge]=refused[edge]?INFINITY:original[edge];
}


// Eligibility reads an immutable snapshot before the commit kernel starts.
// Only disjoint-star winners then read/write faces during this commit kernel.
static __global__ void collapse_edges_kernel(
    float3* vertices, int3* faces, const uint64_t* edges,
    const int* vert2face, const int* vert2face_offset,
    const float* edge_collapse_costs,const float3* proposed_positions,
    int E,int* vertices_kept,int* faces_kept,int* merge_partner,
    const QEM* previous_quadrics,double* merge_radius
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E || !isfinite(edge_collapse_costs[tid])) return;
    uint64_t e = edges[tid];
    int e0 = int(e >> 32), e1 = int(e & 0xFFFFFFFF);

    // Commit the same fp32 proposal that passed cost-time geometry predicates.
    if (merge_partner) {
        merge_partner[e0]=e1;
        merge_radius[e0]=cluster_cover_upper(previous_quadrics[e0],previous_quadrics[e1],vertices[e0],vertices[e1],proposed_positions[tid]);
    }
    vertices[e0]=proposed_positions[tid];
    vertices_kept[e1] = 0;
    // delete shared faces
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1 || f_vids.y == e1 || f_vids.z == e1) {
            faces_kept[fid] = 0;
        }
    }
    // update faces
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1) {
            f_vids.x = e0;
        } else if (f_vids.y == e1) {
            f_vids.y = e0;
        } else if (f_vids.z == e1) {
            f_vids.z = e0;
        }
        faces[fid] = f_vids;
    }
}


static __global__ void compress_vertices_kernel(
    const int* vertices_map,
    const float3* old_vertices,
    const int V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    int new_id = vertices_map[tid];
    int is_kept = vertices_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_vertices[new_id] = old_vertices[tid];
    }
}


static __global__ void compress_quadrics_kernel(
    const int* map,const int* partner,const double* merge_radius,const QEM* previous,int V,QEM* next
) {
    int vertex=blockIdx.x*blockDim.x+threadIdx.x;
    if (vertex>=V||map[vertex+1]!=map[vertex]+1) return;
    QEM q=previous[vertex];
    if (partner[vertex]>=0) {q+=previous[partner[vertex]];q.radius=merge_radius[vertex];}
    next[map[vertex]]=q;
}


static __global__ void compress_faces_kernel(
    const int* faces_map,
    const int* vertices_map,
    const int3* old_faces,
    const int F,
    int3* new_faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    int new_id = faces_map[tid];
    int is_kept = faces_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_faces[new_id].x = vertices_map[old_faces[tid].x];
        new_faces[new_id].y = vertices_map[old_faces[tid].y];
        new_faces[new_id].z = vertices_map[old_faces[tid].z];
    }
}


/**
 * Collapse edges parallelly
 */
// Match io.cu: allocate on the actual caller stream and record native use.
static torch::Tensor allocate_quadric_rows(int64_t rows,const torch::TensorOptions& options,
    const c10::cuda::CUDAStream& output_stream,const c10::cuda::CUDAStream& execution_stream) {
    torch::Tensor out;
    {c10::cuda::CUDAStreamGuard allocation_guard(output_stream);out=torch::empty({rows,int64_t(12)},options);}
    if (output_stream!=execution_stream) record_tensor_on_stream(out,execution_stream);
    return out;
}


void collapse_edges(
    CuMesh& ctx,
    float collapse_thresh,
    cudaStream_t stream,
    bool timing,const float3* proposed_positions,
    const torch::Tensor* previous_quadrics,torch::Tensor* next_quadrics,int* merge_partner,double* merge_radius,
    const c10::cuda::CUDAStream& output_stream,const c10::cuda::CUDAStream& execution_stream,const SimplificationControl& control
) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;
    ctx.vertices_map.resize(V + 1);
    ctx.faces_map.resize(F + 1);
    fill_kernel<<<(V+1+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.vertices_map.ptr,
        V + 1,
        1
    );
    fill_kernel<<<(F+1+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.faces_map.ptr,
        F + 1,
        1
    );
    CUDA_CHECK(cudaGetLastError());
    const int device=execution_stream.device_index();
    ctx.last_simplification_admission.clear();
    ScopedCudaTemporary<float> original_costs(E,stream,device);
    CUDA_CHECK(cudaMemcpyAsync(original_costs.get(),ctx.edge_collapse_costs.ptr,E*sizeof(float),cudaMemcpyDeviceToDevice,stream));
    int chosen=select_maximal_collapse_batch(ctx,collapse_thresh,stream,timing,control);
    std::vector<std::unique_ptr<ScopedCudaTemporary<char>>> spatial_storage;
    std::unique_ptr<ExactCollapseAdmission> spatial;
    if(chosen) {
        spatial=std::make_unique<ExactCollapseAdmission>(
            CollapseSurface{ctx.vertices.ptr,ctx.faces.ptr,ctx.edges.ptr,ctx.vert2face.ptr,
                ctx.vert2face_offset.ptr,ctx.edge_collapse_costs.ptr,int(F),int(E)},
            proposed_positions,stream,[&](size_t bytes)->void* {
                auto allocation=std::make_unique<ScopedCudaTemporary<char>>(bytes,stream,device);
                ctx.last_simplification_admission["workspaceBytesExcludingCudaLocalStorage"]+=bytes;
                void* pointer=allocation->get();spatial_storage.push_back(std::move(allocation));return pointer;
            },control.remaining(),[&ctx]{return ctx.simplification_cancelled();});
        do {
            control.check();
            int selected=chosen;
            chosen=spatial->filter_selected();
            if(timing) std::cout << "collapse_spatial: selected=" << selected << " accepted=" << chosen << std::endl;
            if(chosen) break;
            // Ownership selection hid competing endpoint stars. A rejected
            // first batch cannot establish exhaustion of those alternatives.
            restore_spatial_candidates_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,stream>>>(
                original_costs.get(),spatial->refused_edges(),E,ctx.edge_collapse_costs.ptr);
            CUDA_CHECK(cudaGetLastError());
            propagate_cost(ctx,stream);
            chosen=select_maximal_collapse_batch(ctx,collapse_thresh,stream,timing,control);
        } while(chosen);
        auto workspace_bytes=ctx.last_simplification_admission["workspaceBytesExcludingCudaLocalStorage"];
        ctx.last_simplification_admission=spatial->statistics();
        ctx.last_simplification_admission["workspaceBytesExcludingCudaLocalStorage"]=workspace_bytes;
    }
    if(!chosen) {
        // A floor/refusal needs no new quadric rows, scans or geometry copies.
        // The caller already recorded the immutable Tensor on this stream.
        if(next_quadrics) *next_quadrics=*previous_quadrics;
        return;
    }
    // The newly introduced large Torch allocation must fail before geometry
    // changes. Disjoint chosen stars each remove exactly one endpoint.
    if (next_quadrics) *next_quadrics=allocate_quadric_rows(int64_t(V)-chosen,previous_quadrics->options(),output_stream,execution_stream);

    // Admit every compaction workspace before committing geometry. Buffer
    // resize discards bytes on growth, so reserve separate vertex output and
    // retain both original surfaces until face compaction has finished.
    ScopedCudaTemporary<float3> compact_vertices(V,stream,device);
    ctx.temp_storage.resize(F*sizeof(int3));
    size_t vertex_scan_bytes=0,face_scan_bytes=0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,vertex_scan_bytes,ctx.vertices_map.ptr,V+1,stream));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,face_scan_bytes,ctx.faces_map.ptr,F+1,stream));
    ctx.cub_temp_storage.resize(std::max(vertex_scan_bytes,face_scan_bytes));

    control.check();
    collapse_edges_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.edges.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edge_collapse_costs.ptr,
        proposed_positions,
        E,
        ctx.vertices_map.ptr,
        ctx.faces_map.ptr,merge_partner,previous_quadrics?reinterpret_cast<const QEM*>(previous_quadrics->data_ptr<double>()):nullptr,merge_radius
    );
    CUDA_CHECK(cudaGetLastError());

    // update vertices buffer
    // get vertices map
    size_t temp_storage_bytes = vertex_scan_bytes;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ctx.cub_temp_storage.ptr, temp_storage_bytes,
        ctx.vertices_map.ptr, V+1, stream
    ));
    int new_num_vertices;
    CUDA_CHECK(cudaMemcpyAsync(&new_num_vertices, ctx.vertices_map.ptr + V, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (next_quadrics) {
        TORCH_CHECK(next_quadrics->size(0)==new_num_vertices,"[CuMesh] accumulated quadric vertex count mismatch");
        compress_quadrics_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,stream>>>(
            ctx.vertices_map.ptr,merge_partner,merge_radius,reinterpret_cast<const QEM*>(previous_quadrics->data_ptr<double>()),V,
            reinterpret_cast<QEM*>(next_quadrics->data_ptr<double>()));
        CUDA_CHECK(cudaGetLastError());
    }
    // compress vertices
    compress_vertices_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.vertices_map.ptr,
        ctx.vertices.ptr,
        V,
        compact_vertices.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // update faces buffer
    // get faces map
    temp_storage_bytes=face_scan_bytes;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ctx.cub_temp_storage.ptr, temp_storage_bytes,
        ctx.faces_map.ptr, F+1, stream
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpyAsync(&new_num_faces, ctx.faces_map.ptr + F, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // compress faces
    compress_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        ctx.faces_map.ptr,
        ctx.vertices_map.ptr,
        ctx.faces.ptr,
        F,
        reinterpret_cast<int3*>(ctx.temp_storage.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(ctx.vertices.release_async(stream,device));
    TORCH_CHECK(compact_vertices.release_ownership_to(ctx.vertices,new_num_vertices,V,device),
        "[CuMesh] simplify_compaction_publication_refused");
    ctx.temp_storage.resize(new_num_faces*sizeof(int3));
    swap_buffers(ctx.temp_storage, ctx.faces);
}


static std::tuple<int,int> simplify_step_impl(CuMesh& ctx,float lambda_edge_length,float lambda_skinny,float threshold,bool timing,const torch::Tensor* previous_quadrics,torch::Tensor* next_quadrics,double max_error_squared,const c10::cuda::CUDAStream* output_stream,const SimplificationControl& control) {
    CuMeshOperationScope operation(ctx);
    control.check();
    cudaStream_t stream=operation.stream();
    std::chrono::high_resolution_clock::time_point start, end;

    if (timing) start = std::chrono::high_resolution_clock::now();
    ctx.get_vertex_face_adjacency();
    if (timing) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        std::cout << "get_vertex_face_adjacency: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    ctx.get_edges();
    ctx.get_boundary_info();
    get_collapse_vertex_fans(ctx, stream);
    if (timing) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        std::cout << "get_edges: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    const QEM* quadrics;
    if (previous_quadrics) quadrics=reinterpret_cast<const QEM*>(previous_quadrics->data_ptr<double>());
    else {get_qem(ctx,stream);quadrics=reinterpret_cast<const QEM*>(ctx.temp_storage.ptr);}
    ScopedCudaTemporary<float3> proposals(ctx.edges.size,stream,operation.device_index());
    ScopedCudaTemporary<int> partners(next_quadrics?ctx.vertices.size:0,stream,operation.device_index());
    ScopedCudaTemporary<double> radii(next_quadrics?ctx.vertices.size:0,stream,operation.device_index());
    if (next_quadrics) {
        fill_kernel<<<(ctx.vertices.size+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,stream>>>(partners.get(),ctx.vertices.size,-1);
        CUDA_CHECK(cudaGetLastError());
    }
    if (timing) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        std::cout << "get_qem: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    get_edge_collapse_cost(ctx,lambda_edge_length,lambda_skinny,stream,quadrics,proposals.get(),max_error_squared);
    if (timing) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        std::cout << "get_edge_collapse_cost: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    propagate_cost(ctx, stream);
    if (timing) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        std::cout << "propagate_cost: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    collapse_edges(ctx,threshold,stream,timing,proposals.get(),previous_quadrics,next_quadrics,partners.get(),radii.get(),output_stream?*output_stream:operation.caller_stream(),operation.execution_stream(),control);
    if (timing) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        std::cout << "collapse_edges: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    // Delete all cached info since mesh has changed
    ctx.clear_cache();

    return std::make_tuple(ctx.vertices.size, ctx.faces.size);
}


std::tuple<int,int> CuMesh::simplify_step(float edge,float skinny,float threshold,bool timing,double max_step_seconds) {
    TORCH_CHECK(max_step_seconds>=0.0&&!std::isnan(max_step_seconds),"invalid step budget");
    SimplificationControl control{*this,max_step_seconds};
    return simplify_step_impl(*this,edge,skinny,threshold,timing,nullptr,nullptr,INFINITY,nullptr,control);
}

std::map<std::string,uint64_t> CuMesh::simplification_admission_statistics() {
    CuMeshOperationScope operation(*this);
    return last_simplification_admission;
}

// Call-owned Tensor state survives clear_cache without adding sticky native state.
torch::Tensor CuMesh::initial_simplification_quadrics() {
    CuMeshOperationScope operation(*this);
    get_vertex_face_adjacency();
    static_assert(sizeof(QEM)==12*sizeof(double),"quadric Tensor layout");
    auto out=allocate_quadric_rows(int64_t(vertices.size),torch::TensorOptions().dtype(torch::kFloat64).device(torch::Device(torch::kCUDA,operation.device_index())),operation.caller_stream(),operation.execution_stream());
    get_qem_kernel<<<(vertices.size+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,operation.stream()>>>(
        vertices.ptr,faces.ptr,vert2face.ptr,vert2face_offset.ptr,vertices.size,faces.size,reinterpret_cast<QEM*>(out.data_ptr<double>()));
    CUDA_CHECK(cudaGetLastError());
    return out;
}

std::tuple<int,int,torch::Tensor> CuMesh::simplify_step_accumulated(
    const torch::Tensor& quadrics,float edge,float skinny,float threshold,double max_error_squared,bool timing,double max_step_seconds
) {
    TORCH_CHECK(max_step_seconds>=0.0&&!std::isnan(max_step_seconds),"invalid step budget");
    SimplificationControl control{*this,max_step_seconds};
    CuMeshOperationScope operation(*this);
    control.check();
    TORCH_CHECK(quadrics.is_cuda()&&quadrics.get_device()==operation.device_index(),"quadrics must share canonical CUDA device");
    TORCH_CHECK(quadrics.scalar_type()==torch::kFloat64&&quadrics.is_contiguous(),"quadrics must be contiguous float64");
    TORCH_CHECK(quadrics.dim()==2&&quadrics.size(0)==vertices.size&&quadrics.size(1)==12,"quadrics must match vertices and 12-double layout");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(quadrics.data_ptr<double>())%alignof(QEM)==0,"quadrics must be 16-byte aligned");
    TORCH_CHECK(max_error_squared>=0.0&&!std::isnan(max_error_squared),"invalid quadric error bound");
    record_tensor_on_stream(quadrics,operation.execution_stream());
    torch::Tensor next;
    auto caller=operation.caller_stream();
    auto counts=simplify_step_impl(*this,edge,skinny,threshold,timing,&quadrics,&next,max_error_squared,&caller,control);
    return std::make_tuple(std::get<0>(counts),std::get<1>(counts),next);
}


} // namespace cumesh
