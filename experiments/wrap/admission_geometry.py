"""CPU fixture geometry for admission controls; never executes a CuMesh operation."""
import numpy as np


def oriented_triangles(v,f):
    return {min(tuple(np.roll(t,i,axis=0).reshape(-1)) for i in range(3)) for t in v[f]}


def exact_codes(v,f,library):
    pairs=np.array([(a,b) for a in range(len(f)) for b in range(a+1,len(f))],dtype=np.int32)
    triangles=np.ascontiguousarray(v[f[pairs]].reshape(-1,18),dtype=np.float64)
    # Every fixture has unique exact positions; faces supply canonical IDs.
    groups=np.ascontiguousarray(f[pairs].reshape(-1,6),dtype=np.int32);out=np.zeros(len(pairs),dtype=np.int32)
    library.exact_pair_codes(triangles.ctypes.data,groups.ctypes.data,out.ctypes.data,len(pairs))
    return {'intersections':int(np.count_nonzero((out&3)==1)),'undecided':int(np.count_nonzero((out&3)==2))}


def trap_geometry(v,f,nv,nf,library):
    if len(nf)!=4:raise RuntimeError('fixture unconstrained public collapse did not produce a tetrahedron')
    original=v[f].astype(np.float64);normals=np.cross(original[:,1]-original[:,0],original[:,2]-original[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    # Pick a new triangle whose centroid is strictly inside every old halfspace.
    centroids=nv[nf].astype(np.float64).mean(axis=1)
    margins=np.array([np.min(-np.einsum('ij,ij->i',p-original[:,0],normals)) for p in centroids]);index=int(np.argmax(margins));center=centroids[index];margin=float(margins[index])
    if margin<=0.001:raise RuntimeError('no interior collapse witness available')
    epsilon=margin/32.0
    tv=(center+epsilon*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])).astype(np.float32)
    tf=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.int32)
    normals_t=np.cross(tv[tf[:,1]]-tv[tf[:,0]],tv[tf[:,2]]-tv[tf[:,0]]);wrong=np.einsum('ij,ij->i',normals_t,tv[tf].mean(1)-tv.mean(0))<0;tf[wrong]=tf[wrong][:,(0,2,1)]
    av=np.concatenate([v,tv]);af=np.concatenate([f,tf+len(v)])
    if exact_codes(av,af,library)!={'intersections':0,'undecided':0}:raise RuntimeError('trap input is not clean')
    # Geometric witness: the old preferred collapsed surface cuts the trap.
    badv=np.concatenate([nv,tv]);badf=np.concatenate([nf,tf+len(nv)])
    if exact_codes(badv,badf,library)['intersections']==0:raise RuntimeError('trap does not actually obstruct preferred proposal')
    return v,f,tv,tf
