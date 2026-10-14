"""Real public CuMesh admission controls; host exact oracle only checks outputs.

The adversarial obstacle is placed on a measured unconstrained public collapse,
strictly inside the original convex body. Adding a locked tetrahedron changes
neither that body's QEM planes nor candidate ranking; it makes the old preferred
proposal unsafe. Output validity and untouched geometry are independent checks.
"""
import argparse,ctypes,json,time,threading
from pathlib import Path
import numpy as np
import torch
import cumesh
from collapse_controls import fixture,cube_lattice


def oriented_triangles(v,f):
    return {min(tuple(np.roll(t,i,axis=0).reshape(-1)) for i in range(3)) for t in v[f]}


def mesh_for(v,f):
    mesh=cumesh.CuMesh();mesh.init(torch.from_numpy(v).cuda(),torch.from_numpy(f).cuda());return mesh


def arrays(mesh):
    v,f=mesh.read();torch.cuda.synchronize();return v.cpu().numpy().copy(),f.cpu().numpy().copy()


def exact_codes(v,f,library):
    pairs=np.array([(a,b) for a in range(len(f)) for b in range(a+1,len(f))],dtype=np.int32)
    triangles=np.ascontiguousarray(v[f[pairs]].reshape(-1,18),dtype=np.float64)
    # Every fixture has unique exact positions; faces supply canonical IDs.
    groups=np.ascontiguousarray(f[pairs].reshape(-1,6),dtype=np.int32);out=np.zeros(len(pairs),dtype=np.int32)
    library.exact_pair_codes(triangles.ctypes.data,groups.ctypes.data,out.ctypes.data,len(pairs))
    return {'intersections':int(np.count_nonzero((out&3)==1)),'undecided':int(np.count_nonzero((out&3)==2))}


def step(mesh,seconds=float('inf')):
    quadrics=mesh.cu_mesh.initial_simplification_quadrics();begun=time.monotonic()
    value=mesh.cu_mesh.simplify_step_accumulated(quadrics,1.0,0.0,1000.0,float('inf'),False,seconds)
    torch.cuda.synchronize();return value,time.monotonic()-begun


def trapped_body(library):
    v,f=fixture('safe_bipyramid');mesh=mesh_for(v,f)
    try:
        step(mesh);nv,nf=arrays(mesh)
    finally:mesh.free()
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


def run(out,bridge):
    out.mkdir(parents=True,exist_ok=False);library=ctypes.CDLL(str(bridge));library.exact_pair_codes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
    rows=[]
    def record(name,fn):
        begun=time.monotonic();data=fn();data.update(name=name,seconds=time.monotonic()-begun);rows.append(data);(out/'admission-controls.json').write_text(json.dumps(rows,indent=2)+'\n')
        if not data['passed']:raise RuntimeError('admission control failed: '+name)
    v,f,tv,tf=trapped_body(library)
    def refill():
        iv=np.concatenate([v,tv]);iff=np.concatenate([f,tf+len(v)]);mesh=mesh_for(iv,iff)
        try:
            value,seconds=step(mesh);ov,of=arrays(mesh);stats=mesh.cu_mesh.simplification_admission_statistics();valid=exact_codes(ov,of,library)
            return {'passed':stats.get('selectionBatches',0)>=2 and len(of)<len(iff) and valid=={'intersections':0,'undecided':0} and value[2].shape[0]==len(ov),'statistics':stats,'exact':valid,'nativeSeconds':seconds,'facesBefore':len(iff),'facesAfter':len(of)}
        finally:mesh.free()
    record('refill_hidden_alternative',refill)
    def mixed():
        bv=v+np.array([5,0,0],dtype=np.float32);iv=np.concatenate([v,bv,tv]);iff=np.concatenate([f,f+len(v),tf+2*len(v)]);mesh=mesh_for(iv,iff)
        try:
            value,seconds=step(mesh);ov,of=arrays(mesh);valid=exact_codes(ov,of,library);same=oriented_triangles(v,f).issubset(oriented_triangles(ov,of))
            return {'passed':len(of)==len(iff)-2 and same and value[2].shape[0]==len(ov) and valid=={'intersections':0,'undecided':0},'refusedBodyUnchanged':same,'statistics':mesh.cu_mesh.simplification_admission_statistics(),'exact':valid,'nativeSeconds':seconds,'facesBefore':len(iff),'facesAfter':len(of)}
        finally:mesh.free()
    record('mixed_accepted_subset',mixed)
    def resumption():
        base=mesh_for(v,f)
        try:step(base);expected_v,expected_f=arrays(base)
        finally:base.free()
        points=expected_v
        if len(points)!=4:raise RuntimeError('resumption fixture requires four output vertices')
        positions=[]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weights=np.array([.2+i*.05,.2+j*.05,.2+k*.05,1-.6-(i+j+k)*.05])
                    positions.append(weights@points)
        original=v[f].astype(np.float64);n=np.cross(original[:,1]-original[:,0],original[:,2]-original[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
        margin=min(np.min(-np.einsum('ij,ij->i',c-original[:,0],n)) for c in positions)
        if margin<=0:raise RuntimeError('contained decoy fixture is outside original body')
        epsilon=margin/1024
        dvs=np.concatenate([(c+epsilon*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])).astype(np.float32) for c in positions]);dfs=np.concatenate([tf+4*i for i in range(len(positions))])
        iv=np.concatenate([v,dvs]);iff=np.concatenate([f,dfs+len(v)]);mesh=mesh_for(iv,iff)
        try:
            value,seconds=step(mesh);ov,of=arrays(mesh);stats=mesh.cu_mesh.simplification_admission_statistics();oracle=exact_codes(ov,of,library)
            expected=oriented_triangles(expected_v,expected_f)|oriented_triangles(dvs,dfs)
            return {'passed':oriented_triangles(ov,of)==expected and oracle=={'intersections':0,'undecided':0} and stats.get('maximumNodeVisitsPerQuery',0)>64,'statistics':stats,'exact':oracle,'nativeSeconds':seconds,'containedLockedParts':len(positions)}
        finally:mesh.free()
    record('resumption_preserves_contained_part_surface',resumption)
    def invalid():
        iv=np.concatenate([v,v+np.array([.05,.07,.03],dtype=np.float32)]);iff=np.concatenate([f,f+len(v)]);mesh=mesh_for(iv,iff);error=None
        try:
            try:step(mesh,2.0)
            except RuntimeError as e:error=str(e)
            ov,of=arrays(mesh)
            return {'passed':error is not None and 'simplify_spatial_input_intersection' in error and np.array_equal(iv,ov) and np.array_equal(iff,of),'failure':error,'geometryUnchanged':np.array_equal(iv,ov) and np.array_equal(iff,of)}
        finally:mesh.free()
    record('decisive_input_refusal_no_commit',invalid)
    cv,cf=cube_lattice(64)
    # Warm the same path and measure an unbounded step before choosing a budget.
    mesh=mesh_for(cv,cf)
    try:_,calibration=step(mesh)
    finally:mesh.free()
    def deadline():
        mesh=mesh_for(cv,cf);error=None;begun=time.monotonic();quadrics=mesh.cu_mesh.initial_simplification_quadrics();budget=calibration/4
        try:
            try:mesh.cu_mesh.simplify_step_accumulated(quadrics,1.,0.,1000.,float('inf'),False,budget)
            except RuntimeError as e:error=str(e)
            ov,of=arrays(mesh)
            return {'passed':error is not None and 'simplify_spatial_deadline' in error and np.array_equal(cv,ov) and np.array_equal(cf,of),'failure':error,'geometryUnchanged':np.array_equal(cv,ov) and np.array_equal(cf,of),'requestedSeconds':budget,'elapsedSeconds':time.monotonic()-begun,'unboundedCalibrationSeconds':calibration}
        finally:mesh.free()
    record('native_deadline_no_commit',deadline)
    def cancel():
        mesh=mesh_for(cv,cf);quadrics=mesh.cu_mesh.initial_simplification_quadrics();error=None;timer=threading.Timer(calibration/8,mesh.cancel_simplification);timer.start();begun=time.monotonic()
        try:
            try:mesh.cu_mesh.simplify_step_accumulated(quadrics,1.,0.,1000.,float('inf'),False,10.)
            except RuntimeError as e:error=str(e)
            timer.join();ov,of=arrays(mesh)
            return {'passed':error is not None and 'simplify_spatial_cancelled' in error and np.array_equal(cv,ov) and np.array_equal(cf,of),'failure':error,'geometryUnchanged':np.array_equal(cv,ov) and np.array_equal(cf,of),'cancelDelaySeconds':calibration/8,'elapsedSeconds':time.monotonic()-begun}
        finally:timer.cancel();timer.join();mesh.free()
    record('concurrent_cancel_no_commit',cancel)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--bridge',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.out,a.bridge),indent=2))
