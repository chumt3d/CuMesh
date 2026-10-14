"""CPU preflight of real installed APIs and all device-free admission fixture paths.

This validates call shapes, input geometry and CPU guards. It never substitutes
for native GPU admission, cancellation, allocation or complete product proof.
"""
import argparse,ast,ctypes,hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
import cumesh
import cumesh_wrap
from cumesh_wrap.warp_bvh import WarpUnsignedDistanceBVH,validate_mesh
from admission_geometry import exact_codes,trap_geometry,oriented_triangles,resumption_geometry
from collapse_controls import fixture,audit


def signature(target):
    try:return inspect.signature(target),'inspect.signature'
    except ValueError:
        # Pybind11 does not publish __text_signature__; its real built-in docs
        # carry named arguments/defaults. No class or callable is replaced.
        line=(target.__doc__ or '').splitlines()[0]
        if not line.startswith(target.__name__+'('):raise
        definition=ast.parse('def '+line.split(' -> ',1)[0]+': pass').body[0]
        args=definition.args;parameters=[];required=len(args.args)-len(args.defaults)
        for i,arg in enumerate(args.args):
            parameters.append(inspect.Parameter(arg.arg,inspect.Parameter.POSITIONAL_OR_KEYWORD,default=inspect.Parameter.empty if i<required else Ellipsis))
        if parameters and parameters[0].name=='self':parameters=parameters[1:]
        return inspect.Signature(parameters),'real_native_builtin_doc_after_inspect_ValueError'


def chain(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):
        base=chain(node.value)
        return base+'.'+node.attr if base else None
    return None


def preflight(args):
    begun=time.monotonic();mesh=cumesh.CuMesh();mesh.simplify(1,verbose=False)
    native=mesh.cu_mesh
    assert torch.__version__=='2.6.0+cu124'
    binary_sha=hashlib.sha256(Path(cumesh._C.__file__).read_bytes()).hexdigest()
    if args.binary_sha and binary_sha!=args.binary_sha:raise RuntimeError('installed_native_binary_hash_mismatch')
    roots={ 'mesh':mesh,'base':mesh,'cumesh.CuMesh':cumesh.CuMesh,
            'WarpUnsignedDistanceBVH':WarpUnsignedDistanceBVH,
            'bvh':WarpUnsignedDistanceBVH,'witness_bvh':WarpUnsignedDistanceBVH }
    files=[Path(__file__).with_name('collapse_controls.py'),args.admission or Path(__file__).with_name('admission_controls.py')]
    package=Path(cumesh_wrap.__file__).parent
    files.extend(package/n for n in ('reduction.py','engine.py','deviation.py'))
    rows=[];calls=0;attributes=[]
    initializer=ast.parse(inspect.getsource(WarpUnsignedDistanceBVH))
    device_fields={chain(t).removeprefix('self.') for n in ast.walk(initializer) if isinstance(n,ast.Assign) for t in n.targets if (chain(t) or '').startswith('self.')}
    for path in files:
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute):
                route=chain(node);root=next((r for r in ('mesh','base','bvh','witness_bvh') if route and route.startswith(r+'.')),None)
                if root:
                    value=roots[root];source='real_installed_object_attribute'
                    tail=route[len(root)+1:].split('.')
                    if root in ('bvh','witness_bvh') and tail[0] in device_fields:
                        source='installed_instance_field_declaration_GPU_state_deferred'
                    else:
                        for name in tail:value=getattr(value,name)
                    attributes.append({'file':str(path),'line':node.lineno,'attribute':route,'source':source})
            if not isinstance(node,ast.Call):continue
            route=chain(node.func)
            if not route:continue
            root=next((r for r in roots if route==r or route.startswith(r+'.')),None)
            if root is None:continue
            target=roots[root]
            for attr in route[len(root):].strip('.').split('.') if route!=root else []:target=getattr(target,attr)
            sig,source=signature(target)
            positional=[None for _ in node.args];keywords={k.arg:None for k in node.keywords if k.arg is not None}
            if any(k.arg is None for k in node.keywords) or any(isinstance(a,ast.Starred) for a in node.args):raise RuntimeError('unresolved_API_call_expansion:'+route)
            # BVH methods are inspected on their real class because creating
            # its CUDA mesh is a device-dependent operation.
            if root in ('bvh','witness_bvh'):positional.insert(0,None)
            sig.bind(*positional,**keywords)
            rows.append({'file':str(path),'line':node.lineno,'call':route,'signature':str(sig),'signatureSource':source});calls+=1
    # CPU-created public state: no tensor/device allocation or fake object.
    assert isinstance(mesh.last_simplification,dict) and native.num_faces()==0
    mesh.cancel_simplification();assert native.simplification_cancelled();native.clear_simplification_cancel();assert not native.simplification_cancelled()
    fixture_rows=[]
    names=('cube72','terminal_tetra','invalid_equator','safe_bipyramid','boundary_disk','nan_no_commit','terminal_public','boundary_chord','last_triangle','offset_disk','disconnected_fan','repeated_index','error_floor_tetra','error_first_cube','zero_plane_budget','quadric_cross_stream','quadric_alignment')
    for name in names:
        v,f=fixture(name);report=audit(v,f)
        if not (v.dtype==np.float32 and f.dtype==np.int32):raise RuntimeError('fixture_storage_dtype:'+name)
        rejected=None
        try:validate_mesh(torch.from_numpy(v),torch.from_numpy(f),require_cuda=False)
        except ValueError as error:
            if name!='nan_no_commit':raise
            rejected=str(error)
        try:WarpUnsignedDistanceBVH(torch.from_numpy(v),torch.from_numpy(f),max_distance=1.)
        except ValueError as error:
            if str(error)!='vertices and faces must be on the same CUDA device':raise
        else:raise RuntimeError('BVH_device_guard_missing')
        fixture_rows.append({'name':name,'vertices':len(v),'faces':len(f),'audit':report,'CPUValidatorRefusal':rejected,'BVHCPURefusal':'passed'})
    library=ctypes.CDLL(str(args.bridge));library.exact_pair_codes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
    original=np.load(args.before);collapsed=np.load(args.after);v,f=original['vertices'],original['faces'];nv,nf=collapsed['vertices'],collapsed['faces']
    v,f,tv,tf=trap_geometry(v,f,nv,nf,library)
    clean=exact_codes(np.concatenate([v,tv]),np.concatenate([f,tf+len(v)]),library)
    blocked=exact_codes(np.concatenate([nv,tv]),np.concatenate([nf,tf+len(nv)]),library)
    mixed=exact_codes(np.concatenate([v,v+np.array([5,0,0],dtype=np.float32),tv]),np.concatenate([f,f+len(v),tf+2*len(v)]),library)
    invalid=exact_codes(np.concatenate([v,v+np.array([.05,.07,.03],dtype=np.float32)]),np.concatenate([f,f+len(v)]),library)
    assert clean==mixed=={'intersections':0,'undecided':0} and blocked['intersections']>0 and invalid['intersections']>0 and blocked['undecided']==invalid['undecided']==0
    iv,iff,dvs,dfs=resumption_geometry(v,f,nv,tf)
    resumed_input=exact_codes(iv,iff,library)
    resumed_expected=exact_codes(np.concatenate([nv,dvs]),np.concatenate([nf,dfs+len(nv)]),library)
    assert resumed_input==resumed_expected=={'intersections':0,'undecided':0}
    assert oriented_triangles(v,f)==oriented_triangles(v,f[:,[1,2,0]])
    result={'status':'passed','seconds':time.monotonic()-begun,'torch':torch.__version__,'nativeBinarySha256':binary_sha,'wrapInstalledPath':str(package),'CuMeshCallsChecked':calls,'calls':rows,'attributes':attributes,'fixtures':fixture_rows,'trapExact':{'clean':clean,'blocked':blocked,'mixedInput':mixed,'invalidInput':invalid,'resumptionInput':resumed_input,'resumptionExpected':resumed_expected},'GPUProof':False,'BVHInstanceAttributes':'CUDA-instance fields remain GPU-bound; installed snapshot methods checked; no cubvh.cuBVH is used'}
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','seconds','CuMeshCallsChecked','torch','nativeBinarySha256','trapExact')}));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--bridge',required=True,type=Path);p.add_argument('--before',required=True,type=Path);p.add_argument('--after',required=True,type=Path);p.add_argument('--binary-sha');p.add_argument('--admission',type=Path);preflight(p.parse_args())
