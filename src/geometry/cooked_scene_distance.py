"""Experimental stateless cooked-hull query for the fixed current scene.

Unsigned separation and closest-point derivatives. No penetration gradient.
USD exclusions and full collider coverage must pass the scene audit before use.
"""
from itertools import combinations,product
import time
import numpy as np
from src.geometry.convex_distance import convex_distance
from src.geometry.physx_clearance import PhysXClearance


class CookedSceneDistance:
    def __init__(self,scene,activation_distance=.08):
        from pxr import UsdGeom,UsdPhysics,PhysicsSchemaTools,Gf
        from omni.physx import get_physx_cooking_interface
        import omni.usd
        self.scene=scene;self.activation_distance=activation_distance;self.jacobian=PhysXClearance(scene)
        stage=scene.monitor.stage;cache=UsdGeom.XformCache();cooking=get_physx_cooking_interface()
        self.hulls={};self.static={};self.inventory=[];self.exclusions=set();self.unsupported=[];self.planes=[]
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint=UsdPhysics.Joint(prim)
                if not joint.GetCollisionEnabledAttr().Get():
                    a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
                    if a and b:self.exclusions.add(tuple(sorted((str(a[0]),str(b[0])))))
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                for other in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                    self.exclusions.add(tuple(sorted((str(prim.GetPath()),str(other)))))
            if prim.IsA(UsdPhysics.CollisionGroup):self.unsupported.append('CollisionGroup:'+str(prim.GetPath()))
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
            body=prim
            while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            if not body.IsValid():
                if prim.GetTypeName()=='Plane':
                    axis=str(UsdGeom.Plane(prim).GetAxisAttr().Get());m=np.array(cache.GetLocalToWorldTransform(prim)).T
                    normal=m[:3,'XYZ'.index(axis)];normal=normal/np.linalg.norm(normal)
                    self.planes.append((str(prim.GetPath()),normal,m[:3,3]));continue
                # A collider without RigidBodyAPI is a fixed world collider.
                body=prim
            owner=str(body.GetPath());path=str(prim.GetPath());vertices=[];typ=prim.GetTypeName();status=[]
            if typ=='Mesh':
                def on_result(result,data):
                    status.append(str(result))
                    for hull in data:vertices.append(np.array([list(x) for x in hull.vertices],np.float64))
                cooking.request_convex_collision_representation(stage_id=omni.usd.get_context().get_stage_id(),
                    collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),run_asynchronously=False,on_result=on_result)
            elif typ=='Cube':
                size=float(UsdGeom.Cube(prim).GetSizeAttr().Get());vertices=[np.array(list(product((-size/2,size/2),repeat=3)))]
            if not vertices:self.unsupported.append(path+':'+typ)
            body_world=cache.GetLocalToWorldTransform(body)
            rigid=np.eye(4);rigid[:3,:3]=np.array(Gf.Matrix3d(Gf.Transform(body_world).GetRotation())).T
            rigid[:3,3]=np.array(body_world.ExtractTranslation())
            # PhysX body poses carry rotation/translation, never USD scale or
            # reflection. Bake those into collider vertices, including tools.
            relative=np.linalg.inv(rigid)@np.array(cache.GetLocalToWorldTransform(prim)).T
            self.hulls.setdefault(owner,[]).extend(v@relative[:3,:3].T+relative[:3,3] for v in vertices)
            self.static[owner]=rigid
            self.inventory.append({'path':path,'body':owner,'type':typ,'hulls':len(vertices),'status':status})
        def excluded(a,b):
            for x,y in self.exclusions:
                if ((a==x or a.startswith(x+'/')) and (b==y or b.startswith(y+'/'))) or ((b==x or b.startswith(x+'/')) and (a==y or a.startswith(y+'/'))):return True
            return False
        self.pairs=[(a,b,scene.monitor._kind(a,b)) for a,b in combinations(sorted(self.hulls),2)
            if scene.monitor._kind(a,b) in ('left_self','left_environment','inter_arm') and not excluded(a,b)]
        if self.unsupported:raise RuntimeError('unsupported_geometry:'+str(self.unsupported))
        self.reset()

    def reset(self):self.calls=0;self.last_rows=[]

    def query(self,q=None):
        # Caller holds q in the simulator. No probe/physics step is performed.
        from src.geometry.rotations import quaternion_wxyz_to_matrix
        started=time.perf_counter();scene=self.scene
        pose=scene.articulation.data.body_link_pose_w[0]
        rotation=quaternion_wxyz_to_matrix(pose[:,3:7]).cpu().numpy().astype(np.float64)
        translation=pose[:,:3].cpu().numpy().astype(np.float64)
        world={};bounds={}
        for path,hulls in self.hulls.items():
            i=scene.state.body_indices.get(path)
            if i is None:m=self.static[path];r,t=m[:3,:3],m[:3,3]
            else:r,t=rotation[i],translation[i]
            shapes=[v@r.T+t for v in hulls];world[path]=shapes
            bounds[path]=(np.min([v.min(0) for v in shapes],axis=0),np.max([v.max(0) for v in shapes],axis=0))
        jac,anchors=self.jacobian.body_jacobians()
        def point_jac(path,point):
            i=scene.state.body_indices.get(path)
            return np.zeros((3,8)) if i is None else jac[i,:3]+np.cross(jac[i,3:].T,point-anchors[i]).T
        def lower_bound(alo,ahi,blo,bhi):return float(np.linalg.norm(np.maximum(np.maximum(alo-bhi,blo-ahi),0)))
        rows=[];queries=0;candidates=0
        for plane,normal,origin in self.planes:
            for body,shapes in world.items():
                if scene.monitor._side(body)!='left':continue
                vertices=np.concatenate(shapes);heights=(vertices-origin)@normal;i=int(np.argmin(heights));height=float(heights[i])
                if height>self.activation_distance:continue
                point=vertices[i]
                rows.append({'pair':(body,plane),'kind':'left_environment','distance':max(0,height),
                    'normal':normal,'point_a':point,'point_b':point-height*normal,'gradient':normal@point_jac(body,point)})
        for a,b,kind in self.pairs:
            if lower_bound(*bounds[a],*bounds[b])>self.activation_distance:continue
            candidates+=1;best=None
            for va in world[a]:
                for vb in world[b]:
                    if lower_bound(va.min(0),va.max(0),vb.min(0),vb.max(0))>(best['distance'] if best else self.activation_distance):continue
                    try:result=convex_distance(va,vb)
                    except RuntimeError:
                        self.failed_query={'vertices_a':va,'vertices_b':vb,'pair':np.array((a,b))}
                        raise
                    queries+=1
                    if best is None or result['distance']<best['distance']:best=result
            if best is None or best['distance']>self.activation_distance:continue
            gradient=best['normal']@(point_jac(a,best['point_a'])-point_jac(b,best['point_b']))
            rows.append({'pair':(a,b),'kind':kind,**best,'gradient':gradient})
        self.last_rows=rows;self.calls+=1
        d=np.array([r['distance'] for r in rows]);g=np.array([r['gradient'] for r in rows]).reshape(-1,8)
        mins=np.array([min((r['distance'] for r in rows if (r['kind']=='left_self')==is_self),default=float('nan')) for is_self in (True,False)])
        return d,g,mins,{'active_pairs':len(rows),'broadphase_candidates':candidates,'hull_queries':queries,'query_ms':(time.perf_counter()-started)*1000,
            'unsigned_overlap_pairs':sum(r['distance']<1e-8 for r in rows),'fd_queries':0,'gradient_audit_max_error':None,'force_fd':False}
