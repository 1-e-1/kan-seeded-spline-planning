"""Experimental metric near-contact separations, NOT a global signed field.

Uses PhysX cooked collider geometry and its existing collision exclusions.
Missing contact patches mean out-of-query-range/unknown, not infinite distance.
World-space normal/Jacobian derivatives must pass the independent FD audit.
"""
import numpy as np


def configure_proximity_stage(stage,contact_offset=.06):
    from pxr import Usd,UsdPhysics,PhysxSchema
    counts={'colliders':0,'fixed_root_bodies':0}
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr().Set(contact_offset)
                counts['colliders']+=1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and str(prim.GetPath()).count('/')==1:
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr().Set(False)
                counts['fixed_root_bodies']+=1
    return counts


def initialize_all_body_filters(scene):
    from src.env.dual_arm_scene import LEFT_PREFIX,RIGHT_PREFIX,LEFT_TOOL,RIGHT_TOOL
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import UsdPhysics
    bodies=scene.state.body_paths
    arms=[s for s in bodies if s.startswith(LEFT_PREFIX+'/') or s.startswith(RIGHT_PREFIX+'/')]
    fixed=[s for s in bodies if s not in arms and s not in (LEFT_TOOL,RIGHT_TOOL)]
    extras=[str(p.GetPath()) for p in scene.monitor.stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()) not in bodies]
    simview=SimulationManager.get_physics_sim_view()
    scene.monitor.initialize_tensor_views(simview,arms,[*fixed,*extras])
    views=[]
    for path in extras:
        assert UsdPhysics.RigidBodyAPI(scene.monitor.stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Get()
        v=simview.create_rigid_body_view(path);views.append((path,v,v.get_transforms().clone()))
    return views,{'arms':arms,'fixed':fixed,'extras':extras}


class PhysXClearance:
    def __init__(self,scene,anchor='com'):
        self.scene=scene;self.anchor=anchor

    def body_jacobians(self):
        scene=self.scene;paths=scene.state.body_paths
        raw=scene.articulation.root_physx_view.get_jacobians()[0].cpu().numpy().astype(np.float64)
        assert raw.shape[1]==6
        fixed=raw.shape[0]==len(paths)-1
        assert fixed or raw.shape[0]==len(paths)
        columns=np.array(scene.state.joint_indices['left'])+(0 if fixed else 6)
        assert raw.shape[-1]==len(scene.state.dof_paths)+(0 if fixed else 6)
        jac=np.zeros((len(paths),6,8),np.float64)
        jac[1 if fixed else 0:]=raw[:,:,columns]
        data=scene.articulation.data
        anchors=(data.body_link_pos_w if self.anchor=='link' else data.body_com_pos_w)[0].cpu().numpy().astype(np.float64)
        return jac,anchors

    def read(self,with_gradients=True):
        monitor=self.scene.monitor;by_pair={};total=0
        if with_gradients:jac,anchors=self.body_jacobians()
        def point_jac(path,point):
            i=self.scene.state.body_indices.get(path)
            if i is None:return np.zeros((3,8))
            return jac[i,:3]+np.cross(jac[i,3:].T,point-anchors[i]).T
        for view in monitor._tensor_views:
            _,points,normals,separation,counts,starts=view.get_contact_data(dt=1.)
            points=points.cpu().numpy();normals=normals.cpu().numpy()
            separation=separation.reshape(-1).cpu().numpy()
            counts=counts.reshape(view.sensor_count,view.filter_count).cpu().numpy()
            starts=starts.reshape(view.sensor_count,view.filter_count).cpu().numpy()
            # Count overflow invalidates the query, never implies a safe result.
            if int(counts.sum())>=len(separation):raise RuntimeError('contact_buffer_may_be_full')
            total+=int(counts.sum())
            sensors=list(map(str,view.sensor_paths));filters=monitor._tensor_filter_paths[id(view)]
            for i,j in zip(*np.nonzero(counts)):
                a,b=sensors[i],filters[j]
                kind=monitor._kind(a,b)
                if a==b or kind not in ('left_self','left_environment','inter_arm'):continue
                begin,count=int(starts[i,j]),int(counts[i,j])
                patch=separation[begin:begin+count]
                if begin<0 or len(patch)!=count or not np.isfinite(patch).all():raise RuntimeError('invalid_contact_patch')
                k=begin+int(np.argmin(patch));d=float(separation[k]);key=tuple(sorted((a,b)))
                if key in by_pair and by_pair[key]['distance']<=d:continue
                normal=normals[k].astype(np.float64);point=points[k].astype(np.float64)
                if not np.isfinite(point).all() or abs(np.linalg.norm(normal)-1)>.01:raise RuntimeError('invalid_contact_geometry')
                row={'pair':key,'kind':kind,'distance':d,'point':point,'normal':normal,'sensor':a,'filter':b}
                if with_gradients:row['gradient']=normal@(point_jac(a,point)-point_jac(b,point))
                by_pair[key]=row
        return list(by_pair.values()),{'contacts':total,'pairs':len(by_pair)}

    def oriented(self):
        """Experimental articulation-order convention, gated by FD scene audit.

        Not an API-wide normal-order contract. External rigid-body pairs need
        finite differences; never guess their ordering from path names.
        """
        rows,stats=self.read()
        for row in rows:
            a=self.scene.state.body_indices.get(row['sensor'])
            b=self.scene.state.body_indices.get(row['filter'])
            if a is None or b is None:
                row['gradient']=None
            else:
                row['gradient']=row['gradient']*(1 if a>b else -1)
        return rows,stats
