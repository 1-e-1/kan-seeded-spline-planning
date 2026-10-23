"""Scene adapter shared by baseline and candidate, not algorithmic repair.

Fixed robot links remain in self collision (moving-vs-fixed). Their world
constraints are constants, checked with exact original filters before planning.
Native sphere-world queries cannot express the USD a1/C_structure exclusion,
so fixed-vs-fixed terms must not be injected into the optimization problem.
"""
from dataclasses import replace
from types import MethodType
import torch


def eliminate_constant_world_terms(planner, model, names):
    kin=planner.trajopt_solver.kinematics.kinematics_config
    # Obtain the native link-to-sphere ordering, never infer it from JSON order.
    fixed=[]
    for path,name in names.items():
        if model.depth[model.index[path]]==0:
            idx=kin.link_name_to_idx_map[name]
            fixed.extend(torch.nonzero(kin.link_sphere_idx_map==idx).flatten().tolist())
    indices=torch.tensor(fixed,device='cuda',dtype=torch.long)
    scale=torch.ones((len(kin.link_sphere_idx_map),4),device='cuda')
    offset=torch.zeros_like(scale)
    scale[indices,3]=0.;offset[indices,3]=-1.
    checker=planner.scene_collision_checker
    for method in ('get_sphere_distance','get_sphere_collision','get_swept_sphere_distance','get_swept_sphere_collision'):
        original=getattr(checker,method)
        def wrapped(self,state,*args,_original=original,**kwargs):
            spheres=state.robot_spheres*scale+offset
            return _original(replace(state,robot_spheres=spheres),*args,**kwargs)
        setattr(checker,method,MethodType(wrapped,checker))
    return dict(fixed_world_sphere_indices=fixed,
                semantics='constant fixed-vs-fixed constraints eliminated only from optimizer; exact original-filter endpoint/trajectory gate retains them')
