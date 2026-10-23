"""Configuration-independent sphere reach bound for this exported axis chain."""
import numpy as np


def sphere_reach_bound(model,spheres,names):
    axes=np.cross(model.screws[:,3:],model.screws[:,:3])
    for i in range(8):
        anchor=model.home[model.index[f'/World/Base_01/zbot_8s_new_01/b{i+1}'],:3,3]
        w=model.screws[i,3:];axes[i]+=w*(w@anchor)
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(axes,axis=0),axis=1))]
    bound=0.
    for path,name in names.items():
        index=model.index[path];depth=model.depth[index];T=model.home[index]
        for sphere in spheres[name]:
            center=T[:3,:3]@sphere['center']+T[:3,3]
            distance=np.linalg.norm(center-axes[0]) if depth==0 else length[depth-1]+np.linalg.norm(center-axes[depth-1])
            bound=max(bound,distance+sphere['radius'])
    return axes[0],float(bound)
