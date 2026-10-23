"""Reproduce tiny-mesh false collision and mesh cost-gradient sign errors."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np
import torch,trimesh,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
from curobo._src.geom.collision import SceneCollision,SceneCollisionCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.types import SceneCfg
from curobo.types import DeviceCfg
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--winding-number',action='store_true');args=p.parse_args()
device=DeviceCfg();rows=[]
for name,size,center,radius in [('tiny_far',.002,.2,.02),('outside_near',.1,.055,.02),('inside_near',.1,.04,.02)]:
    mesh=trimesh.creation.box(extents=[size]*3)
    scene=SceneCollision.from_config(SceneCollisionCfg(scene_model=SceneCfg.create(dict(mesh=dict(test=dict(vertices=mesh.vertices.tolist(),faces=mesh.faces.tolist(),pose=[0,0,0,1,0,0,0],winding_number=args.winding_number))))))
    buffer=CollisionBuffer.from_shape(torch.Size([1,1,1,4]),device)
    def query(x,grad=False):
        spheres=torch.tensor([[[[x,0.,0.,radius]]]],device='cuda',requires_grad=grad)
        cost=scene.get_sphere_distance_raw(spheres,buffer,torch.ones(1,device='cuda'),torch.zeros(1,device='cuda'),return_loss=True)
        value=float(cost.sum())
        g=float(torch.autograd.grad(cost.sum(),spheres)[0].flatten()[0]) if grad else None
        return value,g
    value,g=query(center,True)
    fd=(query(center+1e-5)[0]-query(center-1e-5)[0])/2e-5
    expected=max(0.,radius-(center-size/2))
    rows.append(dict(name=name,cost=value,expected_cost=expected,gradient=g,fd=fd,passed=abs(value-expected)<1e-5 and abs(g-fd)<.002))
report=dict(rows=rows,gate=all(r['passed'] for r in rows),winding_number=args.winding_number)
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
