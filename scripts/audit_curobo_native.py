"""Locate failed native endpoint constraints before scoring a baseline."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import torch,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir=str(ROOT/'results/sota_simple_v1/curobo_cache')
from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
from src.planning.curobo_adapter import eliminate_constant_world_terms
from src.geometry.portable_scene import PortableScene
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=ROOT/'results/sota_simple_v1/curobo_scene')
p.add_argument('--world-file',default='world.json');args=p.parse_args()
assets=args.assets.resolve()
planner=MotionPlanner(MotionPlannerCfg.create(robot=str(assets/'robot.json'),scene_model=str(assets/args.world_file),use_cuda_graph=False))
eliminate_constant_world_terms(planner,PortableScene(ROOT/'results/sota_simple_v1/scene_model.json'),json.loads((assets/'build_report.json').read_text())['names'])
tasks=json.loads((ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json').read_text())['tasks']
rows=[]
for index in (15,21,47,94,92):
    for endpoint in ('q_start','q_goal'):
        q=torch.tensor(tasks[index]['task']['left'][endpoint],device='cuda',dtype=torch.float32)
        rollout=planner.graph_planner.feasibility_rollout
        metrics=rollout.compute_metrics_from_action(q[None,None])
        cc=metrics.costs_and_constraints
        values={}
        for category in ('constraints','hybrid_costs_constraints','costs'):
            coll=getattr(cc,category,None)
            if coll is not None:
                values[category]={n:dict(max=float(v.max()),sum=float(v.sum()),shape=list(v.shape)) for n,v in zip(coll.names,coll.values)}
        row=dict(index=index,endpoint=endpoint,metrics=values,available=list(vars(cc)))
        if index==15 and endpoint=='q_start':
            data=planner.scene_collision_checker.data.meshes
            original=data.enable.clone();isolated=[]
            for mi,name in enumerate(data.names[0]):
                if name is None:continue
                data.enable.zero_();data.enable[0,mi]=1
                mc=rollout.compute_metrics_from_action(q[None,None]).costs_and_constraints.constraints
                value=mc.values[mc.names.index('scene_collision')]
                if value.max()>0:isolated.append(dict(mesh=name,max=float(value.max()),spheres=torch.nonzero(value.flatten()>0).flatten().tolist()))
            data.enable.copy_(original);row['isolated_mesh_collisions']=isolated
        rows.append(row);print('NATIVE_ENDPOINT',json.dumps(row),flush=True)
(assets/('native_endpoint_audit_'+Path(args.world_file).stem+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
