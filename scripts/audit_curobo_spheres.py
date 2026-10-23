"""Inspect sphere approximation false endpoint collisions independently."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages')]
import numpy as np
import fcl
from src.geometry.portable_scene import PortableScene,FCLSceneDistance
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=ROOT/'results/sota_simple_v1/curobo_scene')
p.add_argument('--indices',default='15,21,47,94,92');args=p.parse_args()
model=PortableScene(ROOT/'results/sota_simple_v1/scene_model.json');exact=FCLSceneDistance(model)
cfg=json.loads((args.assets/'robot.json').read_text())['robot_cfg']['kinematics']
build=json.loads((args.assets/'build_report.json').read_text());names=build['names']
tasks=json.loads((ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json').read_text())['tasks']
rows=[];req=fcl.DistanceRequest(enable_signed_distance=True)
for index in map(int,args.indices.split(',')):
    for endpoint in ('q_start','q_goal'):
        q=np.array(tasks[index]['task']['left'][endpoint]);T,_=model.forward(q);spheres={}
        for path,name in names.items():
            pos=T[model.index[path]]
            spheres[path]=[(pos[:3,:3]@np.array(s['center'])+pos[:3,3],s['radius']) for s in cfg['collision_spheres'][name]]
        exact.query(q);actual={tuple(r['pair']):r['distance'] for r in exact.last_rows}
        collisions=[]
        for a,b,kind in model.pairs:
            if a not in spheres and b not in spheres:continue
            best=float('inf')
            if a in spheres and b in spheres:
                for pa,ra in spheres[a]:
                    for pb,rb in spheres[b]:best=min(best,np.linalg.norm(pa-pb)-ra-rb)
            else:
                arm,static=(a,b) if a in spheres else (b,a)
                trans=T[model.index[static]]
                for v,obj in zip(model.hulls[static],exact.objects[static]):
                    vw=v@trans[:3,:3].T+trans[:3,3];lo=vw.min(0);hi=vw.max(0)
                    for center,radius in spheres[arm]:
                        lower=np.linalg.norm(np.maximum(np.maximum(lo-center,center-hi),0))-radius
                        if lower>min(best,0.):continue
                        sphere=fcl.CollisionObject(fcl.Sphere(radius),fcl.Transform(center))
                        best=min(best,fcl.distance(sphere,obj,req,fcl.DistanceResult()))
            if best<0:collisions.append(dict(pair=[a,b],kind=kind,sphere_distance_m=float(best),exact_distance_m=actual.get((a,b))))
        rows.append(dict(index=index,endpoint=endpoint,collisions=collisions))
        print('SPHERE_ENDPOINT',index,endpoint,collisions,flush=True)
(args.assets/'endpoint_sphere_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
