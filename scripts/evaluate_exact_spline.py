"""Development evaluation of a single exact-geometry spline optimizer."""
from pathlib import Path
import sys,json,argparse,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages')]
import numpy as np
import torch
from dataclasses import asdict
from src.geometry.portable_scene import PortableScene,FCLSceneDistance
from src.planning.exact_spline import ExactSpline,SplineConfig
from src.planning.smooth_execution import smoothness
from src.planning.kan_seed import KANSeed
from src.models.trajectory_model import TrajectoryModel,TrajectoryModelConfig

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--indices',default='15,21,47,94,92');p.add_argument('--method',choices=['linear','kan'],default='linear')
p.add_argument('--iterations',type=int,default=100);p.add_argument('--controls',type=int,default=12)
p.add_argument('--clearance',type=float,default=.002);p.add_argument('--collision-weight',type=float,default=1e6)
p.add_argument('--solver',choices=['lbfgs','slsqp'],default='lbfgs')
args=p.parse_args()


def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists(),'Use a new output directory'
    torch.set_num_threads(4);torch.manual_seed(20260912)
    geometry=PortableScene(ROOT/'results/sota_simple_v1/scene_model.json')
    assert geometry.meta['fk_gate'] and geometry.meta['scene_sha256']==hashlib.sha256((ROOT/'finally.usd').read_bytes()).hexdigest()
    sensor=FCLSceneDistance(geometry);config=SplineConfig(controls=args.controls,iterations=args.iterations,clearance=args.clearance,collision_weight=args.collision_weight,solver=args.solver)
    opt=ExactSpline(sensor,config)
    tasks=json.loads((ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json').read_text())['tasks']
    indices=list(map(int,args.indices.split(',')))
    protocol=dict(method=args.method,config=asdict(config),indices=indices,scene_sha256=geometry.meta['scene_sha256'],
                  heldout=False,online_validated=False,scope='static planning development; exact sampled collision acceptance; not closed-loop execution',
                  endpoint_tolerances=dict(q_rad=.01,tcp_m=.005,rotation_rad=.02),constant_fixed_constraints='included with original USD exclusions')
    generator=None
    if args.method=='kan':
        file=ROOT/'checkpoints/left/left_all_kan_h128_astronaut_v1_best.pt';ckpt=torch.load(file,map_location='cuda',weights_only=False)
        actor=TrajectoryModel(TrajectoryModelConfig(**ckpt['config'])).to('cuda');actor.load_compatible_state_dict(ckpt['model']);actor.eval();actor.requires_grad_(False)
        generator=KANSeed(actor,geometry);protocol['actor_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    save(args.output/'protocol.json',protocol);rows=[]
    for index in indices:
        begun=time.perf_counter();spec=tasks[index]['task']['left'];start=np.array(spec['q_start']);goal=start+(np.array(spec['q_goal'])-start+np.pi)%(2*np.pi)-np.pi
        row=dict(index=index,difficulty=tasks[index]['difficulty'],success=False)
        mins=[]
        for q in (start,goal):
            sensor.query(q);mins.append(min((r['distance'] for r in sensor.last_rows),default=float('inf')))
        row['endpoint_minimum_m']=mins
        if min(mins)<-1e-7:row['reason']='invalid_start' if mins[0]<-1e-7 else 'invalid_goal'
        else:
            prior=None
            if generator:
                prior,prior_info=generator.generate(start,goal);prior+=start-prior[0]
                prior+=np.linspace(0,1,len(prior))[:,None]*(goal-prior[-1]);row['kan_seed']=prior_info
                np.savez_compressed(args.output/f'proposal_{index:04d}.npz',q=prior)
            data,info=opt.solve(start,goal,prior);row['optimizer']=info
            np.savez_compressed(args.output/f'trajectory_{index:04d}.npz',**data)
            row['smoothness']=smoothness(data['q'],data['dt'])
            # Evaluate the actual spline, not chords between sampled positions.
            from scipy.interpolate import BSpline
            curve=BSpline(data['knots'],data['controls'],5)
            phase=np.linspace(0,1,max(int(np.ceil(data['duration']*120)),int(np.ceil(np.max(abs(data['velocity']))*data['duration']/.01)))+1)
            hit=None;minimum=float('inf');checks=0
            for t,q in zip(phase,curve(phase)):
                sensor.query(q);checks+=1;closest=min(sensor.last_rows,key=lambda r:r['distance'],default=None)
                if closest:
                    minimum=min(minimum,closest['distance'])
                    if closest['distance'] < -1e-7:
                        hit=dict(phase=float(t),pair=closest['pair'],distance_m=closest['distance']);break
            terminal=float(np.max(abs(data['q'][-1]-goal)))
            stopped=bool(np.max(abs(data['velocity'][[0,-1]]))<.001 and np.max(abs(data['acceleration'][[0,-1]]))<.01)
            row.update(success=hit is None and terminal<.01 and stopped,collision=hit,exact_minimum_m=minimum,exact_checks=checks,
                       terminal_joint_error=terminal,stopped=stopped)
            row['reason']='success' if row['success'] else 'exact_collision' if hit else 'terminal_error'
        row['wall_s']=time.perf_counter()-begun
        rows.append(row);save(args.output/f'task_{index:04d}.json',row)
        save(args.output/'status.json',dict(status='complete' if len(rows)==len(indices) else 'running',rows=rows,successes=sum(r['success'] for r in rows),completed=len(rows),total=len(indices)))
        print('EXACT_SPLINE',index,row['reason'],row['wall_s'],row.get('optimizer'),flush=True)


if __name__=='__main__':main()
