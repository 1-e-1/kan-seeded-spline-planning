"""Strong-baseline development pilot. Not a SOTA performance claim.

Native cuRoboV2 graph seeding and spline optimization, independently accepted
against exact FCL hulls at dt <= 1/120 and joint step <= .01 rad.
"""
from pathlib import Path
import shutil
import sys, json, argparse, time, traceback, hashlib, faulthandler
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np
import torch
import yaml
import warp as wp
wp.config.kernel_cache_dir = str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir = str(ROOT/'results/sota_simple_v1/curobo_cache')
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState
from src.geometry.portable_scene import PortableScene,FCLSceneDistance
from src.planning.smooth_execution import smoothness
from src.planning.curobo_adapter import eliminate_constant_world_terms
from src.planning.native_spline_sampling import sample_native
from src.geometry.kinematic_jets import tcp_jets, derivative_metrics
from src.planning.benchmark_acceptance import kinematic_gate,finish_trial_row
from src.planning.benchmark_gpu_lock import BenchmarkGPULock
from src.planning.cpu_collision_validation import validate_trajectory,validate_archive_subprocess

p = argparse.ArgumentParser()
p.add_argument('--indices',default='15,21,47,94,92')
p.add_argument('--manifest',type=Path,default=ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json')
p.add_argument('--output',type=Path,default=ROOT/'results/sota_simple_v1/curobo_diagnostic5')
p.add_argument('--assets',type=Path,default=ROOT/'results/sota_simple_v1/curobo_scene')
p.add_argument('--attempts',type=int,default=5)
p.add_argument('--seeds',type=int,default=4)
p.add_argument('--no-graph',action='store_true')
p.add_argument('--method',choices=['native','kan','linear'],default='native')
p.add_argument('--degree',type=int,choices=[3,4,5],default=3)
p.add_argument('--jerk-weight',type=float,default=5.)
p.add_argument('--acceleration-weight',type=float,default=10000.)
p.add_argument('--velocity-weight',type=float,default=1000.)
p.add_argument('--kan-steps',type=int,default=180)
p.add_argument('--world-file',default='world.json')
p.add_argument('--knots',type=int,default=16)
p.add_argument('--seed-noise',type=float,default=0.)
p.add_argument('--self-margin',type=float,default=0.,help='Additional per-link self-collision sphere padding in m, uniform across tasks')
p.add_argument('--collision-weight-scale',type=float,default=1.)
p.add_argument('--optimizer-iterations',type=int,default=100)
p.add_argument('--esdf-directory',type=Path)
p.add_argument('--seed-parameterization',choices=['virtual_time','arclength'],default='virtual_time')
p.add_argument('--replace-kan-with-random',action='store_true',help='Ablation: replace KAN seed slot with an additional random seed, not a duplicate straight seed')
p.add_argument('--tcp-acceleration-weight',type=float,default=0.)
p.add_argument('--normalized-smoothness-weight',type=float,default=0.)
p.add_argument('--normalized-jerk-ratio',type=float,default=.002)
p.add_argument('--normalized-tcp-ratio',type=float,default=0.)
p.add_argument('--normalized-tcp-jerk-ratio',type=float,default=0.)
p.add_argument('--graph-policy',action='store_true')
p.add_argument('--planner-budget-s',type=float,default=0.,help='Fixed hot planning deadline; 0 for untimed development. Verification measured separately.')
p.add_argument('--warmup-tasks',type=int,default=0,help='Repeat development task 15 before evaluation; excluded and separately saved')
p.add_argument('--warmup-indices',default='15',help='Development tasks in each warmup cycle; confirmation must freeze this list')
p.add_argument('--validation-workers',type=int,default=0,help='CPU-only independent check processes; GPU planning remains serial. 0 means inline.')
p.add_argument('--freeze-file',type=Path,help='Required immutable decision for confirmation manifests')
p.add_argument('--seed-encoder',type=Path,help='Audited fixed geometric-samples to native-knots map; initialization only')
args = p.parse_args()


def save(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    torch.set_num_threads(4);torch.manual_seed(20260912);np.random.seed(20260912)
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists(), 'Use a fresh output; do not overwrite a trial'
    # Diagnostic only: a blocked CUDA copy must not be mistaken for an
    # algorithmic planning failure. The dump does not cancel or repair a solve.
    stack_log=(args.output/'long_operation_stacks.txt').open('w',encoding='utf-8')
    faulthandler.enable(file=stack_log)
    sources=['scripts/evaluate_curobo_development.py','src/planning/native_spline_sampling.py',
             'src/planning/curobo_adapter.py','src/planning/kan_seed.py','src/geometry/portable_scene.py',
             'src/geometry/kinematic_jets.py','src/geometry/sphere_reach.py',
             'src/planning/tcp_spline_objective.py',
             'src/planning/normalized_spline_objective.py',
             'src/planning/spline_seed_encoding.py',
             'src/planning/graph_policy.py',
             'src/planning/benchmark_acceptance.py',
             'src/planning/benchmark_gpu_lock.py',
             'src/planning/cpu_collision_validation.py','scripts/cpu_collision_worker.py',
             'external/curobo_v0_8_0/curobo/_src/geom/data/data_mesh.py',
             'external/curobo_v0_8_0/curobo/_src/geom/data/data_voxel.py',
             'external/curobo_v0_8_0/curobo/_src/geom/types.py']
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    for name in sources:
        target=args.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    build=json.loads((args.assets/'build_report.json').read_text())
    assert build['fk_gate']
    if 'union' in args.world_file:
        assert json.loads((args.assets/'union_report.json').read_text())['gate']
        assert json.loads((args.assets/'union_gpu_audit.json').read_text())['gate'], 'Union GPU geometry audit failed'
    model=PortableScene(ROOT/'results/sota_simple_v1/scene_model.json')
    assert model.meta['scene_sha256']==hashlib.sha256((ROOT/'finally.usd').read_bytes()).hexdigest()
    sensor=FCLSceneDistance(model)
    manifest=json.loads(args.manifest.read_text())
    indices=list(range(len(manifest['tasks']))) if args.indices=='all' else [int(x) for x in args.indices.split(',')]
    assert len(indices)==len(set(indices)) and all(0<=i<len(manifest['tasks']) for i in indices)
    is_holdout=manifest['protocol'].get('purpose')=='frozen_confirmation_not_for_tuning'
    if is_holdout:assert manifest['protocol']['scene_sha256']==model.meta['scene_sha256']
    optim=yaml.safe_load((ROOT/'external/curobo_v0_8_0/curobo/content/configs/task/trajopt/lbfgs_bspline_trajopt.yml').read_text())
    # Only calibrated kinematic quantities may affect this benchmark. No masses
    # were exported, so native default energy regularization must be disabled.
    optim['rollout']['cost_cfg']['cspace_cfg']['squared_l2_regularization_weight'][4]=0.
    optim['rollout']['cost_cfg']['cspace_cfg']['squared_l2_regularization_weight'][2]=args.jerk_weight
    optim['rollout']['cost_cfg']['cspace_cfg']['squared_l2_regularization_weight'][1]=args.acceleration_weight
    optim['rollout']['cost_cfg']['cspace_cfg']['squared_l2_regularization_weight'][0]=args.velocity_weight
    if args.normalized_smoothness_weight:
        assert args.normalized_smoothness_weight>0 and args.tcp_acceleration_weight==0
        optim['rollout']['cost_cfg']['cspace_cfg']['squared_l2_regularization_weight'][:3]=[0.,0.,0.]
    for field in ('scene_collision_cfg','self_collision_cfg'):
        optim['rollout']['constraint_cfg'][field]['weight']*=args.collision_weight_scale
    optim['optimizer']['num_iters']=args.optimizer_iterations
    transition=yaml.safe_load((ROOT/'external/curobo_v0_8_0/curobo/content/configs/task/trajopt/transition_bspline_trajopt.yml').read_text())
    transition['transition_model_cfg']['control_space']=f'BSPLINE_{args.degree}'
    transition['transition_model_cfg']['n_knots']=args.knots
    world=json.loads((args.assets/args.world_file).read_text())
    mesh_hashes={name:hashlib.sha256(Path(value['file_path']).read_bytes()).hexdigest()
                 for name,value in world.get('mesh',{}).items() if value.get('file_path')}
    robot_path=(args.assets/'robot.json').resolve()
    encoder=None;encoder_sha=None
    if args.seed_encoder:
        encoder_report=json.loads(args.seed_encoder.with_suffix('.json').read_text())
        encoder_sha=hashlib.sha256(args.seed_encoder.read_bytes()).hexdigest()
        assert args.method!='native' and encoder_report['gate'] and encoder_report['matrix_sha256']==encoder_sha
        assert encoder_report['knots']==args.knots and encoder_report['degree']==args.degree
        with np.load(args.seed_encoder) as z:encoder={k:z[k].copy() for k in z.files}
    if args.self_margin:
        robot=json.loads(robot_path.read_text())
        padding=robot['robot_cfg']['kinematics']['self_collision_buffer']
        robot['robot_cfg']['kinematics']['self_collision_buffer']={k:v+args.self_margin for k,v in padding.items()}
        robot_path=(args.output/'robot_with_margin.json').resolve()
        save(robot_path,robot)
    esdf_info=None
    scene_argument=str((args.assets/args.world_file).resolve())
    if args.esdf_directory:
        from curobo._src.geom.types import SceneCfg,VoxelGrid
        from src.geometry.sphere_reach import sphere_reach_bound
        esdf_info=json.loads((args.esdf_directory/'config.json').read_text())
        audit=json.loads((args.esdf_directory/'audit.json').read_text())
        assert audit['gate'] and audit['field_sha256']==esdf_info['field_sha256']
        assert esdf_info['scene_sha256']==model.meta['scene_sha256']
        center,reach=sphere_reach_bound(model,json.loads(robot_path.read_text())['robot_cfg']['kinematics']['collision_spheres'],build['names'])
        assert np.max(abs(center-esdf_info['center_world_m']))<1e-6
        assert reach+2*esdf_info['spacing_m']<min(esdf_info['dims_m'])/2
        esdf_info['effective_robot_reach_bound_m']=reach
        assert hashlib.sha256((args.esdf_directory/'field.npy').read_bytes()).hexdigest()==esdf_info['field_sha256']
        features=torch.tensor(np.load(args.esdf_directory/'field.npy'),device='cuda',dtype=torch.float16)
        scene_argument=SceneCfg(voxel=[VoxelGrid(name='fixed_world',dims=esdf_info['dims_m'],
            pose=[*esdf_info['center_world_m'],1,0,0,0],voxel_size=esdf_info['spacing_m'],feature_tensor=features)])
    save(args.output/'protocol.json',dict(indices=indices,seeds=args.seeds,attempts=args.attempts,
        source_sha256=source_hashes,duration_preserving_native_sampling=True,
        baseline='cuRoboV2 v0.8.0 kinematic adaptation',scene_sha256=model.meta['scene_sha256'],
        native_mesh_source_sha256=hashlib.sha256((ROOT/'external/curobo_v0_8_0/curobo/_src/geom/data/data_mesh.py').read_bytes()).hexdigest(),
        torch_threads=4,warp_version=wp.__version__,asset_sha256=hashlib.sha256((args.assets/'robot.json').read_bytes()).hexdigest(),
        world_file=args.world_file,world_sha256=hashlib.sha256((args.assets/args.world_file).read_bytes()).hexdigest(),
        referenced_mesh_sha256=mesh_hashes,
        effective_world_representation='static_ESDF' if esdf_info else 'mesh',esdf=esdf_info,
        native_voxel_source_sha256=hashlib.sha256((ROOT/'external/curobo_v0_8_0/curobo/_src/geom/data/data_voxel.py').read_bytes()).hexdigest(),
        graph_enabled=args.method=='native',cuda_graph_enabled=not args.no_graph,optimizer=optim,transition=transition,method=args.method,
        seed_noise=args.seed_noise,
        seed_parameterization=args.seed_parameterization,
        seed_encoder_sha256=encoder_sha,
        replace_kan_with_random=args.replace_kan_with_random,
        tcp_acceleration_weight=args.tcp_acceleration_weight,
        normalized_smoothness_weight=args.normalized_smoothness_weight,normalized_jerk_ratio=args.normalized_jerk_ratio,normalized_tcp_ratio=args.normalized_tcp_ratio,
        normalized_tcp_jerk_ratio=args.normalized_tcp_jerk_ratio,
        graph_policy=args.graph_policy,
        gpu_benchmark_serialization=True,
        cpu_validation_workers=args.validation_workers,
        planner_budget_s=args.planner_budget_s,warmup_tasks=args.warmup_tasks,warmup_indices=args.warmup_indices,
        kinematic_gate=dict(position_limit_rad=2*np.pi,velocity_limit_rad_s=1.5,acceleration_limit_rad_s2=4.,jerk_limit_rad_s3=30.,absolute_tolerance=1e-5),
        self_margin_per_link_m=args.self_margin,effective_robot_sha256=hashlib.sha256(robot_path.read_bytes()).hexdigest(),
        collision_acceptance='FCL original cooked hulls on actual native B-spline, dt <= 1/120 and max velocity*dt <= .01 rad; strict penetration < -1e-7 m; sampled, not continuous certificate',
        goal_joint_max_rad=.01,goal_tcp_m=.005,goal_orientation_rad=.02,heldout=is_holdout,
        manifest=str(args.manifest),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        note='All requested tasks retained including invalid endpoints. Native internal sphere model is approximate. Native energy term disabled because inertia is not calibrated.'))
    if is_holdout:
        assert args.freeze_file is not None,'Confirmation requires a frozen decision, not ad-hoc arguments'
        frozen=json.loads(args.freeze_file.read_text())
        assert frozen['status']=='frozen_before_confirmation'
        manifest_sha=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        assert manifest_sha in frozen['allowed_confirmation_manifests']
        assert args.indices=='all','Do not select favorable subsets from confirmation'
        expected=frozen['methods'][args.method]
        actual=json.loads((args.output/'protocol.json').read_text())
        assert all(actual.get(k)==v for k,v in expected['protocol'].items()),'Frozen method configuration differs'
        if args.method=='kan':assert args.kan_steps==expected['actor_max_steps']
        for name,sha in frozen['files_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,f'Frozen source/artifact changed: {name}'
        save(args.output/'confirmation_decision.json',dict(file=str(args.freeze_file),sha256=hashlib.sha256(args.freeze_file.read_bytes()).hexdigest()))
    config=MotionPlannerCfg.create(robot=str(robot_path),scene_model=scene_argument,
        num_trajopt_seeds=args.seeds,trajopt_optimizer_configs=[optim],trajopt_transition_model=transition,use_cuda_graph=not args.no_graph,random_seed=20260912)
    print('CREATING_PLANNER',flush=True)
    planner=MotionPlanner(config)
    assert planner.trajopt_solver.transition_model.config.robot_config.dynamics is None,'Kinematic benchmark must not use uncalibrated dynamics'
    save(args.output/'constant_constraints.json',eliminate_constant_world_terms(planner,model,build['names']))
    if args.tcp_acceleration_weight:
        from src.planning.tcp_spline_objective import add_tcp_acceleration_objective
        save(args.output/'tcp_objective.json',add_tcp_acceleration_objective(planner,args.tcp_acceleration_weight))
    if args.normalized_smoothness_weight:
        from src.planning.normalized_spline_objective import add_normalized_smoothness
        save(args.output/'normalized_objective.json',add_normalized_smoothness(planner,args.normalized_smoothness_weight,args.normalized_jerk_ratio,args.normalized_tcp_ratio,args.normalized_tcp_jerk_ratio))
    max_radius=max(s['radius'] for hs in json.loads((args.assets/'robot.json').read_text())['robot_cfg']['kinematics']['collision_spheres'].values() for s in hs)
    if esdf_info:
        assert esdf_info['clipping_m']>max_radius+.02
    else:
        assert planner.scene_collision_checker.data.meshes.max_dist > max_radius+.02
    print('PLANNER_READY',flush=True)
    generator=None
    if args.method=='kan':
        from src.models.trajectory_model import TrajectoryModel,TrajectoryModelConfig
        from src.planning.kan_seed import KANSeed
        checkpoint=ROOT/'checkpoints/left/left_all_kan_h128_astronaut_v1_best.pt'
        ckpt=torch.load(checkpoint,map_location='cuda',weights_only=False)
        actor=TrajectoryModel(TrajectoryModelConfig(**ckpt['config'])).to('cuda')
        actor.load_compatible_state_dict(ckpt['model']);actor.eval();actor.requires_grad_(False)
        if args.graph_policy:
            from src.planning.graph_policy import GraphPolicy
            audit=json.loads((ROOT/'results/sota_simple_v1/graph_policy_audit.json').read_text())
            assert audit['gate'] and audit['actor_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            actor=GraphPolicy(actor)
        generator=KANSeed(actor,model)
        save(args.output/'actor.json',dict(sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),frozen=True,seed_history='virtual kinematic only; not execution history',max_steps=args.kan_steps))
    rows=[];pending={}
    if args.validation_workers:
        assert args.validation_workers>0
        cpu_audit=json.loads((ROOT/'results/sota_simple_v1/cpu_collision_workers_audit.json').read_text())
        assert cpu_audit['gate'] and all(source_hashes[name]==sha for name,sha in cpu_audit['source_sha256'].items())
    pool=ThreadPoolExecutor(max_workers=args.validation_workers) if args.validation_workers else None
    def persist(row):
        row['end_to_end_wall_s']=time.perf_counter()-row.pop('_task_started')
        rows.append(row);save(args.output/f"task_{row['index']:04d}.json",row)
        save(args.output/'status.json',dict(status='complete' if len(rows)==len(indices) else 'running',rows=sorted(rows,key=lambda r:r['index']),
            successes=sum(r['success'] for r in rows),completed=len(rows),total=len(indices),pending_validation=len(pending)))
        print('CUROBO_TASK',row['index'],row['reason'],row.get('planner_wall_s'),flush=True)
    def drain(block=False):
        ready=as_completed(list(pending)) if block else [f for f in pending if f.done()]
        for future in ready:
            row=pending.pop(future)
            persist(finish_trial_row(row,future.result()))
    warmup_manifest=json.loads((ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json').read_text())
    warmup_indices=[int(x) for x in args.warmup_indices.split(',')]*args.warmup_tasks
    for ordinal,index in enumerate(warmup_indices+indices):
        faulthandler.dump_traceback_later(60,repeat=True,file=stack_log)
        is_warmup=ordinal<len(warmup_indices)
        file_prefix=f'warmup_{ordinal:02d}_' if is_warmup else ''
        task=(warmup_manifest if is_warmup else manifest)['tasks'][index];spec=task['task']['left']
        start=np.array(spec['q_start']);goal=start+(np.array(spec['q_goal'])-start+np.pi)%(2*np.pi)-np.pi
        row=dict(index=index,difficulty=task['difficulty'],success=False,reason='not_run',_task_started=time.perf_counter())
        scheduled=False
        mins=[]
        for q in (start,goal):
            sensor.query(q);mins.append(min((r['distance'] for r in sensor.last_rows),default=float('inf')))
        row['endpoint_minimum_m']=mins
        if min(mins)<-1e-7:row['reason']='invalid_start' if mins[0]<-1e-7 else 'invalid_goal'
        else:
            qs=JointState.from_position(torch.tensor(start-model.q_ref,device='cuda',dtype=torch.float32)[None],joint_names=planner.joint_names)
            qg=JointState.from_position(torch.tensor(goal-model.q_ref,device='cuda',dtype=torch.float32)[None],joint_names=planner.joint_names)
            t=time.perf_counter()
            if args.method=='native':
                result=planner.plan_cspace(qg,qs,max_attempts=args.attempts)
            else:
                seed=None
                if generator is not None:
                    path,info=generator.generate(start,goal,max_steps=args.kan_steps)
                    row['kan_seed']=info
                    # Shift the continuous proposal to the requested start branch,
                    # then distribute any full-turn endpoint mismatch smoothly.
                    path+=start-path[0]
                    path+=np.linspace(0,1,len(path))[:,None]*(goal-path[-1])
                    horizon=planner.trajopt_solver.action_horizon
                    parameter=np.linspace(0,1,len(path))
                    if args.seed_parameterization=='arclength':
                        parameter=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
                        parameter/=max(parameter[-1],1e-12)
                        keep=np.r_[True,np.diff(parameter)>1e-10];path=path[keep];parameter=parameter[keep]
                    sampled=np.column_stack([np.interp(np.linspace(0,1,horizon),parameter,path[:,j]) for j in range(8)])
                    seed=torch.tensor(sampled-model.q_ref,device='cuda',dtype=torch.float32)[None,None]
                    np.savez_compressed(args.output/f'{file_prefix}seed_{index:04d}.npz',q=path,control_seed=sampled)
                if args.seed_noise>0:
                    assert args.seeds>=2
                    horizon=planner.trajopt_solver.action_horizon
                    phase=np.linspace(0,1,horizon)
                    straight=start[None]+phase[:,None]*(goal-start)[None]
                    proposed=sampled if generator is not None else straight
                    seed_paths=[proposed,straight]
                    rng=np.random.default_rng(20260912+index)
                    for _ in range(args.seeds-2):
                        coefficients=rng.normal(size=(2,8))*args.seed_noise
                        perturbation=np.sin(np.pi*phase)[:,None]*coefficients[0]+.5*np.sin(2*np.pi*phase)[:,None]*coefficients[1]
                        seed_paths.append(straight+perturbation)
                    if args.replace_kan_with_random:
                        assert args.method=='linear'
                        coefficients=rng.normal(size=(2,8))*args.seed_noise
                        seed_paths[0]=straight+np.sin(np.pi*phase)[:,None]*coefficients[0]+.5*np.sin(2*np.pi*phase)[:,None]*coefficients[1]
                    seed=torch.tensor(np.clip(np.array(seed_paths)-model.q_ref,-2*np.pi,2*np.pi),device='cuda',dtype=torch.float32)[None]
                # A single multistart spline optimization. No PRM/RRT fallback.
                # Remaining seeds are native straight interpolations (ablation).
                if encoder is not None:
                    from src.planning.spline_seed_encoding import encode_seed
                    assert seed is not None
                    geometric=seed[0].detach().cpu().numpy()
                    encoded=np.clip(encode_seed(geometric,start-model.q_ref,goal-model.q_ref,encoder),-2*np.pi,2*np.pi)
                    seed=torch.tensor(encoded,device='cuda',dtype=torch.float32)[None]
                    np.savez_compressed(args.output/f'{file_prefix}encoded_seed_{index:04d}.npz',geometric=geometric,knots=encoded)
                result=planner.trajopt_solver.solve_cspace(qg,qs,seed_traj=seed)
            torch.cuda.synchronize()
            row['planner_wall_s']=time.perf_counter()-t
            row['planner_budget_ok']=bool(args.planner_budget_s<=0 or row['planner_wall_s']<=args.planner_budget_s)
            row['native_success']=bool(result is not None and result.success.any())
            if result is None:row['reason']='native_no_result'
            else:
                plan=result.get_interpolated_plan()
                q=plan.position.detach().cpu().numpy().reshape(-1,8)+model.q_ref
                dt=float(planner.trajopt_solver.config.interpolation_dt)
                arrays=dict(q=q,dt=dt)
                if result.js_solution.knot is not None:
                    arrays.update(native_knots=result.js_solution.knot.detach().cpu().numpy(),
                                  native_knot_dt=result.js_solution.knot_dt.detach().cpu().numpy(),native_degree=args.degree,
                                  native_start=start-model.q_ref,native_goal=goal-model.q_ref)
                for field in ('velocity','acceleration','jerk'):
                    value=getattr(plan,field,None)
                    if value is not None:arrays[field]=value.detach().cpu().numpy().reshape(-1,8)
                assert 'native_knots' in arrays, 'Exact native curve required for benchmark'
                substeps=max(3,int(np.ceil(np.max(abs(arrays['velocity']))*dt/.009)))
                while True:
                    dense_dt=dt/substeps
                    dense=sample_native(arrays['native_knots'],arrays['native_knot_dt'],args.degree,
                                        arrays['native_start'],arrays['native_goal'],dense_dt,duration=(len(q)-1)*dt)
                    if np.max(abs(dense['velocity']))*dense_dt <= .01:break
                    substeps*=2
                dense_q=dense['position']+model.q_ref
                arrays.update(dense_dt=dense_dt,dense_q=dense_q,
                              **{f'dense_{k}':dense[k] for k in ('velocity','acceleration','jerk')})
                duration=(len(dense_q)-1)*dense_dt
                analytic=dict(duration_s=duration)
                for field in ('velocity','acceleration','jerk'):
                    integral=float(np.trapz(np.sum(dense[field].astype(float)**2,axis=1),dx=dense_dt))
                    analytic.update({field+'_energy':integral,field+'_rms':float(np.sqrt(integral/duration/8)),
                                     field+'_max':float(np.max(abs(dense[field])))})
                row['analytic_smoothness']=analytic
                tcp=tcp_jets(model,dense_q,dense['velocity'],dense['acceleration'],dense['jerk'])
                row['tcp_analytic_smoothness']={name:derivative_metrics(tcp[k],dense_dt,vector_norm=True)
                                               for k,name in enumerate(('position','velocity','acceleration','jerk')) if k}
                arrays['tcp_jets']=tcp
                row['kinematic_gate']=kinematic_gate(dense['position'],dense['velocity'],dense['acceleration'],dense['jerk'])
                np.savez_compressed(args.output/f'{file_prefix}trajectory_{index:04d}.npz',**arrays)
                row.update(samples=len(q),dt=dt,smoothness=smoothness(q,dt),terminal_joint_error=float(np.max(abs((q[-1]-goal+np.pi)%(2*np.pi)-np.pi))))
                Ta=model.forward(q[-1])[0][model.index[model.meta['tool_path']]]
                Tg=model.forward(goal)[0][model.index[model.meta['tool_path']]]
                row['terminal_tcp_m']=float(np.linalg.norm(Ta[:3,3]-Tg[:3,3]))
                row['terminal_orientation_rad']=float(np.arccos(np.clip((np.trace(Ta[:3,:3].T@Tg[:3,:3])-1)/2,-1,1)))
                stopped=('velocity' in arrays and np.max(abs(arrays['velocity'][[0,-1]]))<.001 and
                         'acceleration' in arrays and np.max(abs(arrays['acceleration'][[0,-1]]))<.01)
                row['stopped']=bool(stopped)
                if pool is not None and not is_warmup:
                    future=pool.submit(validate_archive_subprocess,sys.executable,ROOT/'scripts/cpu_collision_worker.py',
                        ROOT/'results/sota_simple_v1/scene_model.json',args.output/f'trajectory_{index:04d}.npz')
                    pending[future]=row;scheduled=True
                    print('CUROBO_PLANNED',index,row['planner_wall_s'],'verification_pending',flush=True)
                else:
                    row=finish_trial_row(row,validate_trajectory(sensor,dense_q,dense_dt))
        if is_warmup:
            faulthandler.cancel_dump_traceback_later()
            row['end_to_end_wall_s']=time.perf_counter()-row.pop('_task_started')
            save(args.output/f'{file_prefix}task.json',row)
            print('CUROBO_WARMUP',ordinal,row['reason'],row.get('planner_wall_s'),flush=True)
            continue
        if not scheduled:persist(row)
        drain()
        faulthandler.cancel_dump_traceback_later()
    drain(block=True)
    if pool is not None:pool.shutdown(wait=True)


if __name__=='__main__':
    try:
        with BenchmarkGPULock(ROOT/'results/sota_simple_v1/benchmark_gpu.lock'):main()
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        save(args.output/'exception.json',dict(status='exception',traceback=traceback.format_exc()))
        raise
