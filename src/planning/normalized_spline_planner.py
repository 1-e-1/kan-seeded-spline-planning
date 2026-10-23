"""Isolated-worker adapter for the frozen two-module candidate.

Not yet an accepted online controller. No USD edits, network updates, search
fallbacks, trajectory blending, or output retiming. A single native solver
invocation retains its documented internal time-optimization/refinement.
Import only after configuring the isolated cuRobo/Warp environment.
"""
from pathlib import Path
from contextlib import nullcontext
import json,hashlib,time
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
from curobo.types import JointState
from curobo._src.geom.types import SceneCfg,VoxelGrid
from src.geometry.portable_scene import PortableScene,FCLSceneDistance
from src.geometry.sphere_reach import sphere_reach_bound
from src.models.trajectory_model import TrajectoryModel,TrajectoryModelConfig
from src.planning.kan_seed import KANSeed
from src.planning.graph_policy import GraphPolicy
from src.planning.curobo_adapter import eliminate_constant_world_terms
from src.planning.normalized_spline_objective import add_normalized_smoothness
from src.planning.native_spline_sampling import sample_native
from src.planning.benchmark_acceptance import kinematic_gate
from src.planning.cpu_collision_validation import validate_trajectory
from src.planning.cubic_execution_curve import CubicExecutionCurve


class NormalizedSplinePlanner:
    def __init__(self,root,decision,contact_only_verification=False):
        self.root=Path(root);decision=Path(decision)
        frozen=json.loads(decision.read_text());assert frozen['status']=='frozen_before_confirmation'
        for name,sha in frozen['files_sha256'].items():
            assert hashlib.sha256((self.root/name).read_bytes()).hexdigest()==sha,f'Frozen artifact changed: {name}'
        self.spec=frozen['methods']['kan']['protocol']
        self.kan_steps=frozen['methods']['kan']['actor_max_steps']
        assert self.spec['seed_encoder_sha256'] is None and self.spec['self_margin_per_link_m']==0
        assert self.spec['normalized_smoothness_weight']>0 and self.spec['graph_policy']
        assert self.spec['transition']['transition_model_cfg']['control_space']=='BSPLINE_3'
        torch.set_num_threads(self.spec['torch_threads']);torch.manual_seed(20260912);np.random.seed(20260912)
        self.model=PortableScene(self.root/'results/sota_simple_v1/scene_model.json')
        self.sensor=FCLSceneDistance(self.model)
        self.contact_only_verification=bool(contact_only_verification)
        self.verification_pool=None
        self.verification_sensor=self.sensor
        if self.contact_only_verification:
            audit=json.loads((self.root/'results/sota_simple_v1/contact_only_verification_audit_v1.json').read_text())
            assert audit['gate'] and audit['contact_activation_m']==1e-6
            for name,key in [('src/geometry/portable_scene.py','portable_source_sha256'),
                             ('results/sota_simple_v1/scene_model.json','scene_sha256')]:
                assert hashlib.sha256((self.root/name).read_bytes()).hexdigest()==audit[key]
            self.verification_sensor=FCLSceneDistance(self.model,activation_distance=1e-6)
        assets=self.root/'results/sota_simple_v1/curobo_scene_fine16'
        field=self.root/'results/sota_simple_v1/static_esdf2mm'
        build=json.loads((assets/'build_report.json').read_text());assert build['fk_gate']
        robot_path=(assets/'robot.json').resolve();robot=json.loads(robot_path.read_text())
        assert hashlib.sha256(robot_path.read_bytes()).hexdigest()==self.spec['effective_robot_sha256']
        info=json.loads((field/'config.json').read_text());audit=json.loads((field/'audit.json').read_text())
        assert audit['gate'] and audit['field_sha256']==info['field_sha256']
        assert info['scene_sha256']==self.model.meta['scene_sha256']
        assert hashlib.sha256((field/'field.npy').read_bytes()).hexdigest()==info['field_sha256']
        center,reach=sphere_reach_bound(self.model,robot['robot_cfg']['kinematics']['collision_spheres'],build['names'])
        assert np.max(abs(center-info['center_world_m']))<1e-6 and reach+2*info['spacing_m']<min(info['dims_m'])/2
        features=torch.tensor(np.load(field/'field.npy'),device='cuda',dtype=torch.float16)
        world=SceneCfg(voxel=[VoxelGrid(name='fixed_world',dims=info['dims_m'],pose=[*info['center_world_m'],1,0,0,0],
                                     voxel_size=info['spacing_m'],feature_tensor=features)])
        config=MotionPlannerCfg.create(robot=str(robot_path),scene_model=world,
            num_trajopt_seeds=self.spec['seeds'],trajopt_optimizer_configs=[self.spec['optimizer']],
            trajopt_transition_model=self.spec['transition'],use_cuda_graph=self.spec['cuda_graph_enabled'],random_seed=20260912)
        self.planner=MotionPlanner(config)
        assert self.planner.trajopt_solver.transition_model.config.robot_config.dynamics is None
        eliminate_constant_world_terms(self.planner,self.model,build['names'])
        add_normalized_smoothness(self.planner,self.spec['normalized_smoothness_weight'],self.spec['normalized_jerk_ratio'],
            self.spec['normalized_tcp_ratio'],self.spec.get('normalized_tcp_jerk_ratio',0.))
        checkpoint=self.root/'checkpoints/left/left_all_kan_h128_astronaut_v1_best.pt'
        saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
        actor=TrajectoryModel(TrajectoryModelConfig(**saved['config'])).to('cuda')
        loaded,skipped=actor.load_compatible_state_dict(saved['model'])
        assert not (loaded.missing_keys or loaded.unexpected_keys or skipped)
        actor.eval();actor.requires_grad_(False);self.generator=KANSeed(GraphPolicy(actor),self.model)
        self.decision_sha256=hashlib.sha256(decision.read_bytes()).hexdigest()

    def _seed(self,start,goal,index,history):
        path,info=self.generator.generate(start,goal,max_steps=self.kan_steps,history=history)
        path+=start-path[0]
        path+=np.linspace(0,1,len(path))[:,None]*(goal-path[-1])
        horizon=self.planner.trajopt_solver.action_horizon
        parameter=np.linspace(0,1,len(path))
        if self.spec['seed_parameterization']=='arclength':
            parameter=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
            parameter/=max(parameter[-1],1e-12)
            keep=np.r_[True,np.diff(parameter)>1e-10];path=path[keep];parameter=parameter[keep]
        sampled=np.column_stack([np.interp(np.linspace(0,1,horizon),parameter,path[:,j]) for j in range(8)])
        phase=np.linspace(0,1,horizon);straight=start[None]+phase[:,None]*(goal-start)[None]
        assert self.spec['seeds']>=2 and self.spec['seed_noise']>0
        paths=[sampled,straight];rng=np.random.default_rng(20260912+index)
        for _ in range(self.spec['seeds']-2):
            c=rng.normal(size=(2,8))*self.spec['seed_noise']
            paths.append(straight+np.sin(np.pi*phase)[:,None]*c[0]+.5*np.sin(2*np.pi*phase)[:,None]*c[1])
        seed=torch.tensor(np.clip(np.array(paths)-self.model.q_ref,-2*np.pi,2*np.pi),device='cuda',dtype=torch.float32)[None]
        return seed,info

    def plan(self,start,goal,index=0,history=None,start_velocity=None,start_acceleration=None,initial_duration=None,fixed_duration=None):
        """Verified static/rest-goal plan; moving-start API needs separate tests.

        history must contain executed increments (or explicitly tagged future
        reference prediction outside this class). It is never modified here.
        No payload support is implied by this robot-only frozen scene adapter.
        """
        begun=time.perf_counter();start=np.asarray(start,dtype=float)
        goal=start+(np.asarray(goal,dtype=float)-start+np.pi)%(2*np.pi)-np.pi
        velocity=np.zeros(8) if start_velocity is None else np.asarray(start_velocity,dtype=float)
        acceleration=np.zeros(8) if start_acceleration is None else np.asarray(start_acceleration,dtype=float)
        assert all(x.shape==(8,) and np.isfinite(x).all() for x in (start,goal,velocity,acceleration))
        for name,q in (('start',start),('goal',goal)):
            self.sensor.query(q)
            if min((r['distance'] for r in self.sensor.last_rows),default=.08)<-1e-7:
                return dict(success=False,reason='invalid_'+name,end_to_end_wall_s=time.perf_counter()-begun)
        tensor=lambda a:torch.tensor(a,device='cuda',dtype=torch.float32)[None]
        qs=JointState.from_position(tensor(start-self.model.q_ref),joint_names=self.planner.joint_names)
        qg=JointState.from_position(tensor(goal-self.model.q_ref),joint_names=self.planner.joint_names)
        if start_velocity is not None:qs.velocity=tensor(velocity)
        if start_acceleration is not None:qs.acceleration=tensor(acceleration)
        solving=time.perf_counter();seed,seed_info=self._seed(start,goal,index,history)
        options={};clock=nullcontext();effective_fixed_duration=None
        if fixed_duration is not None:
            from src.planning.fixed_spline_clock import fixed_native_clock,aligned_duration
            if initial_duration is not None:raise ValueError('Choose initial or fixed duration, not both')
            if not np.isfinite(fixed_duration) or fixed_duration<=0:raise ValueError('Positive fixed duration required')
            segments=self.planner.trajopt_solver.action_horizon+4
            quantum=segments*self.planner.trajopt_solver.config.interpolation_dt
            effective_fixed_duration=aligned_duration(float(fixed_duration),float(quantum))
            initial_duration=effective_fixed_duration
            collocation_dt=initial_duration/(segments*self.planner.trajopt_solver.interpolation_steps)
            clock=fixed_native_clock(self.planner.trajopt_solver,collocation_dt)
            options['finetune_attempts']=0
        if initial_duration is not None:
            if not np.isfinite(initial_duration) or initial_duration<=0:raise ValueError('Positive initial duration required')
            intervals=(self.planner.trajopt_solver.action_horizon+4)*self.planner.trajopt_solver.interpolation_steps
            options['dt']=torch.full((1,self.spec['seeds']),initial_duration/intervals,device='cuda',dtype=torch.float32)
        with clock:
            result=self.planner.trajopt_solver.solve_cspace(qg,qs,seed_traj=seed,**options)
        torch.cuda.synchronize();solve_time=time.perf_counter()-solving
        if result is None:return dict(success=False,reason='native_no_result',planner_wall_s=solve_time)
        native_success=bool(result.success.any());trajectory=result.get_interpolated_plan()
        q=trajectory.position.detach().cpu().numpy().reshape(-1,8)+self.model.q_ref
        dt=float(self.planner.trajopt_solver.config.interpolation_dt);duration=(len(q)-1)*dt
        knots=result.js_solution.knot.detach().cpu().numpy();knot_dt=result.js_solution.knot_dt.detach().cpu().numpy()
        vmax=float(trajectory.velocity.abs().max());substeps=max(3,int(np.ceil(vmax*dt/.009)))
        while True:
            dense_dt=dt/substeps
            dense=sample_native(knots,knot_dt,3,start-self.model.q_ref,goal-self.model.q_ref,dense_dt,duration=duration,
                                start_velocity=velocity,start_acceleration=acceleration)
            if np.max(abs(dense['velocity']))*dense_dt<=.01:break
            substeps*=2
        dense_q=dense['position']+self.model.q_ref
        check=(self.verification_pool.validate(dense_q,dense_dt) if self.verification_pool is not None
               else validate_trajectory(self.verification_sensor,dense_q,dense_dt))
        if self.contact_only_verification and self.verification_pool is None:
            minimum=check.pop('exact_minimum_m')
            check.update(minimum_queried_contact_distance_m=minimum if np.isfinite(minimum) else None,
                         positive_clearance_measured=False,verification_mode='contact_only_same_signed_fcl')
        limits=kinematic_gate(dense['position'],dense['velocity'],dense['acceleration'],dense['jerk'])
        actual=self.model.forward(q[-1])[0][self.model.index[self.model.meta['tool_path']]]
        desired=self.model.forward(goal)[0][self.model.index[self.model.meta['tool_path']]]
        q_error=float(np.max(abs((q[-1]-goal+np.pi)%(2*np.pi)-np.pi)))
        tcp_error=float(np.linalg.norm(actual[:3,3]-desired[:3,3]))
        angle_error=float(np.arccos(np.clip((np.trace(actual[:3,:3].T@desired[:3,:3])-1)/2,-1,1)))
        start_errors=dict(position=float(np.max(abs(dense_q[0]-start))),
            velocity=float(np.max(abs(dense['velocity'][0]-velocity))),acceleration=float(np.max(abs(dense['acceleration'][0]-acceleration))))
        boundary_ok=(start_errors['position']<1e-5 and start_errors['velocity']<.001 and start_errors['acceleration']<.01
                     and np.max(abs(dense['velocity'][-1]))<.001 and np.max(abs(dense['acceleration'][-1]))<.01)
        flags=dict(native_failed=not native_success,collision=check['collision'] is not None,
            terminal_error=q_error>=.01 or tcp_error>=.005 or angle_error>=.02,
            boundary_error=not boundary_ok,kinematic_limit=not limits['passed'],
            planner_deadline=solve_time>self.spec['planner_budget_s'])
        accepted=not any(flags.values())
        arrays=dict(q=q,dt=dt,native_knots=knots,native_knot_dt=knot_dt,native_degree=3,
            native_start=start-self.model.q_ref,native_goal=goal-self.model.q_ref,
            native_start_velocity=velocity,native_start_acceleration=acceleration,dense_q=dense_q,dense_dt=dense_dt,
            **{'dense_'+k:dense[k] for k in ('velocity','acceleration','jerk')})
        return dict(success=accepted,reason='success' if accepted else next(k for k,v in flags.items() if v),
            failure_flags=flags,planner_wall_s=solve_time,end_to_end_wall_s=time.perf_counter()-begun,
            seed=seed_info,start_errors=start_errors,terminal_joint_error=q_error,terminal_tcp_m=tcp_error,
            initial_duration=initial_duration,
            fixed_duration_requested=fixed_duration,effective_fixed_duration=effective_fixed_duration,
            terminal_orientation_rad=angle_error,kinematic_gate=limits,**check,arrays=arrays,
            curve=CubicExecutionCurve(knots+self.model.q_ref,start,goal,duration,velocity,acceleration) if accepted else None)
