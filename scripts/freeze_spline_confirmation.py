"""Freeze selected DEVELOPMENT configurations before any confirmation run."""
from pathlib import Path
import argparse,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]


FIELDS=('method','seeds','attempts','optimizer','transition','seed_noise','seed_parameterization','seed_encoder_sha256',
    'replace_kan_with_random','tcp_acceleration_weight','normalized_smoothness_weight','normalized_jerk_ratio',
    'normalized_tcp_ratio','normalized_tcp_jerk_ratio','graph_policy','graph_enabled','cuda_graph_enabled','self_margin_per_link_m',
    'scene_sha256','effective_robot_sha256','world_sha256','planner_budget_s','torch_threads','warmup_tasks',
    'warmup_indices','cpu_validation_workers','gpu_benchmark_serialization','kinematic_gate',
    'collision_acceptance','goal_joint_max_rad','goal_tcp_m','goal_orientation_rad')
OPTIONAL_DEFAULTS=dict(normalized_smoothness_weight=0.,normalized_jerk_ratio=.002,normalized_tcp_ratio=0.,normalized_tcp_jerk_ratio=0.,seed_encoder_sha256=None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--manifests',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists(),'A frozen decision is immutable; use a new decision version'
    methods={};source_names=set()
    for role,directory in [('baseline',args.baseline),('candidate',args.candidate)]:
        protocol=json.loads((directory/'protocol.json').read_text());status=json.loads((directory/'status.json').read_text())
        assert status['status']=='complete' and not protocol['heldout']
        assert protocol['method']==('native' if role=='baseline' else 'kan')
        actor=json.loads((directory/'actor.json').read_text()) if role=='candidate' else None
        methods[protocol['method']]=dict(role=role,development_trial=str(directory),
            protocol={key:protocol[key] if key in protocol else OPTIONAL_DEFAULTS[key] for key in FIELDS},actor_max_steps=actor['max_steps'] if actor else None,
            source_status_sha256=hashlib.sha256((directory/'status.json').read_bytes()).hexdigest())
        source_names.update(protocol['source_sha256'])
    source_names.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src/models').glob('*.py'))
    source_names.update(['scripts/compare_native_splines.py','scripts/assess_paired_benchmark.py',
        'scripts/freeze_spline_confirmation.py','src/planning/paired_statistics.py',
        'results/sota_simple_v1/scene_model.json','checkpoints/left/left_all_kan_h128_astronaut_v1_best.pt','finally.usd'])
    manifests={}
    for file in args.manifests:
        manifest=json.loads(file.read_text());assert manifest['protocol']['purpose']=='frozen_confirmation_not_for_tuning'
        manifests[hashlib.sha256(file.read_bytes()).hexdigest()]=dict(path=str(file),tasks=len(manifest['tasks']))
    result=dict(status='frozen_before_confirmation',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        methods=methods,allowed_confirmation_manifests=manifests,
        files_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(source_names)},
        acceptance=dict(primary_tasks=1400,success_absolute_minimum=.9,paired_noninferiority_margin=.02,
            smoothness='all 8 equal-duration geometric mean ratios and individual 95% upper bounds < 1',
            low_impact_scope='kinematic acceleration and jerk only',
            online_and_delivery='not certified by this static confirmation'),
        instruction='Run complete suites without tuning or stopping at favorable interim results. Failed confirmation requires a new version and fresh confirmation data if used for development.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FROZEN',hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__=='__main__':main()
