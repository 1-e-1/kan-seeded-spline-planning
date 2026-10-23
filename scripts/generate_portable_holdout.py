"""Freeze a fresh episode-level suite without running either planner.

Same 20/60/20 displacement sampling and percentile difficulty rule as the
project's evaluation profile. Endpoint oracle is independently audited FCL on
the current cooked solids. No obstacle-based rejection of otherwise valid tasks.
"""
from pathlib import Path
import sys, argparse, json, hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages')]
import numpy as np
from src.geometry.portable_scene import PortableScene,FCLSceneDistance


def signature(a,b):
    return np.round(np.r_[(a+np.pi)%(2*np.pi)-np.pi,(b+np.pi)%(2*np.pi)-np.pi],4).astype('<f8').tobytes()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',type=Path,default=ROOT/'results/sota_simple_v1/scene_model.json')
    p.add_argument('--seed',type=int,default=2026091201);p.add_argument('--count',type=int,default=350)
    p.add_argument('--exclude-manifests',type=Path,nargs='*',default=[])
    args=p.parse_args();assert args.count>=5 and args.count%5==0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    assert not args.output.exists(),'Never overwrite a frozen suite'
    model=PortableScene(args.scene);sensor=FCLSceneDistance(model)
    seen=set();source_hashes={}
    for file in sorted((ROOT/'dataset_rebuild/astronaut_strict_v4_geometric/left').glob('*.npz')):
        with np.load(file) as z:
            pairs=np.unique(np.column_stack((z['q_start'],z['q_goal'])),axis=0)
        for pair in pairs:
            seen.add(signature(pair[:8],pair[8:]));seen.add(signature(pair[8:],pair[:8]))
        source_hashes[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    old=json.loads((ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json').read_text())
    for task in old['tasks']:
        s=task['task']['left'];a,b=np.array(s['q_start']),np.array(s['q_goal'])
        seen.add(signature(a,b));seen.add(signature(b,a))
    excluded_hashes={}
    for file in args.exclude_manifests:
        excluded_hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        for task in json.loads(file.read_text())['tasks']:
            s=task['task']['left'];a,b=np.array(s['q_start']),np.array(s['q_goal'])
            seen.add(signature(a,b));seen.add(signature(b,a))
    rejected=[];pool=[];rng=np.random.default_rng(args.seed+10000019)
    def safe(q):
        sensor.query(q)
        return min((r['distance'] for r in sensor.last_rows),default=.08)>=-1e-7
    for attempt in range(1,50001):
        q=rng.uniform(-np.pi,np.pi,8)
        if safe(q):pool.append(q)
        else:rejected.append(dict(stage='pool',attempt=attempt,reason='collision'))
        if len(pool)==512:break
    else:raise RuntimeError('Insufficient safe configurations')
    print('POOL_READY',attempt,flush=True)
    pool_attempts=attempt;pool=np.array(pool);rng=np.random.default_rng(args.seed)
    bands=np.array(['easy_like']*(args.count//5)+['medium_like']*(3*args.count//5)+['hard_like']*(args.count//5));rng.shuffle(bands)
    def pose(q):
        T=model.forward(q)[0]
        local=np.linalg.inv(T[model.index[model.meta['base_path']]])@T[model.index[model.meta['tool_path']]]
        return np.r_[local[:3,3],local[:3,0],local[:3,1]].tolist()
    records=[]
    for attempt in range(1,20001):
        band=bands[len(records)];i,j=rng.integers(0,len(pool),2);a=pool[i].copy()
        delta=(pool[j]-a+np.pi)%(2*np.pi)-np.pi
        lo,hi={'easy_like':(.1,.3),'medium_like':(.3,.7),'hard_like':(.7,1.)}[band]
        b=(a+rng.uniform(lo,hi)*delta+rng.normal(0,.03,8)+np.pi)%(2*np.pi)-np.pi
        if band=='hard_like':b=pool[j].copy()
        reason='short_displacement' if np.linalg.norm((b-a+np.pi)%(2*np.pi)-np.pi)<.1 else 'duplicate_episode' if signature(a,b) in seen else 'goal_collision' if not safe(b) else None
        if reason:
            rejected.append(dict(stage='task',attempt=attempt,reason=reason));continue
        seen.add(signature(a,b));seen.add(signature(b,a))
        records.append(dict(task_group=f'holdout_seed{args.seed}_episode{len(records):04d}',sampling_band=str(band),
                            task=dict(left=dict(q_start=a.tolist(),q_goal=b.tolist(),x_start=pose(a),x_goal=pose(b)))))
        if len(records)%50==0:print('ACCEPTED',len(records),attempt,flush=True)
        if len(records)==args.count:break
    else:raise RuntimeError('Insufficient valid endpoint tasks')
    features=[]
    for r in records:
        s=r['task']['left'];delta=(np.array(s['q_goal'])-s['q_start']+np.pi)%(2*np.pi)-np.pi
        dx=np.array(s['x_goal'])-s['x_start'];features.append([np.linalg.norm(delta),max(abs(delta)),np.linalg.norm(dx[:3]),np.linalg.norm(dx[3:])])
    features=np.array(features);rank=np.empty_like(features)
    for j in range(4):rank[np.argsort(features[:,j],kind='stable'),j]=np.linspace(0,1,args.count)
    score=rank@np.array([.35,.15,.30,.20]);order=np.argsort(score,kind='stable')
    labels=np.full(args.count,'medium',dtype=object);labels[order[:args.count//5]]='easy';labels[order[-args.count//5:]]='hard'
    for i,r in enumerate(records):r.update(difficulty=str(labels[i]),difficulty_score=float(score[i]),difficulty_features=features[i].tolist())
    report=dict(protocol=dict(purpose='frozen_confirmation_not_for_tuning',seed=args.seed,task_count=args.count,
                scene_sha256=model.meta['scene_sha256'],scene_model_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                scene_model_path=str(args.scene.resolve()),scene_identity_scope=model.meta.get('identity_scope','source_usd'),
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                collision_oracle=f'FCL original {len(model.pairs)} cooked-hull pairs',other_arm='fixed_q_zero',
                pool_sampling_attempts=pool_attempts,task_sampling_attempts=attempt,train_episode_pair_exclusion_round_rad=1e-4,
                training_files_checked=len(source_hashes),difficulty_distribution={x:int(sum(labels==x)) for x in ('easy','medium','hard')},
                additional_excluded_manifests_sha256=excluded_hashes,
                sampling_profile='evaluation: displacement bands 20/60/20; hard endpoints from independent uniform safe pool',
                caveat='Random endpoints, not deployment frequency measurements. No expert-path feasibility assertion. Exact pair exclusion is not a near-neighbor generalization guarantee.'),tasks=records)
    args.output.with_suffix('.sources.json').write_text(json.dumps(source_hashes,indent=2),encoding='utf-8')
    args.output.with_suffix('.rejections.json').write_text(json.dumps(rejected,indent=2),encoding='utf-8')
    np.savez_compressed(args.output.with_suffix('.pool.npz'),q=pool)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('HOLDOUT_FROZEN',hashlib.sha256(args.output.read_bytes()).hexdigest(),report['protocol'],flush=True)


if __name__=='__main__':main()
