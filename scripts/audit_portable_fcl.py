"""Independent FCL collision/gradient audit, without launching Isaac."""
from pathlib import Path
import sys, json, argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'external/research_packages')]
import numpy as np
from src.geometry.portable_scene import PortableScene, FCLSceneDistance

p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, default=ROOT/'results/sota_simple_v1/scene_model.json')
p.add_argument('--output', type=Path, default=ROOT/'results/sota_simple_v1/fcl_audit.json')
p.add_argument('--limit', type=int, default=64)
args = p.parse_args()
model = PortableScene(args.model)
assert model.meta['fk_gate']
sensor = FCLSceneDistance(model)
old = json.loads((ROOT/'results/geometric_projection_v2/cooked_pairwise_audit.json').read_text(encoding='utf-8'))
old = {r['index']: r for r in old['rows']}
rows = []
gradients = []
for item in model.meta['fk_audit'][:args.limit]:
    q = np.array(item['q'])
    _, _, _, stats = sensor.query(q)
    selected = [dict(r) for r in sensor.last_rows]
    fcl = {tuple(r['pair']) for r in selected if r['distance'] < -1e-7}
    touching = {tuple(r['pair']) for r in selected if abs(r['distance']) <= 1e-7}
    physx = set(map(tuple, item['physx_pairs']))
    gjk = {tuple(r['pair']) for r in old[item['index']]['overlap_pairs']}
    row = dict(index=item['index'], stats=stats, fcl_count=len(fcl), physx_count=len(physx),
               fcl_missing_physx=sorted(physx-fcl-touching), fcl_extra_physx=sorted(fcl-physx),
               gjk_missing_fcl=sorted(fcl-gjk), gjk_extra_fcl=sorted(gjk-fcl-touching),
               overlaps=[dict(pair=r['pair'], distance_m=r['distance'], hull_a=r['hull_a'], hull_b=r['hull_b'])
                         for r in selected if r['distance'] < 1e-7])
    rows.append(row)
    print('FCL_AUDIT', item['index'], row['fcl_count'], row['physx_count'], stats['query_ms'], flush=True)
    if len(rows) <= 3:
        checks = [r for r in selected if abs(r['distance']) > .002 and abs(r['distance']) < .05]
        fd = {r['pair']: [] for r in checks}
        for axis in range(8):
            values = []
            for sign in (-1, 1):
                qt = q.copy(); qt[axis] += sign*1e-4
                sensor.query(qt)
                values.append({r['pair']: r['distance'] for r in sensor.last_rows})
            for pair in fd:
                fd[pair].append((values[1].get(pair, np.nan)-values[0].get(pair, np.nan))/2e-4)
        for r in checks:
            gradients.append(dict(index=item['index'], pair=r['pair'], distance=r['distance'],
                                  error=float(np.linalg.norm(np.array(fd[r['pair']])-r['gradient']))))
    report = dict(status='complete' if len(rows)==min(args.limit,len(model.meta['fk_audit'])) else 'running',
                  scene_sha256=model.meta['scene_sha256'], rows=rows, gradients=gradients,
                  note='FCL strict overlap threshold -1e-7 m; touching is reported separately; no exclusions changed.')
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('FCL_AUDIT_COMPLETE', flush=True)
