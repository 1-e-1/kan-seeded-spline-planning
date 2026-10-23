"""Summarize complete like-for-like DEVELOPMENT baseline selection."""
from pathlib import Path
import json,argparse,collections,numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    base=ROOT/'results/sota_simple_v1'
    names=['curobo_k32_j6500_c10_s6_serial_dev350',
           'curobo_k32_j6500_c10_s6_i200_serial_dev350',
           'curobo_k32_j6500_c10_s6_i400_serial_dev350',
           'kan_normalized_tcp01_k32_i200_dev350']
    rows=[];protocols=[]
    for name in names:
        status=json.loads((base/name/'status.json').read_text())
        protocol=json.loads((base/name/'protocol.json').read_text());protocols.append(protocol)
        assert status['status']=='complete' and status['completed']==350 and not protocol['heldout']
        invalid=sum(r['reason'] in ('invalid_start','invalid_goal') for r in status['rows'])
        rows.append(dict(trial=name,method=protocol['method'],iterations=protocol['optimizer']['optimizer']['num_iters'],
            success=status['successes'],total=350,valid_endpoints=350-invalid,
            reasons=dict(collections.Counter(r['reason'] for r in status['rows'])),
            latency={k:dict(zip(('median','p95','p99'),np.quantile([r[k] for r in status['rows'] if k in r],[.5,.95,.99])))
                for k in ('planner_wall_s','independent_validation_wall_s','end_to_end_wall_s')}))
    keys=('manifest_sha256','scene_sha256','effective_robot_sha256','planner_budget_s',
          'warmup_tasks','warmup_indices','cpu_validation_workers','kinematic_gate','collision_acceptance')
    assert all(all(p[k]==protocols[0][k] for p in protocols) for k in keys)
    # Success defines the user's primary comparator. Ties use more iterations
    # to retain the more optimized baseline, not the easiest smoothness target.
    native=sorted((r for r in rows if r['method']=='native'),key=lambda r:(r['success'],r['iterations']),reverse=True)
    result=dict(status='development_only',selection='highest all-request on-time success; ties retain larger iteration budget',
        selected_baseline=native[0]['trial'],rows=rows,
        caveat='Static solver budget excludes independent verification. No global SOTA, deployment distribution, online or impact-force claim.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
