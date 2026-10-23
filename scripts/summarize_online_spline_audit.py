"""Summarize completed reference-loop development without promoting it to SOTA."""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    directory=ROOT/'results/sota_simple_v1/wall_clock_handoff_dev30_parallel_v1'
    trial=json.loads((directory/'status.json').read_text());assert trial['status']=='complete'
    audit=json.loads((ROOT/'results/sota_simple_v1/wall_clock_reference_geometry_audit_v1.json').read_text())
    assert audit['gate'] and audit['trial_status_sha256']==hashlib.sha256((directory/'status.json').read_bytes()).hexdigest()
    original=ROOT/'results/sota_simple_v1/wall_clock_handoff_dev12_v1';identities=[]
    for file in sorted(original.glob('candidate_*.npz')):
        with np.load(file) as a,np.load(directory/file.name) as b:
            errors={k:float(np.max(abs(a[k]-b[k]))) if a[k].shape==b[k].shape else None
                    for k in ('q','native_knots','native_knot_dt','dense_q')}
        identities.append(dict(file=file.name,maximum_errors=errors,gate=all(v is not None and v<1e-5 for v in errors.values())))
    rows=trial['rows'];accepted=[r for r in rows if r['handoff']['accepted']]
    percentile=lambda values:dict(zip(('median','p95','p99','max'),map(float,np.quantile(values,[.5,.95,.99,1]))))
    report=dict(scope='Conditional moving-start DEVELOPMENT, not independent online confirmation or Isaac delivery',
        count=len(rows),updated_goal_reached=sum(r['updated_goal_reached'] for r in rows),
        moving_boundaries=sum(r['moving_boundary'] for r in rows),
        native_rejections=[r['index'] for r in rows if r['handoff']['reason']=='planner_rejected'],
        late_rejections=[r['index'] for r in rows if r['handoff']['reason']=='late_or_stale'],
        readiness_s_all=percentile([r['ready_s'] for r in rows]),
        readiness_s_accepted=percentile([r['ready_s'] for r in accepted]),
        maximum_command_interval_s=max(r['command_dt_max_s'] for r in rows),
        maximum_per_trial_p95_scheduling_lateness_s=max(r['scheduling_lateness_p95_s'] for r in rows),
        cpu_geometry_and_limits_gate=audit['gate'],
        maximum_seam_jerk_jump=max(r['seam_jerk_jump'] or 0 for r in audit['rows']),
        parallel_does_not_change_candidate_gate=all(r['gate'] for r in identities),candidate_identity=identities)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print({k:v for k,v in report.items() if k!='candidate_identity'})


if __name__=='__main__':main()
