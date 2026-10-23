"""Replay identity audit; never adjusts labels or trajectory values."""
from pathlib import Path
import argparse,json,hashlib,numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True)
    p.add_argument('--repeat',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    original=json.loads((args.original/'status.json').read_text())
    repeat=json.loads((args.repeat/'status.json').read_text())
    assert all(x['status']=='complete' for x in (original,repeat))
    lookup={r['index']:r for r in original['rows']};rows=[]
    for row in repeat['rows']:
        old=lookup[row['index']];name=f"trajectory_{row['index']:04d}.npz"
        error={};shapes_equal=True
        if (args.original/name).exists() and (args.repeat/name).exists():
            with np.load(args.original/name) as a,np.load(args.repeat/name) as b:
                for key in ('q','native_knots','native_knot_dt','dense_q'):
                    match=a[key].shape==b[key].shape;shapes_equal &= match
                    error[key]=float(np.max(abs(a[key]-b[key]))) if match else None
        else:shapes_equal=(args.original/name).exists()==(args.repeat/name).exists()
        rows.append(dict(index=row['index'],same_success=row['success']==old['success'],
            same_reason=row['reason']==old['reason'],shapes_equal=shapes_equal,maximum_errors=error,
            within_1e5=shapes_equal and all(v is not None and v<1e-5 for v in error.values())))
    result=dict(gate=all(r['same_success'] and r['same_reason'] and r['within_1e5'] for r in rows),
        samples=len(rows),rows=rows,source_sha256={role:hashlib.sha256((d/'status.json').read_bytes()).hexdigest()
            for role,d in [('original',args.original),('repeat',args.repeat)]},
        interpretation='Development replay identity, not new independent performance evidence')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='rows'})
    print('MISMATCHES',[r for r in rows if not (r['same_success'] and r['same_reason'] and r['within_1e5'])])


if __name__=='__main__':main()
