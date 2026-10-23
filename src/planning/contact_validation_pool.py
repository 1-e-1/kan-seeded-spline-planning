"""Parallel verification of unchanged samples with persistent CPU workers.

Not a planner. Same FCL/pair list/threshold, deterministic earliest collision.
Own subprocesses are hidden and closed; no CUDA imports in child servers.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,TimeoutError
import subprocess,json,time,os,threading,hashlib
import numpy as np


class ContactValidationPool:
    def __init__(self,root,python,workers=2):
        self.root=Path(root);self.processes=[];self.lock=threading.Lock();self.closed=False
        if workers not in (1,2,4):raise ValueError('Choose 1, 2 or 4 CPU workers')
        audit=json.loads((self.root/'results/sota_simple_v1/contact_only_verification_audit_v1.json').read_text())
        assert audit['gate']
        for name,key in [('src/geometry/portable_scene.py','portable_source_sha256'),
                         ('results/sota_simple_v1/scene_model.json','scene_sha256')]:
            assert hashlib.sha256((self.root/name).read_bytes()).hexdigest()==audit[key]
        self.executor=ThreadPoolExecutor(max_workers=workers)
        try:
            for _ in range(workers):
                process=subprocess.Popen([str(python),'-u',str(self.root/'scripts/contact_validation_server.py'),
                    '--scene',str(self.root/'results/sota_simple_v1/scene_model.json')],stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,text=True,encoding='utf-8',bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                self.processes.append(process)
            for process in self.processes:
                line=process.stdout.readline()
                if not line or not json.loads(line).get('ready'):raise RuntimeError('CPU verifier startup failed')
        except BaseException:
            self.close();raise

    @staticmethod
    def _query(process,q,dt):
        process.stdin.write(json.dumps(dict(q=q.tolist(),dt=dt))+'\n');process.stdin.flush()
        line=process.stdout.readline()
        if not line:raise RuntimeError('CPU verifier exited without result')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result

    def validate(self,q,dt):
        q=np.asarray(q,dtype=float)
        if q.ndim!=2 or q.shape[1]!=8 or not len(q) or not np.isfinite(q).all() or not np.isfinite(dt) or dt<=0:
            raise ValueError('Nonempty finite eight-joint trajectory and positive dt required')
        with self.lock:
            if self.closed:raise RuntimeError('Verifier is closed')
            begun=time.perf_counter();chunks=np.array_split(q,min(len(q),len(self.processes)))
            offsets=np.cumsum([0]+[len(c) for c in chunks[:-1]])
            jobs=[self.executor.submit(self._query,p,c,dt) for p,c in zip(self.processes,chunks)]
            try:results=[job.result(timeout=30) for job in jobs]
            except TimeoutError:
                # Only this pool's explicitly owned verifier processes. Never
                # leave a blocked readline preventing shutdown after timeout.
                for process in self.processes:
                    if process.poll() is None:process.terminate();process.wait(timeout=5)
                raise RuntimeError('CPU collision verification timed out; path is not accepted')
            collisions=[]
            for offset,result in zip(offsets,results):
                if result['collision'] is not None:
                    collision=result['collision'].copy();collision['step']+=int(offset)
                    collision['time_s']=collision['step']*dt;collisions.append(collision)
            first=min(collisions,key=lambda c:c['step']) if collisions else None
            relevant=[r['exact_minimum_m'] for offset,r in zip(offsets,results)
                      if r['exact_minimum_m'] is not None and (first is None or offset<=first['step'])]
            return dict(collision=first,exact_checks=first['step']+1 if first else len(q),
                parallel_processed_checks=sum(r['exact_checks'] for r in results),
                minimum_queried_contact_distance_m=min(relevant) if relevant else None,
                positive_clearance_measured=False,verification_mode='parallel_contact_only_same_signed_fcl',
                verification_workers=len(self.processes),independent_validation_wall_s=time.perf_counter()-begun)

    def close(self):
        if self.closed:return
        self.closed=True
        self.executor.shutdown(wait=True)
        for process in self.processes:
            try:
                if process.poll() is None:
                    process.stdin.write(json.dumps(dict(command='close'))+'\n');process.stdin.flush()
                    process.wait(timeout=5)
            except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                if process.poll() is None:process.terminate();process.wait(timeout=5)
            finally:
                process.stdin.close();process.stdout.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
