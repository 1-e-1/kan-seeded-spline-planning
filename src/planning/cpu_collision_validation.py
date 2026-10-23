"""Independent FCL trajectory checking; CPU only, never repairs a path."""
import time


def validate_trajectory(sensor,configurations,dt):
    minimum=float('inf');checks=0;hit=None;begun=time.perf_counter()
    for step,configuration in enumerate(configurations):
        sensor.query(configuration);checks+=1
        closest=min(sensor.last_rows,key=lambda r:r['distance'],default=None)
        if closest is not None:
            minimum=min(minimum,closest['distance'])
            if closest['distance'] < -1e-7:
                hit=dict(step=step,time_s=step*dt,pair=list(closest['pair']),distance_m=closest['distance'])
                break
    return dict(exact_minimum_m=minimum,exact_checks=checks,collision=hit,
                independent_validation_wall_s=time.perf_counter()-begun)


def validate_archive_subprocess(python,script,scene,archive):
    """Run one isolated CPU job. The parent may schedule several CPU jobs.

    Each child has its own FCL state and imports no CUDA framework. A separate
    process avoids Python's GIL and does not copy a live CUDA context.
    """
    import subprocess,json,os
    result=subprocess.run([str(python),str(script),'--scene',str(scene),'--archive',str(archive)],
        capture_output=True,text=True,encoding='utf-8',timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:
        raise RuntimeError(f'CPU verification failed ({result.returncode}): {result.stderr[-4000:]}')
    return json.loads(result.stdout)
