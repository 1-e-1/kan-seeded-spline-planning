"""Export cooked geometry and measured POE; audit FK against live Isaac poses."""
from pathlib import Path
import sys, argparse, hashlib, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=ROOT/'results/sota_simple_v1/scene_model.json')
p.add_argument('--random', type=int, default=64)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import numpy as np
import torch
from src.env.dual_arm_scene import DualArmScene, LEFT_TOOL
from src.geometry.physx_clearance import configure_proximity_stage, initialize_all_body_filters
from src.geometry.cooked_scene_distance import CookedSceneDistance
from src.geometry.portable_scene import PortableScene
from src.data.strict_expert import StableOracle, atomic_json


def main():
    sha = hashlib.sha256((ROOT/'finally.usd').read_bytes()).hexdigest()
    scene = DualArmScene(ROOT/'finally.usd', device=args.device, execution_mode='kinematic', pin_kinematic_articulation=True,
                         collision_mode='separation', stage_setup=lambda s: configure_proximity_stage(s, .06))
    fixtures, _ = initialize_all_body_filters(scene)
    sensor = CookedSceneDistance(scene)
    assert not sensor.planes, 'Export explicit planes before supporting this scene'
    zero = torch.zeros(8, device=scene.device)
    scene.teleport(zero, zero, settle_steps=5)
    qref = scene.state.q('left')[0].cpu().numpy()
    jac, anchors = sensor.jacobian.body_jacobians()
    j = jac[scene.state.body_indices[LEFT_TOOL]]
    screws = np.concatenate((j[:3]-np.cross(j[3:].T, anchors[scene.state.body_indices[LEFT_TOOL]]).T, j[3:])).T
    assert np.allclose(np.linalg.norm(screws[:, 3:], axis=1), 1, atol=1e-5)
    # Remove float32 angular normalization errors before exponentiation.
    screws /= np.linalg.norm(screws[:, 3:], axis=1)[:, None]
    paths = sorted(set(scene.state.body_paths)|set(sensor.hulls))
    home, depth = [], []
    for path in paths:
        idx = scene.state.body_indices.get(path)
        if idx is None:
            home.append(sensor.static[path]); depth.append(0)
        else:
            home.append(scene.state.pose_world(path)[0].cpu().numpy())
            active = np.linalg.norm(jac[idx, 3:], axis=0) > .5
            n = int(active.sum())
            assert np.array_equal(active, np.arange(8)<n), (path, active)
            depth.append(n)
    meta = dict(status='exported_not_audited', scene_sha256=sha, body_paths=paths, home=np.array(home).tolist(), depth=depth,
                screws=screws.tolist(), q_ref=qref.tolist(), hulls={k:[v.tolist() for v in h] for k,h in sensor.hulls.items()},
                pairs=sensor.pairs, exclusions=sorted(sensor.exclusions), inventory=sensor.inventory,
                joint_paths=[scene.state.dof_paths[i] for i in scene.state.joint_indices['left']],
                fixed_right_q=[0.]*8, tool_path=LEFT_TOOL, base_path=scene.state.base_paths['left'])
    atomic_json(args.output, meta)
    model = PortableScene(args.output)
    rng = np.random.default_rng(20260911)
    oracle = StableOracle(scene)
    rows = []
    monitored = set(scene.state.body_paths)|{path for path,_,_ in fixtures}
    for i, q in enumerate(rng.uniform(-np.pi,np.pi,(args.random,8))):
        labels = oracle.label(q)
        predicted, _ = model.forward(scene.state.q('left')[0].cpu().numpy())
        pos, rot = [], []
        for path in scene.state.body_paths:
            actual = scene.state.pose_world(path)[0].cpu().numpy()
            diff = actual-predicted[model.index[path]]
            pos.append(np.linalg.norm(diff[:3, 3])); rot.append(np.linalg.norm(diff[:3, :3]))
        actual_pairs = {tuple(sorted((r.collider_a, r.collider_b))) for r in scene.monitor.active.values()
                        if r.kind in ('left_self','left_environment','inter_arm') and r.collider_a in monitored and r.collider_b in monitored}
        rows.append(dict(index=f'random_{i}', q=q.tolist(), pos_max_m=float(max(pos)), rotation_max_fro=float(max(rot)),
                         physx_labels=labels.tolist(), physx_pairs=sorted(actual_pairs)))
        print('PORTABLE_FK', i, max(pos), max(rot), flush=True)
    gate = bool(rows) and all(r['pos_max_m']<2e-5 and r['rotation_max_fro']<2e-5 for r in rows)
    assert all(float((v.get_transforms()-initial).abs().max())<1e-6 for _,v,initial in fixtures)
    assert hashlib.sha256((ROOT/'finally.usd').read_bytes()).hexdigest()==sha
    meta.update(status='fk_audit_passed' if gate else 'fk_audit_failed', fk_gate=gate, fk_audit=rows)
    atomic_json(args.output, meta)
    print('PORTABLE_EXPORT_COMPLETE', gate, flush=True)
    assert gate


if __name__ == '__main__':
    try: main()
    finally: app.close(wait_for_replicator=False, skip_cleanup=True)
