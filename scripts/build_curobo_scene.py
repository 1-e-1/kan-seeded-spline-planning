"""Build isolated cuRobo assets from audited cooked geometry, not visual meshes.

Moving links use native MorphIt sphere fitting; static solids remain separate
watertight convex meshes. The only fixed link with a moving-link exclusion is
kept in the robot sphere model so the original filter can be represented.
Sphere fitting is an approximation, requiring independent FCL acceptance.
"""
from pathlib import Path
import sys, json, argparse, dataclasses
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'external/research_packages'), str(ROOT/'external/curobo_v0_8_0')]
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
import warp as wp
wp.config.kernel_cache_dir = str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir = str(ROOT/'results/sota_simple_v1/curobo_cache')
from curobo._src.geom.sphere_fit import fit_spheres_to_mesh
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState
from src.geometry.portable_scene import PortableScene

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, default=ROOT/'results/sota_simple_v1/curobo_scene')
p.add_argument('--density', type=float, default=2.)
p.add_argument('--iterations', type=int, default=200)
p.add_argument('--protrusion-weight', type=float, default=10.)
args = p.parse_args()


def save(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    torch.manual_seed(20260912); np.random.seed(20260912)
    model = PortableScene(ROOT/'results/sota_simple_v1/scene_model.json')
    assert model.meta['fk_gate']
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    pairs = {tuple(sorted((a,b))) for a,b,_ in model.pairs}
    moving = [p for p in model.hulls if model.depth[model.index[p]]>0]
    fixed_robot = [p for p in model.hulls if p not in moving and any(tuple(sorted((a,p))) not in pairs for a in moving)]
    robot_paths = sorted(moving+fixed_robot)
    names = {p:f'body_{model.index[p]}' for p in robot_paths}
    frames = [np.eye(4)]
    robot = ET.Element('robot', name='audited_left_8dof')
    ET.SubElement(robot, 'link', name='world_base')

    def joint(name, parent, child, T, typ):
        j = ET.SubElement(robot, 'joint', name=name, type=typ)
        ET.SubElement(j, 'parent', link=parent); ET.SubElement(j, 'child', link=child)
        ET.SubElement(j, 'origin', xyz=' '.join(map(str,T[:3,3])), rpy=' '.join(map(str,Rotation.from_matrix(T[:3,:3]).as_euler('xyz'))))
        return j

    for i,s in enumerate(model.screws):
        w = s[3:]; v = s[:3]
        helper = np.eye(3)[np.argmin(abs(w))]
        x = np.cross(helper,w); x /= np.linalg.norm(x)
        T = np.eye(4); T[:3,:3] = np.column_stack((x,np.cross(w,x),w)); T[:3,3] = np.cross(w,v)
        ET.SubElement(robot, 'link', name=f'axis_{i+1}')
        j = joint(f'joint{i+1}', f'axis_{i}' if i else 'world_base', f'axis_{i+1}', np.linalg.inv(frames[-1])@T, 'revolute')
        ET.SubElement(j,'axis',xyz='0 0 1')
        # Unwrapped periodic planning coordinates. Runtime input is q-q_ref.
        ET.SubElement(j,'limit',lower=str(-2*np.pi),upper=str(2*np.pi),effort='200',velocity='1.5')
        frames.append(T)
    spheres = {}; fits = []
    for path in robot_paths:
        name = names[path]; i = model.index[path]; depth = model.depth[i]
        link = ET.SubElement(robot,'link',name=name)
        joint('fixed_'+name, f'axis_{depth}' if depth else 'world_base', name, np.linalg.inv(frames[depth])@model.home[i], 'fixed')
        spheres[name] = []
        for h,vertices in enumerate(model.hulls[path]):
            mesh = trimesh.convex.convex_hull(vertices)
            file = out/f'{name}_{h}.stl'; mesh.export(file)
            collision = ET.SubElement(link,'collision')
            ET.SubElement(ET.SubElement(collision,'geometry'),'mesh',filename=str(file))
            cached = out/f'{name}_{h}_fit.json'
            if cached.exists():
                record = json.loads(cached.read_text()); assert record['density']==args.density and record['iterations']==args.iterations and record.get('protrusion_weight',10.)==args.protrusion_weight
            else:
                fit = fit_spheres_to_mesh(mesh, sphere_density=args.density, iterations=args.iterations, compute_metrics=True, protrusion_weight=args.protrusion_weight)
                record = dict(path=path,hull=h,density=args.density,iterations=args.iterations,protrusion_weight=args.protrusion_weight,
                              spheres=[dict(center=list(map(float,c)),radius=float(r)) for c,r in zip(fit.centers,fit.radii)],
                              metrics=dataclasses.asdict(fit.metrics),debug=fit.debug_info)
                save(cached,record)
            spheres[name].extend(record['spheres']); fits.append(record)
        print('CUROBO_FIT',path,len(spheres[name]),flush=True)
    urdf = out/'left.urdf'; ET.ElementTree(robot).write(urdf, encoding='utf-8',xml_declaration=True)
    ignore = {names[a]:[names[b] for b in robot_paths if b>a and tuple(sorted((a,b))) not in pairs] for a in robot_paths}
    cfg = dict(robot_cfg=dict(kinematics=dict(format_version=2., urdf_path=str(urdf),asset_root_path=str(out),base_link='world_base',
        tool_frames=list(names.values()), collision_link_names=list(names.values()), collision_spheres=spheres,
        self_collision_ignore=ignore,self_collision_buffer={n:0. for n in names.values()},collision_sphere_buffer=0.,
        cspace=dict(joint_names=[f'joint{i+1}' for i in range(8)], default_joint_position=[0.]*8,
                    cspace_distance_weight=[1.]*8, null_space_weight=[1.]*8,max_acceleration=4.,max_jerk=30.))))
    save(out/'robot.json',cfg)
    world = {}
    for path,hulls in model.hulls.items():
        if path in robot_paths: continue
        T = model.home[model.index[path]]
        for h,v in enumerate(hulls):
            mesh = trimesh.convex.convex_hull(v@T[:3,:3].T+T[:3,3])
            world[f'fixed_{model.index[path]}_{h}'] = dict(vertices=mesh.vertices.tolist(),faces=mesh.faces.tolist(),pose=[0.,0.,0.,1.,0.,0.,0.])
    save(out/'world.json',dict(mesh=world))
    # Test the complete independently parsed chain, not only the tool.
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(str(out/'robot.json')))
    audit = []
    for item in model.meta['fk_audit']:
        q = np.array(item['q']); expected,_ = model.forward(q)
        state = kin.compute_kinematics(JointState.from_position(torch.tensor(q-model.q_ref,device='cuda',dtype=torch.float32)[None],joint_names=kin.joint_names))
        errors = []
        for path,name in names.items():
            pose = state.tool_poses.get_link_pose(name).get_numpy_affine_matrix().reshape(3,4)
            errors.append(float(np.max(abs(pose-expected[model.index[path], :3]))))
        audit.append(max(errors))
    report = dict(status='complete',scene_sha256=model.meta['scene_sha256'], native_version='v0.8.0',
        source_commit='4ea77366ca48ee453e7df139e39fa6532af49f3b', robot_paths=robot_paths,names=names,
        fixed_mesh_count=len(world),sphere_count=sum(map(len,spheres.values())),fits=fits,
        fk_audit_max_abs=max(audit),fk_gate=max(audit)<2e-5, q_offset=model.q_ref.tolist(),
        note='Native MorphIt approximation; no collision-pair pruning beyond original USD filters. All plans require exact cooked-hull acceptance. Kinematic only, inertia is not calibrated.')
    save(out/'build_report.json',report)
    cfg['robot_cfg']['kinematics']['tool_frames'] = [names[model.meta['tool_path']]]
    save(out/'robot.json',cfg)
    print('CUROBO_BUILD_COMPLETE',report['sphere_count'],report['fixed_mesh_count'],report['fk_audit_max_abs'],flush=True)
    assert report['fk_gate']


if __name__ == '__main__': main()
