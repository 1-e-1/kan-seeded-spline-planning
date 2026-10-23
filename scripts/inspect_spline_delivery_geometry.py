"""Read-only USD extraction for future payload-aware spline experiments.

Writes a separate evidence artifact only. Does not change USD, fixtures,
backbone, frozen benchmark, or claim PhysX cooked-payload equivalence.
"""
from pathlib import Path
import argparse,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd,UsdGeom,UsdPhysics
from src.geometry.portable_scene import PortableScene


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    scene=ROOT/'finally.usd';fixture=ROOT/'results/smooth_kan_v1/payload_delivery_tasks_local.json'
    config=json.loads(fixture.read_text());model=PortableScene(ROOT/'results/sota_simple_v1/scene_model.json')
    assert digest(scene)==config['scene_sha256']==model.meta['scene_sha256']
    stage=Usd.Stage.Open(str(scene));assert UsdGeom.GetStageMetersPerUnit(stage)==1.
    cache=UsdGeom.XformCache();objects={}
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not (path.startswith('/Bolt_') or path.startswith('/Nut_')):continue
        assert prim.IsA(UsdGeom.Mesh),'Inspect non-mesh payload conversion separately'
        world=np.asarray(cache.GetLocalToWorldTransform(prim)).T
        rigid=world.copy();rigid[:3,:3]/=np.linalg.norm(rigid[:3,:3],axis=0)
        assert np.linalg.det(rigid[:3,:3])>0
        vertices=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
        vertices_world=(np.c_[vertices,np.ones(len(vertices))]@world.T)[:,:3]
        local=(np.c_[vertices_world,np.ones(len(vertices))]@np.linalg.inv(rigid).T)[:,:3]
        hull=ConvexHull(local);cover=local[hull.vertices]
        # Covers raw authored vertices; NOT an audit of PhysX decomposition.
        cover_error=float(np.max(local@hull.equations[:,:3].T+hull.equations[:,3]))
        objects[path]=dict(body_world=rigid.tolist(),raw_vertices_local_m=local.tolist(),
            conservative_convex_cover_vertices_local_m=cover.tolist(),raw_cover_error_m=cover_error,
            authored_collision_approximation=prim.GetAttribute('physics:approximation').Get(),
            scope='raw mesh cover only; must audit cooked hulls and all payload collision pairs before use')
    assert set(objects)=={'/Bolt_01','/Bolt_02','/Nut_01','/Nut_02'}
    rows=[]
    for task in config['tasks']:
        part=np.array(objects[task['part_prim']]['body_world'])
        tool_pick=model.forward(np.array(task['q_pick']))[0][model.index[model.meta['tool_path']]]
        tool_deliver=model.forward(np.array(task['q_deliver']))[0][model.index[model.meta['tool_path']]]
        relative=np.linalg.inv(tool_pick)@part;delivered=tool_deliver@relative
        offset=np.array(config['magnet_contact_offset_tool_m']);tip=tool_pick[:3,:3]@offset+tool_pick[:3,3]
        surface=(delivered@np.r_[task['delivery_contact_offset_part_m'],1.])[:3]
        rows.append(dict(task=task['name'],part=task['part_prim'],tool_to_part=relative.tolist(),
            pickup_gap_m=float(np.linalg.norm(tip-task['pickup_surface_world_m'])),
            delivered_surface_world_m=surface.tolist(),delivery_gap_m=float(np.linalg.norm(surface-config['delivery_target_world_m'])),
            nonselected_loose_objects=[p for p in objects if p!=task['part_prim']],
            phase_targets={k:task[k] for k in ('q_start','q_pick','q_extract','q_deliver')}))
    report=dict(status='geometry_inspection_only',scene_sha256=digest(scene),fixture_sha256=digest(fixture),
        coordinate_convention='world meters; rigid local payload vertices include authored mesh scale',
        loose_objects=objects,tasks=rows,payload_cooked_collision_audit=False,
        instruction='Do not treat pickup/extraction/delivery as successful planning trials. No curve has been generated or executed here.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('GEOMETRY_INSPECTED',len(objects),'loose objects')
    print([{k:r[k] for k in ('task','pickup_gap_m','delivery_gap_m')} for r in rows])


if __name__=='__main__':main()
