"""Experimental frozen-scene POE kinematics and independent FCL distances.

No Isaac dependency. The exported model must pass FK and collision audits.
FCL uses the *same cooked convex solids*, not visual triangle meshes.
"""
import json
import time
from pathlib import Path
import numpy as np


def skew(w):
    x, y, z = w
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def screw_exp(s, angle):
    v, w = s[:3], s[3:]
    W = skew(w)
    out = np.eye(4)
    out[:3, :3] += np.sin(angle)*W + (1-np.cos(angle))*(W@W)
    out[:3, 3] = (np.eye(3)*angle + (1-np.cos(angle))*W + (angle-np.sin(angle))*(W@W))@v
    return out


class PortableScene:
    def __init__(self, path):
        self.meta = json.loads(Path(path).read_text(encoding='utf-8'))
        self.paths = self.meta['body_paths']
        self.index = {p: i for i, p in enumerate(self.paths)}
        self.home = np.array(self.meta['home'])
        self.depth = np.array(self.meta['depth'])
        self.screws = np.array(self.meta['screws'])
        self.q_ref = np.array(self.meta['q_ref'])
        self.hulls = {p: [np.array(v) for v in hs] for p, hs in self.meta['hulls'].items()}
        self.pairs = self.meta['pairs']

    def forward(self, q):
        prefix = [np.eye(4)]
        spatial = []
        for s, angle in zip(self.screws, np.asarray(q)-self.q_ref):
            T = prefix[-1]
            w = T[:3, :3]@s[3:]
            v = T[:3, :3]@s[:3] + np.cross(T[:3, 3], w)
            spatial.append(np.r_[v, w])
            prefix.append(T@screw_exp(s, angle))
        return np.array(prefix)[self.depth]@self.home, np.array(spatial).T

    def point_jacobian(self, path, point, spatial):
        out = spatial[:3] + np.cross(spatial[3:].T, point).T
        out[:, self.depth[self.index[path]]:] = 0
        return out


class FCLSceneDistance:
    def __init__(self, model, activation_distance=.08, cache_fixed=True):
        import fcl
        from scipy.spatial import ConvexHull
        self.fcl = fcl
        self.model = model
        self.activation_distance = activation_distance
        self.objects = {}
        for path, hulls in model.hulls.items():
            self.objects[path] = []
            for vertices in hulls:
                hull = ConvexHull(vertices)
                faces = hull.simplices.copy()
                tri = vertices[faces]
                inward = np.sum(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])*hull.equations[:, :3], axis=1) < 0
                faces[inward] = faces[inward][:, [0, 2, 1]]
                encoded = np.column_stack((np.full(len(faces), 3), faces)).ravel().tolist()
                geom = fcl.Convex(vertices, len(faces), encoded)
                self.objects[path].append(fcl.CollisionObject(geom))
        self.request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
        self.pair_indices=np.array([[model.index[a],model.index[b]] for a,b,_ in model.pairs])
        self.last_rows = []
        # Frozen-scene cache only. Every moving link is still updated for every
        # query; this does not remove pairs or alter distances/gradients.
        self.fixed_bounds={}
        self.fixed_lo=np.full((len(model.paths),3),np.inf)
        self.fixed_hi=np.full((len(model.paths),3),-np.inf)
        if cache_fixed:
            for path,hulls in model.hulls.items():
                index=model.index[path]
                if model.depth[index]!=0:continue
                T=model.home[index]
                tf=fcl.Transform(T[:3,:3].copy(),T[:3,3].copy())
                bounds=[]
                for obj,v in zip(self.objects[path],hulls):
                    obj.setTransform(tf);world=v@T[:3,:3].T+T[:3,3]
                    bounds.append((world.min(0),world.max(0)))
                self.fixed_bounds[path]=bounds
                self.fixed_lo[index]=np.min([b[0] for b in bounds],axis=0)
                self.fixed_hi[index]=np.max([b[1] for b in bounds],axis=0)
        self.dynamic_hulls={p:h for p,h in model.hulls.items() if p not in self.fixed_bounds}

    def query(self, q):
        started = time.perf_counter()
        transforms, spatial = self.model.forward(q)
        bounds = dict(self.fixed_bounds)
        body_lo=self.fixed_lo.copy()
        body_hi=self.fixed_hi.copy()
        for path, hulls in self.dynamic_hulls.items():
            T = transforms[self.model.index[path]]
            verts = []
            tf = self.fcl.Transform(T[:3, :3].copy(), T[:3, 3].copy())
            for obj, v in zip(self.objects[path], hulls):
                obj.setTransform(tf)
                verts.append(v@T[:3, :3].T+T[:3, 3])
            bounds[path] = [(v.min(0), v.max(0)) for v in verts]
            body_lo[self.model.index[path]]=np.min([b[0] for b in bounds[path]],axis=0)
            body_hi[self.model.index[path]]=np.max([b[1] for b in bounds[path]],axis=0)
        rows = []
        queries = 0
        ia,ib=self.pair_indices.T
        lower=np.linalg.norm(np.maximum(np.maximum(body_lo[ia]-body_hi[ib],body_lo[ib]-body_hi[ia]),0),axis=1)
        candidates=np.flatnonzero(lower<=self.activation_distance)
        for pair_index in candidates:
            a,b,kind=self.model.pairs[pair_index]
            best = None
            for ia, oa in enumerate(self.objects[a]):
                lo, hi = bounds[a][ia]
                for ib, ob in enumerate(self.objects[b]):
                    lb, hb = bounds[b][ib]
                    lower = np.linalg.norm(np.maximum(np.maximum(lo-hb, lb-hi), 0))
                    # Negative distances cannot be pruned by unsigned AABB bound.
                    if lower > max(0, best['distance'] if best else self.activation_distance):
                        continue
                    result = self.fcl.DistanceResult()
                    d = float(self.fcl.distance(oa, ob, self.request, result))
                    queries += 1
                    if not np.isfinite(d):
                        raise RuntimeError('nonfinite_fcl_distance')
                    if best is not None and d >= best['distance']:
                        continue
                    pa, pb = map(np.asarray, result.nearest_points)
                    normal = (pa-pb)/d if abs(d) > 1e-9 else np.zeros(3)
                    if abs(d) > 1e-9 and abs(np.linalg.norm(normal)-1) > 1e-3:
                        raise RuntimeError('invalid_fcl_witness')
                    g = normal@(self.model.point_jacobian(a, pa, spatial)-self.model.point_jacobian(b, pb, spatial))
                    best = dict(pair=(a, b), kind=kind, distance=d, normal=normal, point_a=pa, point_b=pb, gradient=g,
                                hull_a=ia, hull_b=ib)
            if best is not None and best['distance'] <= self.activation_distance:
                rows.append(best)
        self.last_rows = rows
        mins = [min((r['distance'] for r in rows if (r['kind']=='left_self')==s), default=float('nan')) for s in (True, False)]
        return (np.array([r['distance'] for r in rows]), np.array([r['gradient'] for r in rows]).reshape(-1, 8),
                np.array(mins), dict(query_ms=(time.perf_counter()-started)*1000, hull_queries=queries, active_pairs=len(rows)))
