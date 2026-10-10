"""Load the skinned astronaut rig and deform it with linear blend skinning (numpy)."""
import numpy as np
from pxr import Usd, UsdGeom, UsdSkel
from scipy.spatial.transform import Rotation as R


def open_render_stage(path):
    """Open the build rig or the packaged asset for skinning/rendering: the asset's physics variants are switched
    off in the session layer (its root prim carries the physics payloads, so it cannot simply be left unloaded)."""
    stage = Usd.Stage.Open(path, Usd.Stage.LoadNone)
    root = stage.GetDefaultPrim()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for vs in ("Physics", "BodyColliders"):
            if root.GetVariantSets().HasVariantSet(vs):
                root.GetVariantSet(vs).SetVariantSelection("none")
    stage.Load()
    return stage


class Rig:
    def __init__(self, path):
        # the build rig (/Astronaut) or the packaged asset (/astronaut); physics payloads are not needed
        stage = open_render_stage(path)
        root = stage.GetDefaultPrim().GetPath()
        self.root = root
        skel = UsdSkel.Skeleton(stage.GetPrimAtPath(root.AppendChild("Skeleton")))
        self.names = list(skel.GetJointNamesAttr().Get())
        self.index = {n: i for i, n in enumerate(self.names)}
        paths = list(skel.GetJointsAttr().Get())
        pidx = {p: i for i, p in enumerate(paths)}
        self.parent = np.array([pidx.get(p.rsplit("/", 1)[0], -1) if "/" in p else -1 for p in paths])
        self.bind = np.array([np.array(m).T for m in skel.GetBindTransformsAttr().Get()])  # column-vector 4x4
        self.rest = np.array([np.array(m).T for m in skel.GetRestTransformsAttr().Get()])
        self.inv_bind = np.linalg.inv(self.bind)
        self.B = self.bind[:, :3, :3].copy()  # bind joint orientations
        self.offset = np.array([self.bind[i, :3, 3] - (self.bind[p, :3, 3] if p >= 0 else 0.0)
                                for i, p in enumerate(self.parent)])  # world-axis bone offsets
        self.meshes = []
        for prim in stage.GetPrimAtPath(root.AppendChild("Geometry")).GetChildren():
            m, b = UsdGeom.Mesh(prim), UsdSkel.BindingAPI(prim)
            P = np.array(m.GetPointsAttr().Get(), np.float64)
            k = b.GetJointIndicesPrimvar().GetElementSize()
            idx = np.array(b.GetJointIndicesPrimvar().Get()).reshape(-1, k)
            w = np.array(b.GetJointWeightsPrimvar().Get()).reshape(-1, k)
            counts = np.array(m.GetFaceVertexCountsAttr().Get())
            fvi = np.array(m.GetFaceVertexIndicesAttr().Get())
            shapes = {}
            names_attr = b.GetBlendShapesAttr().Get() if b.GetBlendShapesAttr() else None
            if names_attr:
                for nm, tgt in zip(names_attr, b.GetBlendShapeTargetsRel().GetTargets()):
                    bs = UsdSkel.BlendShape(stage.GetPrimAtPath(tgt))
                    shapes[nm] = (np.array(bs.GetPointIndicesAttr().Get()), np.array(bs.GetOffsetsAttr().Get(), np.float64))
            self.meshes.append(dict(name=prim.GetName(), P=P, idx=idx, w=w, counts=counts, fvi=fvi, shapes=shapes))
        import json
        drv = skel.GetPrim().GetCustomDataByKey("astronaut:correctives")
        self.correctives = json.loads(drv)["sides"] if drv else {}
        self.use_correctives = True

    def joint_pos(self, i):
        return self.bind[i, :3, 3]

    def fk(self, local_rot=None, root_xform=None):
        """Forward kinematics from world-aligned rotation deltas.

        local_rot: {name: 3x3} rotation of each joint relative to its parent, expressed in world-aligned axes
        (the identity pose is the bind pose, whatever the joint frames are). Returns joint world 4x4 matrices
        (column vectors) that include the bind joint orientations, i.e. proper UsdSkel joint transforms."""
        local_rot = local_rot or {}
        A = np.zeros((len(self.names), 3, 3))
        W = np.zeros_like(self.rest)
        for i, n in enumerate(self.names):
            p = self.parent[i]
            R = local_rot.get(n, np.eye(3))
            W[i] = np.eye(4)
            if p < 0:
                A[i] = R
                W[i, :3, 3] = self.bind[i, :3, 3]
            else:
                A[i] = A[p] @ R
                W[i, :3, 3] = W[p, :3, 3] + A[p] @ self.offset[i]
            W[i, :3, :3] = A[i] @ self.B[i]
        if root_xform is not None:
            W = root_xform @ W
        return W

    def aligned(self, W):
        """World-aligned rotation deltas (relative to bind) of joint world matrices W."""
        return W[:, :3, :3] @ np.transpose(self.B, (0, 2, 1))

    def fk_joint(self, joint_rot=None, root_xform=None):
        """Forward kinematics from rotations in each joint's own frame (UsdSkel/OpenXR local semantics):
        local transform = rest translation with rotation rest_rot @ joint_rot."""
        joint_rot = joint_rot or {}
        W = np.zeros_like(self.rest)
        for i, n in enumerate(self.names):
            L = self.rest[i].copy()
            if n in joint_rot:
                L[:3, :3] = L[:3, :3] @ joint_rot[n]
            W[i] = L if self.parent[i] < 0 else W[self.parent[i]] @ L
        if root_xform is not None:
            W = root_xform @ W
        return W

    def local_from_world(self, W):
        """Joint-local 4x4 transforms (for SkelAnimation) from joint world matrices."""
        return np.array([W[i] if self.parent[i] < 0 else np.linalg.inv(W[self.parent[i]]) @ W[i]
                         for i in range(len(W))])

    def corrective_weights(self, W):
        """Blend-shape weights from the shoulder RBF drivers (see scripts/correctives.py)."""
        out = {}
        if not self.correctives:
            return out
        A = W[:, :3, :3] @ np.transpose(self.B, (0, 2, 1))
        for side, d in self.correctives.items():
            u, l, c = self.index[d["joint"]], self.index[d["child"]], self.index[d["frame"]]
            v = A[c].T @ (W[l, :3, 3] - W[u, :3, 3])
            v /= np.linalg.norm(v)
            ang = np.arccos(np.clip(np.array(d["dirs"]) @ v, -1, 1))
            phi = np.exp(-(ang / np.radians(d["sigma_deg"])) ** 2)
            for nm, wv in zip(d["shapes"], phi @ np.array(d["coeff"])):
                out[nm] = float(np.clip(wv, -0.25, 1.25))
        return out

    def rest_points(self, m, weights):
        """Bind points of mesh m with blend shapes applied."""
        P = m["P"]
        if not weights or not m["shapes"]:
            return P
        P = P.copy()
        for nm, (ix, off) in m["shapes"].items():
            wv = weights.get(nm, 0.0)
            if wv:
                P[ix] += wv * off
        return P

    def skin(self, world, correctives=None):
        S = world @ self.inv_bind
        use = self.use_correctives if correctives is None else correctives
        weights = self.corrective_weights(world) if use else {}
        out = []
        for m in self.meshes:
            P = self.rest_points(m, weights)
            Sv = np.einsum("nk,nkij->nij", m["w"], S[m["idx"]])
            out.append(np.einsum("nij,nj->ni", Sv[:, :3, :3], P) + Sv[:, :3, 3])
        return out

    def triangles(self):
        """Fan-triangulated (n, 3) index arrays per mesh."""
        tris = []
        for m in self.meshes:
            c, fvi = m["counts"], m["fvi"]
            s = np.r_[0, np.cumsum(c)[:-1]]
            t = [np.stack([fvi[s[c >= k + 2]], fvi[(s + k)[c >= k + 2]], fvi[(s + k + 1)[c >= k + 2]]], 1)
                 for k in range(1, c.max() - 1)]
            tris.append(np.concatenate(t))
        return tris


def rot(axis, deg):
    return R.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg)).as_matrix()
