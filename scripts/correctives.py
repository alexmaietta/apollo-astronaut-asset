"""Suit-aware shoulder corrective blend shapes (pose-space deformation) for the astronaut rig.

For each shoulder and each key arm direction, the shoulder region of the torso/arms suit layer is re-solved
with as-rigid-as-possible (ARAP) deformation: the arm moves rigidly with the upper arm, the torso stays on
linear-blend skinning, and the shoulder cap in between bends without the LBS collapse. The difference is
stored in bind space as a UsdSkel BlendShape (posed = skin(bind + w * offset)). Pieces attached to the suit
(arm patches, straps) inherit the offsets of their nearest suit vertex, as they do for skin weights.

Weights come from an RBF on the upper-arm direction in the chest frame (exact at the keys, zero at rest); the
driver data is written to the skeleton's customData ("astronaut:correctives") so any runtime can evaluate it.
Run after build_rig.py; edits build/astronaut_rig.usd in place.
"""
import json
import os
import sys

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdSkel, Vt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from rig import Rig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIG = os.path.join(ROOT, "build", "astronaut_rig.usd")
SUIT = "suit_torso_arms"
FOLLOWERS = ("harness_straps_arm_patches",)
FREE_RADIUS = 0.20  # metres around the shoulder joint that ARAP may reshape (plus the arm/torso blend band)
SIGMA_DEG = 45.0
# key upper-arm directions in the chest frame (X forward, Y = the arm's own side, Z up)
KEYS = {
    "abd90": (0.0, 1.0, 0.0),
    "abd150": (0.0, 0.5, 0.866),
    "flex90": (1.0, 0.12, 0.0),
    "flex150": (0.5, 0.12, 0.866),
    "ext45": (-0.7, 0.12, -0.7),
}


def _unit(v):
    return v / np.linalg.norm(v)


def _rot_between(a, b):
    from scipy.spatial.transform import Rotation as R
    a, b = _unit(a), _unit(b)
    ax = np.cross(a, b)
    s, c = np.linalg.norm(ax), float(a @ b)
    return np.eye(3) if s < 1e-9 else R.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


def side_vec(d, side):
    d = np.array(d, float)
    if side == "Right":
        d[1] = -d[1]
    return _unit(d)


def rest_dir(rig, side):
    u, l = rig.index[f"{side}ArmUpper"], rig.index[f"{side}ArmLower"]
    return _unit(rig.bind[l, :3, 3] - rig.bind[u, :3, 3])


def key_pose(rig, side, direction):
    """World-aligned local rotation at ArmUpper that points the (straight) upper arm along direction."""
    return {f"{side}ArmUpper": _rot_between(rest_dir(rig, side), direction)}


def arm_direction(rig, W, side):
    """Current upper-arm direction expressed in the chest's world-aligned frame."""
    A = rig.aligned(W)
    u, l, c = rig.index[f"{side}ArmUpper"], rig.index[f"{side}ArmLower"], rig.index["Chest"]
    return _unit(A[c].T @ (W[l, :3, 3] - W[u, :3, 3]))


def rbf_matrix(dirs, sigma):
    ang = np.arccos(np.clip(dirs @ dirs.T, -1, 1))
    return np.exp(-(ang / np.radians(sigma)) ** 2)


def triangles(m):
    c, fvi = m["counts"], m["fvi"]
    s = np.r_[0, np.cumsum(c)[:-1]]
    return np.concatenate([np.stack([fvi[s[c >= k + 2]], fvi[(s + k)[c >= k + 2]], fvi[(s + k + 1)[c >= k + 2]]], 1)
                           for k in range(1, c.max() - 1)])


def arap(P, X0, edges, fixed, iters=12):
    """Uniform-weight ARAP: rest positions P, initial/constraint positions X0, fixed mask."""
    n = len(P)
    i, j = edges[:, 0], edges[:, 1]
    L = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    L = L + L.T
    deg = np.asarray(L.sum(1)).ravel()
    Lap = coo_matrix((deg, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr() - L
    free = np.where(~fixed)[0]
    fix = np.where(fixed)[0]
    Lff = Lap[free][:, free].tocsc()
    Lfc = Lap[free][:, fix]
    lu = splu(Lff)
    X = X0.copy()
    ei = np.r_[i, j]
    ej = np.r_[j, i]
    for _ in range(iters):
        e_rest = P[ei] - P[ej]
        e_cur = X[ei] - X[ej]
        S = np.zeros((n, 3, 3))
        np.add.at(S, ei, e_rest[:, :, None] * e_cur[:, None, :])
        U, _, Vt_ = np.linalg.svd(S)
        Rm = np.transpose(Vt_, (0, 2, 1)) @ np.transpose(U, (0, 2, 1))
        bad = np.linalg.det(Rm) < 0
        if bad.any():
            Vt_[bad, 2] *= -1
            Rm[bad] = np.transpose(Vt_[bad], (0, 2, 1)) @ np.transpose(U[bad], (0, 2, 1))
        rhs = np.zeros((n, 3))
        np.add.at(rhs, ei, 0.5 * np.einsum("eab,eb->ea", Rm[ei] + Rm[ej], e_rest))
        b = rhs[free] - Lfc @ X[fix]
        X[free] = np.stack([lu.solve(b[:, k]) for k in range(3)], 1)
    return X


def main():
    rig = Rig(RIG)
    names = [m["name"] for m in rig.meshes]
    suit = rig.meshes[names.index(SUIT)]
    tris = triangles(suit)
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), 1), axis=0)
    shapes = {}  # mesh name -> {shape: offsets (n,3)}
    driver = {}
    suit_tree = cKDTree(suit["P"])
    for side in ("Left", "Right"):
        sh = rig.bind[rig.index[f"{side}ArmUpper"], :3, 3]
        arm_j = [rig.index[n] for n in rig.names if n.startswith(side) and any(t in n for t in ("ArmUpper", "ArmLower", "Hand"))]
        arm_w = (suit["w"] * np.isin(suit["idx"], arm_j)).sum(1)
        # free: the shoulder cap plus the whole arm/torso blend band (the underarm web that LBS stretches)
        free = (np.linalg.norm(suit["P"] - sh, axis=1) < FREE_RADIUS) | ((arm_w > 0.01) & (arm_w < 0.99))
        fixed = ~free
        print(side, "free verts", free.sum())
        keys = {k: side_vec(d, side) for k, d in KEYS.items()}
        for k, d in keys.items():
            W = rig.fk(key_pose(rig, side, d))
            S = W @ rig.inv_bind
            Sv = np.einsum("nk,nkij->nij", suit["w"], S[suit["idx"]])
            X_lbs = np.einsum("nij,nj->ni", Sv[:, :3, :3], suit["P"]) + Sv[:, :3, 3]
            X = arap(suit["P"], X_lbs, edges, fixed)
            delta = np.einsum("nij,nj->ni", np.linalg.inv(Sv[:, :3, :3]), X - X_lbs)  # bind-space offset
            name = f"{side}Shoulder_{k}"
            shapes.setdefault(SUIT, {})[name] = delta
            moved = np.linalg.norm(X - X_lbs, axis=1)
            print(f"{name}: {np.count_nonzero(moved > 1e-4)} verts corrected, max {moved.max() * 100:.1f} cm")
            for f in FOLLOWERS:
                m = rig.meshes[names.index(f)]
                _, nn = suit_tree.query(m["P"])
                shapes.setdefault(f, {})[name] = delta[nn]
        dirs = np.vstack([rest_dir(rig, side)[None]] + [d[None] for d in keys.values()])
        # chest-frame rest direction (bind chest is world aligned at bind)
        Phi = rbf_matrix(dirs, SIGMA_DEG)
        target = np.vstack([np.zeros(len(keys))[None], np.eye(len(keys))])  # rest -> 0, key k -> e_k
        coeff = np.linalg.solve(Phi, target)
        driver[side] = dict(shapes=[f"{side}Shoulder_{k}" for k in keys], dirs=dirs.tolist(), coeff=coeff.tolist(),
                            sigma_deg=SIGMA_DEG, joint=f"{side}ArmUpper", child=f"{side}ArmLower", frame="Chest")

    stage = Usd.Stage.Open(RIG)
    for mesh_name, sset in shapes.items():
        prim = stage.GetPrimAtPath(f"/Astronaut/Geometry/{mesh_name}")
        targets, tokens = [], []
        for shape, off in sset.items():
            keep = np.where(np.linalg.norm(off, axis=1) > 1e-5)[0]
            if not len(keep):
                continue
            bs = UsdSkel.BlendShape.Define(stage, prim.GetPath().AppendChild(shape))
            bs.CreateOffsetsAttr(Vt.Vec3fArray.FromNumpy(off[keep].astype(np.float32)))
            bs.CreatePointIndicesAttr(Vt.IntArray.FromNumpy(keep.astype(np.int32)))
            targets.append(bs.GetPath())
            tokens.append(shape)
        binding = UsdSkel.BindingAPI(prim)
        binding.CreateBlendShapesAttr(tokens)
        binding.CreateBlendShapeTargetsRel().SetTargets(targets)
    skel = stage.GetPrimAtPath("/Astronaut/Skeleton")
    skel.SetCustomDataByKey("astronaut:correctives", json.dumps(dict(
        doc="RBF driver for shoulder correctives: u = unit(ArmLower - ArmUpper) in the Chest world-aligned frame; "
            "phi_j = exp(-(angle(u, dirs_j)/sigma)^2); weights = phi @ coeff (rows: rest + keys; columns: shapes).",
        sides=driver)))
    stage.GetRootLayer().Save()
    print("wrote correctives into", RIG)


if __name__ == "__main__":
    main()
