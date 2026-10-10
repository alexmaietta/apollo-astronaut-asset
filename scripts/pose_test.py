"""Pose the rig into test poses and write baked meshes for rendering: build/pose_<name>.usd"""
import os, sys
import numpy as np
from pxr import Usd, UsdGeom, Vt
sys.path.insert(0, os.path.dirname(__file__))
from rig import Rig, rot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
fingers = ["Index", "Middle", "Ring", "Little"]


def fist(side, deg):
    s = 1 if side == "Left" else -1
    d = {}
    for f in fingers:
        for k, a in (("Proximal", deg), ("Intermediate", deg), ("Distal", deg * 0.7)):
            d[f"{side}Hand{f}{k}"] = rot(X, -s * a)
    d[f"{side}HandThumbProximal"] = rot(Z, s * deg * 0.4)
    return d


POSES = {
    "seated": {**{f"{s}UpperLeg": rot(Y, -85) for s in ("Left", "Right")},
               **{f"{s}LowerLeg": rot(Y, 80) for s in ("Left", "Right")},
               "LeftArmUpper": rot(Y, -20), "LeftArmLower": rot(Y, -70),
               "RightArmUpper": rot(Y, -20), "RightArmLower": rot(Y, -70), **fist("Right", 70)},
    "stride": {"LeftUpperLeg": rot(Y, -35), "LeftLowerLeg": rot(Y, 30), "RightUpperLeg": rot(Y, 20),
               "RightLowerLeg": rot(Y, 40), "RightFootAnkle": rot(Y, -15),
               "LeftArmUpper": rot(Y, 25), "RightArmUpper": rot(Y, -30), "RightArmLower": rot(Y, -40),
               "SpineMiddle": rot(Z, 8), "Chest": rot(Y, -8)},
    "reach": {"RightArmUpper": rot(X, 80) @ rot(Y, -40), "RightArmLower": rot(Y, -30),
              "LeftArmUpper": rot(Y, -100), **fist("Left", 80), "SpineUpper": rot(Y, -15),
              "LeftUpperLeg": rot(X, -20)},
}

rig = Rig(os.path.join(ROOT, "build", "astronaut_rig.usd"))
for name, pose in POSES.items():
    W = rig.fk(pose)
    pts = rig.skin(W)
    path = os.path.join(ROOT, "build", f"pose_{name}.usd")
    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, "Z"); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    for m, P in zip(rig.meshes, pts):
        g = UsdGeom.Mesh.Define(st, f"/Posed/{m['name']}")
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
        g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(m["counts"]))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m["fvi"]))
        g.CreateSubdivisionSchemeAttr("none")
    st.GetRootLayer().Save()
    print("wrote", path)
