"""Write astronaut motion as a UsdSkel animation that carries the shoulder correctives.

The asset's corrective blend shapes are driven by an RBF on the upper-arm direction (Rig.corrective_weights; the
driver data is in the skeleton's customData["astronaut:correctives"]). UsdSkel has no drivers, so a plain joint
animation would play back without them. This writer evaluates the RBF for every frame and authors the weights as
the animation's blendShapeWeights, so any UsdSkel consumer (Omniverse, Isaac Sim, usdview, Hydra) reproduces the
skinning of the Newton examples and the Cycles renders.

    from skel_animation import write_animation
    write_animation(rig, frames, "build/walk_anim.usda", fps=30)   # frames: list of (n_joints, 4, 4) joint world matrices

For a live stream (OpenXR body tracking into your own UsdSkel animation), do the same per frame: W = joint world
matrices, weights = rig.corrective_weights(W), and write them to blendShapeWeights in the order of corrective_names().
"""
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(ROOT, "assets", "astronaut", "astronaut.usda")


def corrective_names(rig):
    """Every corrective blend-shape name, in a fixed order (the animation's blendShapes)."""
    return [nm for d in rig.correctives.values() for nm in d["shapes"]]


def write_animation(rig, frames, path, fps, asset=ASSET, physics="none"):
    """frames: sequence of joint world matrices (n_joints, 4, 4). Writes a layer that references the asset, selects
    the given Physics variant (default "none": animation only) and binds a SkelAnimation with joint transforms and
    the corrective blend-shape weights."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)
    root = stage.DefinePrim("/astronaut")
    root.GetReferences().AddReference(os.path.relpath(asset, os.path.dirname(os.path.abspath(path))))
    root.GetVariantSet("Physics").SetVariantSelection(physics)
    stage.SetDefaultPrim(root)
    skel_path = "/astronaut/Skeleton"
    anim = UsdSkel.Animation.Define(stage, f"{skel_path}/Animation")
    anim.CreateJointsAttr(UsdSkel.Skeleton(stage.GetPrimAtPath(skel_path)).GetJointsAttr().Get())
    names = corrective_names(rig)
    anim.CreateBlendShapesAttr(names)
    for f, W in enumerate(frames):
        L = rig.local_from_world(W)
        q = R.from_matrix(L[:, :3, :3]).as_quat()  # xyzw
        anim.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(L[:, :3, 3].astype(np.float32)), f)
        anim.GetRotationsAttr().Set(Vt.QuatfArray([Gf.Quatf(float(w), float(x), float(y), float(z)) for x, y, z, w in q]), f)
        w = rig.corrective_weights(W)
        anim.GetBlendShapeWeightsAttr().Set(Vt.FloatArray([w.get(nm, 0.0) for nm in names]), f)
    anim.GetScalesAttr().Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(rig.names)))
    UsdSkel.BindingAPI.Apply(stage.GetPrimAtPath(skel_path)).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    stage.GetRootLayer().Save()
    return path
