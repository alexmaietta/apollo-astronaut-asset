"""Package the astronaut as a SimReady / Isaac-structure USD asset with a Physics variant set.

    .venv/bin/python scripts/package_simready.py      # -> build/asset/astronaut/

Layout (the Isaac Sim robot-asset structure profile, as in the Apollo LRV asset):

    astronaut/astronaut.usda              interface: def SkelRoot "astronaut", variantSet Physics = none|physics|mujoco
    astronaut/payloads/base.usda          references geometries.usd + materials.usda
    astronaut/payloads/geometries.usd     skinned rig (Skeleton, meshes, blend shapes), crate
    astronaut/payloads/materials.usda     NVIDIA's MDL + UsdPreviewSurface looks, textures in ./Textures
    astronaut/payloads/Textures/          hard links to source/textures
    astronaut/payloads/Physics/physics.usda   hands: rigid bodies, capsule/box colliders, revolute joints,
                                              PhysicsDriveAPI servos clipped to human finger torque, wrist mounts
    astronaut/payloads/Physics/mujoco.usda    sublayers physics.usda; A7L glove passive springs (mjc:stiffness,
                                              mjc:springref), damping, armature; MjcActuator servos replace the drives
    astronaut/payloads/BodyColliders/capsules.usda   variant set BodyColliders = none|capsules: kinematic suit
                                              proxies (torso, helmet, PLSS, arms, legs) on pose-driven mounts

Physics content: two 20-DoF hand articulations (scripts/hand_physics.py) whose roots are fixed "wrist mount"
joints to the world; a teleop loop moves the mount (the tracked wrist pose) and drives the finger joints.
Angles are degrees, drive stiffness N*m/deg (UsdPhysics); mjc:stiffness is N*m/rad. Glove parameters come from
data/suit_params.json (pressure-scaled).
"""
import os
import shutil
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_physics import SIDES, HandModel, load_params  # noqa: E402
from rig import Rig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIG = os.path.join(ROOT, "build", "astronaut_rig.usd")
OUT = os.path.join(ROOT, "build", "asset", "astronaut")
NAME = "astronaut"
VERSION = open(os.path.join(ROOT, "VERSION")).read().strip()  # release version, stamped into the asset


def new_layer(path):
    layer = Sdf.Layer.CreateNew(path)
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    layer.defaultPrim = NAME
    return layer, stage


def quat(Rm):
    x, y, z, w = R.from_matrix(Rm).as_quat()
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def joint_frame(axis):
    """Rotation whose +X column is axis."""
    x = axis / np.linalg.norm(axis)
    t = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(t, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def flatten_subdivision(layer):
    """NVIDIA's suit, boots and hoses are catmullClark meshes that also carry authored normals. Every renderer this
    asset targets draws them unrefined (no refinementLevel; Newton GL and Blender's importer ignore subdivision) with
    those normals, so author subdivisionScheme = none: the look is unchanged and the normals become valid
    (NormalsExistChecker / NormalsValidChecker, VG.027 / VG.028)."""
    def visit(spec):
        if spec.typeName == "Mesh":
            a = spec.attributes.get("subdivisionScheme")
            if a is not None and a.default != UsdGeom.Tokens.none and (
                    "normals" in spec.attributes or "primvars:normals" in spec.attributes):
                a.default = UsdGeom.Tokens.none
        for c in spec.nameChildren:
            visit(c)
    for root in layer.rootPrims:
        visit(root)


def fix_preview_normal_inputs(stage):
    """UsdPreviewSurface inputs:normal is a normal3f (Sdr); the rig authored float3. Re-type it, keeping the
    connection to the normal-map texture (MaterialUsdPreviewSurfaceChecker, UsdShadeShaderSdrCompliance, VM.PS.001)."""
    for prim in stage.Traverse():
        sh = UsdShade.Shader(prim)
        if not sh or sh.GetShaderId() != "UsdPreviewSurface":
            continue
        inp = sh.GetInput("normal")
        if not inp or inp.GetTypeName() == Sdf.ValueTypeNames.Normal3f:
            continue
        sources = inp.GetConnectedSources()[0]
        value = inp.Get()
        prim.RemoveProperty(inp.GetFullName())
        new = sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)
        if value is not None:
            new.Set(Gf.Vec3f(value))
        for src in sources:
            new.ConnectToSource(src)


def write_geometry_and_materials(payloads):
    src = Sdf.Layer.FindOrOpen(RIG)
    tex_dir = os.path.join(payloads, "Textures")
    # geometries.usd: the skinned rig minus its looks, root renamed /Astronaut -> /astronaut
    geo, _ = new_layer(os.path.join(payloads, "geometries.usd"))
    Sdf.CopySpec(src, "/Astronaut", geo, "/" + NAME)
    del geo.GetPrimAtPath(f"/{NAME}").nameChildren["Looks"]
    flatten_subdivision(geo)
    geo.Save()
    # materials.usda: the looks under /Looks, texture paths -> ./Textures/...
    mat, mstage = new_layer(os.path.join(payloads, "materials.usda"))
    Sdf.CopySpec(src, "/Astronaut/Looks", mat, "/Looks")
    mat.defaultPrim = "Looks"
    src_dir = os.path.dirname(src.realPath)
    for prim in mstage.Traverse():
        for attr in prim.GetAttributes():
            v = attr.Get()
            if isinstance(v, Sdf.AssetPath) and v.path.startswith("."):
                abs_path = os.path.normpath(os.path.join(src_dir, v.path))
                rel = os.path.relpath(abs_path, os.path.join(ROOT, "source", "textures"))
                dst = os.path.join(tex_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    try:
                        os.link(abs_path, dst)
                    except OSError:
                        shutil.copyfile(abs_path, dst)
                attr.Set(Sdf.AssetPath("./Textures/" + rel.replace(os.sep, "/")))
    fix_preview_normal_inputs(mstage)
    mat.Save()
    # base.usda
    base, bstage = new_layer(os.path.join(payloads, "base.usda"))
    root = bstage.DefinePrim(f"/{NAME}", "SkelRoot")
    root.GetReferences().AddReference("./geometries.usd", f"/{NAME}")
    Usd.ModelAPI(root).SetKind("component")
    Usd.ModelAPI(root).SetAssetInfo({"name": NAME, "version": VERSION})
    looks = bstage.DefinePrim(f"/{NAME}/Looks", "Scope")
    looks.GetReferences().AddReference("./materials.usda", "/Looks")
    base.Save()


def write_physics(payloads, rig, params):
    phys_dir = os.path.join(payloads, "Physics")
    os.makedirs(phys_dir, exist_ok=True)
    layer, st = new_layer(os.path.join(phys_dir, "physics.usda"))
    st.OverridePrim(f"/{NAME}")
    scope = f"/{NAME}/Physics"
    st.DefinePrim(scope, "Scope")
    c = params["contact"]
    mats = {}
    for mname, mu_s in (("GloveMaterial", c["static_friction"]), ("FingertipMaterial", c["fingertip_friction"])):
        m = UsdShade.Material.Define(st, f"{scope}/{mname}")
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(mu_s)
        api.CreateDynamicFrictionAttr(c["dynamic_friction"] * mu_s / c["static_friction"])
        api.CreateRestitutionAttr(0.0)
        p = m.GetPrim()
        p.AddAppliedSchema("NewtonMaterialAPI")
        p.CreateAttribute("newton:contactStiffness", Sdf.ValueTypeNames.Float).Set(c["contact_stiffness"])
        p.CreateAttribute("newton:contactDamping", Sdf.ValueTypeNames.Float).Set(c["contact_damping"])
        p.CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(c["torsional_friction"])
        p.CreateAttribute("newton:rollingFriction", Sdf.ValueTypeNames.Float).Set(c["rolling_friction"])
        mats[mname] = m
    hands = {}
    drive = params["drive"]
    for side in SIDES:
        hand = HandModel(rig, side, params)
        hands[side] = hand
        hscope = f"/{NAME}/Hands/{side}"
        UsdGeom.Scope.Define(st, f"/{NAME}/Hands")
        UsdGeom.Scope.Define(st, hscope)
        paths = {}
        for b in hand.bodies:
            path = f"{hscope}/{b['name']}"
            paths[b["name"]] = path
            x = UsdGeom.Xform.Define(st, path)
            x.AddTranslateOp().Set(Gf.Vec3d(*b["p"]))
            x.AddOrientOp().Set(quat(b["R"]))
            prim = x.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mapi = UsdPhysics.MassAPI.Apply(prim)
            mapi.CreateMassAttr(b["mass"])
            if not b["colliders"]:  # helper (spread) bodies: explicit small-sphere inertia
                i = 0.4 * b["mass"] * 0.005 ** 2
                mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(i, i, i))
                mapi.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))  # identity: same tensor, both authored (MassChecker)
                mapi.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
            if b["parent"] is None:
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                prim.AddAppliedSchema("NewtonArticulationRootAPI")
                prim.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
            if b["skin_joint"]:
                prim.SetCustomDataByKey("astronaut:skinJoint", b["skin_joint"])
            for k, col in enumerate(b["colliders"]):
                if col["type"] == "capsule":
                    a, e = col["a"], col["b"]
                    L = float(np.linalg.norm(e - a))
                    cap = UsdGeom.Capsule.Define(st, f"{path}/Collider{k}")
                    cap.CreateAxisAttr("Z")
                    cap.CreateRadiusAttr(col["r"])
                    cap.CreateHeightAttr(L)
                    Rw = joint_frame(e - a)[:, [1, 2, 0]]  # +Z along the segment
                    loc_p = b["R"].T @ ((a + e) / 2 - b["p"])
                    loc_R = b["R"].T @ Rw
                    gprim = cap
                else:
                    cube = UsdGeom.Cube.Define(st, f"{path}/Collider{k}")
                    cube.CreateSizeAttr(2.0)
                    loc_p = b["R"].T @ (col["center"] - b["p"])
                    loc_R = b["R"].T @ col["R"]
                    gprim = cube
                gprim.AddTranslateOp().Set(Gf.Vec3d(*loc_p))
                gprim.AddOrientOp().Set(quat(loc_R))
                if col["type"] == "box":
                    gprim.AddScaleOp().Set(Gf.Vec3f(*col["half"]))
                gprim.CreatePurposeAttr("guide")
                cp = gprim.GetPrim()
                UsdPhysics.CollisionAPI.Apply(cp)
                cp.AddAppliedSchema("NewtonCollisionAPI")
                tip = b["name"].endswith("Distal")
                UsdShade.MaterialBindingAPI.Apply(cp).Bind(
                    mats["FingertipMaterial" if tip else "GloveMaterial"], UsdShade.Tokens.weakerThanDescendants, "physics")
        # wrist mount: fixed joint to the world at the bind wrist pose (teleop moves it)
        w = hand.bodies[0]
        mount = UsdPhysics.FixedJoint.Define(st, f"{scope}/{side}WristMount")
        mount.CreateBody1Rel().SetTargets([paths[w["name"]]])
        mount.CreateLocalPos0Attr(Gf.Vec3f(*w["p"]))
        mount.CreateLocalRot0Attr(quat(w["R"]))
        mount.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        mount.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        bodies = {b["name"]: b for b in hand.bodies}
        for j in hand.joints:
            pb, cb = bodies[j["parent"]], bodies[j["child"]]
            Q = joint_frame(j["axis"])
            rj = UsdPhysics.RevoluteJoint.Define(st, f"{scope}/{j['name']}")
            rj.CreateBody0Rel().SetTargets([paths[pb["name"]]])
            rj.CreateBody1Rel().SetTargets([paths[cb["name"]]])
            rj.CreateAxisAttr("X")
            rj.CreateLocalPos0Attr(Gf.Vec3f(*(pb["R"].T @ (j["anchor"] - pb["p"]))))
            rj.CreateLocalRot0Attr(quat(pb["R"].T @ Q))
            rj.CreateLocalPos1Attr(Gf.Vec3f(*(cb["R"].T @ (j["anchor"] - cb["p"]))))
            rj.CreateLocalRot1Attr(quat(cb["R"].T @ Q))
            rj.CreateLowerLimitAttr(j["lower_deg"])
            rj.CreateUpperLimitAttr(j["upper_deg"])
            jp = rj.GetPrim()
            jp.SetCustomDataByKey("astronaut:kind", j["kind"])
            jp.SetCustomDataByKey("astronaut:bindDeg", j["bind_deg"])
            d = UsdPhysics.DriveAPI.Apply(jp, "angular")
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(drive["kp_nm_per_rad"] * np.pi / 180.0)  # N*m/deg
            d.CreateDampingAttr(drive["kd_nms_per_rad"] * np.pi / 180.0)
            d.CreateMaxForceAttr(params["joints"][j["kind"]]["max_torque_nm"])
            d.CreateTargetPositionAttr(0.0)
            jp.AddAppliedSchema("PhysicsJointStateAPI:angular")  # PhysX schema, authored by name
            jp.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(0.0)
            jp.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
            jp.AddAppliedSchema("NewtonJointAPI")
            k = params["joints"][j["kind"]]
            jp.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(params["armature"])
            jp.CreateAttribute("newton:damping", Sdf.ValueTypeNames.Float).Set(  # USD convention: per degree
                k["suit_damping_nms_per_rad"] * np.pi / 180.0)
    layer.Save()

    # mujoco.usda: glove springs + MjcActuator servos instead of the drives
    mj, mst = new_layer(os.path.join(phys_dir, "mujoco.usda"))
    mj.subLayerPaths.append("./physics.usda")
    scale = params["pressure_psi"] / params["reference_pressure_psi"]
    for side, hand in hands.items():
        for j in hand.joints:
            jp = mst.OverridePrim(f"{scope}/{j['name']}")
            jp.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(
                prependedItems=["MjcJointAPI"], deletedItems=["PhysicsDriveAPI:angular", "PhysicsJointStateAPI:angular"]))
            k = params["joints"][j["kind"]]
            jp.CreateAttribute("mjc:stiffness", Sdf.ValueTypeNames.Double).Set(k["suit_stiffness_nm_per_rad"] * scale)
            jp.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Double).Set(k["suit_damping_nms_per_rad"] * max(scale, 0.25))
            jp.CreateAttribute("mjc:armature", Sdf.ValueTypeNames.Double).Set(params["armature"])
            neutral = 0.0 if params.get("glove_neutral", "bind") == "bind" else -j["bind_deg"]
            jp.CreateAttribute("mjc:springref", Sdf.ValueTypeNames.Double).Set(neutral)  # degrees, like the limits
            act = mst.DefinePrim(f"{scope}/{j['name']}_servo", "MjcActuator")
            act.CreateRelationship("mjc:target").SetTargets([f"{scope}/{j['name']}"])
            kp, kd = params["drive"]["kp_nm_per_rad"], params["drive"]["kd_nms_per_rad"]
            act.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.Token).Set("fixed")
            act.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.DoubleArray).Set([kp] + [0.0] * 9)
            act.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.Token).Set("affine")
            act.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.DoubleArray).Set([0.0, -kp, -kd] + [0.0] * 7)
            act.CreateAttribute("mjc:forceLimited", Sdf.ValueTypeNames.Token).Set("true")
            act.CreateAttribute("mjc:forceRange:min", Sdf.ValueTypeNames.Double).Set(-k["max_torque_nm"])
            act.CreateAttribute("mjc:forceRange:max", Sdf.ValueTypeNames.Double).Set(k["max_torque_nm"])
    mj.Save()
    return hands


# Body collision proxies (rig frame, bind pose): (name, skeleton joint it rides on, shape, params)
#   capsule: (a, b, radius) with a/b joint names or fixed bind-space points; sphere: (centre, radius); box: (centre, half)
def body_proxies(rig):
    from poses import HELMET_SPHERE, PLSS_BOX, TORSO_CAPSULES
    P = lambda n: rig.bind[rig.index[n], :3, 3]  # noqa: E731
    out = [(f"Torso{i}", "Chest", "capsule", (np.array(a), np.array(b), r)) for i, (a, b, r) in enumerate(TORSO_CAPSULES)]
    out.append(("Helmet", "Chest", "sphere", (np.array(HELMET_SPHERE[0]), HELMET_SPHERE[1])))
    out.append(("PLSS", "Chest", "box", (np.array(PLSS_BOX[0]), np.array(PLSS_BOX[1]))))
    for s in ("Left", "Right"):
        w = P(f"{s}HandWrist")
        e = P(f"{s}ArmLower")
        out.append((f"{s}UpperArm", f"{s}ArmUpper", "capsule", (P(f"{s}ArmUpper"), e, 0.08)))
        # forearm stops short of the wrist so it does not overlap the simulated hand's palm
        out.append((f"{s}Forearm", f"{s}ArmLower", "capsule", (e, w + (e - w) / np.linalg.norm(e - w) * 0.07, 0.065)))
        out.append((f"{s}Thigh", f"{s}UpperLeg", "capsule", (P(f"{s}UpperLeg"), P(f"{s}LowerLeg"), 0.09)))
        out.append((f"{s}Shin", f"{s}LowerLeg", "capsule", (P(f"{s}LowerLeg"), P(f"{s}FootAnkle"), 0.075)))
        out.append((f"{s}Foot", f"{s}FootAnkle", "capsule", (P(f"{s}FootAnkle"), P(f"{s}FootBall"), 0.06)))
    return out


def write_body_colliders(payloads, rig):
    """payloads/BodyColliders/capsules.usda: kinematic body proxies. Each is a rigid body on a fixed mount joint to
    the world at its skeleton joint's frame (customData astronaut:skinJoint), so a pose driver moves the mount to the
    joint's world matrix, as teleop does for the wrists. Welded-to-world bodies do not collide with each other."""
    d = os.path.join(payloads, "BodyColliders")
    os.makedirs(d, exist_ok=True)
    layer, st = new_layer(os.path.join(d, "capsules.usda"))
    st.OverridePrim(f"/{NAME}")
    scope = f"/{NAME}/BodyColliders"
    UsdGeom.Scope.Define(st, scope)
    mat = UsdShade.Material.Define(st, f"{scope}/SuitMaterial")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.8)
    api.CreateDynamicFrictionAttr(0.7)
    for name, joint, shape, prm in body_proxies(rig):
        Rj, pj = rig.bind[rig.index[joint], :3, :3], rig.bind[rig.index[joint], :3, 3]
        x = UsdGeom.Xform.Define(st, f"{scope}/{name}")
        x.AddTranslateOp().Set(Gf.Vec3d(*pj))
        x.AddOrientOp().Set(quat(Rj))
        prim = x.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
        prim.SetCustomDataByKey("astronaut:skinJoint", joint)
        # the palms ride on rigid teleop mounts like these proxies: contacts between two pose-driven bodies cannot
        # be resolved (neither can move), so filter them; the fingers still collide with the body
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
            [f"/{NAME}/Hands/{s}/{s}HandWrist" for s in ("Left", "Right")])
        if shape == "capsule":
            a, b, r = prm
            g = UsdGeom.Capsule.Define(st, f"{scope}/{name}/Collider")
            g.CreateAxisAttr("Z")
            g.CreateRadiusAttr(r)
            g.CreateHeightAttr(float(np.linalg.norm(b - a)))
            loc_p, loc_R = Rj.T @ ((a + b) / 2 - pj), Rj.T @ joint_frame(b - a)[:, [1, 2, 0]]
        elif shape == "sphere":
            c, r = prm
            g = UsdGeom.Sphere.Define(st, f"{scope}/{name}/Collider")
            g.CreateRadiusAttr(r)
            loc_p, loc_R = Rj.T @ (c - pj), np.eye(3)
        else:
            c, half = prm
            g = UsdGeom.Cube.Define(st, f"{scope}/{name}/Collider")
            g.CreateSizeAttr(2.0)
            loc_p, loc_R = Rj.T @ (c - pj), Rj.T
        g.AddTranslateOp().Set(Gf.Vec3d(*loc_p))
        g.AddOrientOp().Set(quat(loc_R))
        if shape == "box":
            g.AddScaleOp().Set(Gf.Vec3f(*prm[1]))
        g.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        g.GetPrim().AddAppliedSchema("NewtonCollisionAPI")
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        mount = UsdPhysics.FixedJoint.Define(st, f"{scope}/{name}Mount")
        mount.CreateBody1Rel().SetTargets([f"{scope}/{name}"])
        mount.CreateLocalPos0Attr(Gf.Vec3f(*pj))
        mount.CreateLocalRot0Attr(quat(Rj))
        mount.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        mount.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        mount.GetPrim().SetCustomDataByKey("astronaut:skinJoint", joint)
    layer.Save()


def write_interface(out, default_physics="mujoco"):
    layer, st = new_layer(os.path.join(out, f"{NAME}.usda"))
    root = st.DefinePrim(f"/{NAME}", "SkelRoot")
    root.GetReferences().AddReference("./payloads/base.usda")
    vs = root.GetVariantSets().AddVariantSet("Physics")
    for v in ("none", "physics", "mujoco"):
        vs.AddVariant(v)
        if v != "none":
            vs.SetVariantSelection(v)
            with vs.GetVariantEditContext():
                root.GetPayloads().AddPayload(f"./payloads/Physics/{v}.usda")
    vs.SetVariantSelection(default_physics)
    bc = root.GetVariantSets().AddVariantSet("BodyColliders")
    for v in ("none", "capsules"):
        bc.AddVariant(v)
        if v != "none":
            bc.SetVariantSelection(v)
            with bc.GetVariantEditContext():
                root.GetPayloads().AddPayload(f"./payloads/BodyColliders/{v}.usda")
    bc.SetVariantSelection("capsules")
    Usd.ModelAPI(root).SetAssetInfo({"name": NAME, "version": VERSION})
    layer.Save()


def write_metadata(out):
    """SimReady provenance in the interface's customLayerData (SR.001 top-level keys, SR.003/SR.004 nested
    SimReady_Metadata). Rigid-body count, mass and extents come from the composed asset with its default variants."""
    import datetime
    path = os.path.join(out, f"{NAME}.usda")
    stage = Usd.Stage.Open(path)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    mass = sum(float(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0.0) for p in bodies)
    vis = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    # the meshes in bind pose (the SkelRoot bound itself is a padded skinning estimate)
    size = vis.ComputeWorldBound(stage.GetPrimAtPath(f"/{NAME}/Geometry")).ComputeAlignedRange().GetSize()
    today = datetime.date.today().isoformat()
    source_file = "build/astronaut_rig.usd (scripts/build_rig.py, scripts/correctives.py)"
    usd_version = ".".join(str(x) for x in Usd.GetVersion())
    meta = {
        "name": NAME, "description": "Rigged Apollo 11 A7L astronaut (NVIDIA Omniverse Astronaut sample): UsdSkel with "
        "OpenXR joints, shoulder correctives, and 20-DoF physics hands with A7L glove stiffness.",
        "source": "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Samples/Examples/2024_1/Rendering/Astronaut/",
        "author": "Alex Maietta", "asset_name": NAME, "asset_type": "character", "category": "humanoid",
        "asset_license": "CC-BY-4.0 (rig, physics, additions); NVIDIA's Omniverse Astronaut mesh and textures under "
        "NVIDIA's terms", "source_file": source_file, "usd_date_generated": today,
        "qcode": "Q11631",  # Wikidata "astronaut"
        "rigid_body_count": len(bodies), "asset_extents": Gf.Vec3f(*size), "mass": round(mass, 4),  # simulated bodies only
        "asset_version": VERSION, "usd_version": usd_version,
    }
    layer = stage.GetRootLayer()
    layer.customLayerData = {"creator": "astronaut pipeline (scripts/package_simready.py)", "asset_name": NAME,
                             "asset_type": meta["asset_type"], "asset_version": VERSION, "usd_version": usd_version,
                             "source_file": source_file, "usd_date_generated": today, "SimReady_Metadata": meta}
    layer.Save()


def write_thumbnail(out, image):
    """Isaac-style asset thumbnail: .thumbs/256x256/<name>.usda.png (square centre crop of a hero render)."""
    from PIL import Image
    with Image.open(image) as im:
        w, h = im.size
        s = min(w, h)
        im = im.convert("RGB").crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((256, 256), Image.LANCZOS)
        d = os.path.join(out, ".thumbs", "256x256")
        os.makedirs(d, exist_ok=True)
        im.save(os.path.join(d, f"{NAME}.usda.png"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbnail", default=None,
                    help="image for .thumbs/256x256 (default: docs/images/thumbnail.jpg, else renders/hero/stand.png)")
    a = ap.parse_args()
    params = load_params()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    payloads = os.path.join(OUT, "payloads")
    os.makedirs(payloads)
    rig = Rig(RIG)
    write_geometry_and_materials(payloads)
    hands = write_physics(payloads, rig, params)
    write_body_colliders(payloads, rig)
    write_interface(OUT)
    write_metadata(OUT)
    thumbs = [a.thumbnail] if a.thumbnail else [os.path.join(ROOT, "docs", "images", "thumbnail.jpg"),
                                                os.path.join(ROOT, "renders", "hero", "stand.png")]
    thumb = next((t for t in thumbs if os.path.exists(t)), None)
    if thumb:
        write_thumbnail(OUT, thumb)
    else:  # SimReady SR.002 needs .thumbs/256x256/astronaut.usda.png
        print("warning: no thumbnail image (tried", ", ".join(thumbs) + ")", file=sys.stderr)
    print("wrote", OUT, "|", ", ".join(f"{s}: {len(h.bodies)} bodies / {len(h.joints)} DoF" for s, h in hands.items()))


if __name__ == "__main__":
    main()
