"""Build the skinned UsdSkel astronaut from source/astronaut.usd.

Output frame: metres, Z-up, facing +X (left = +Y), origin on the ground under the Root joint.
Writes build/astronaut_rig.usd and build/weights.npz.
"""
import os
import sys

import bpy  # noqa: F401  (provides mathutils)
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

sys.path.insert(0, os.path.dirname(__file__))
from skeleton import JOINTS, NON_DEFORMING, ordered, usd_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "source", "astronaut.usd")
OUT = os.path.join(ROOT, "build", "astronaut_rig.usd")

# source mesh -> (part name, binding mode, allowed bone filter)
#   "skin":   distance-to-bone weights over the allowed bones
#   "foot":   rigid boot below the ankle (FootAnkle), blending into LowerLeg up the shaft
#   "rigid":  100% to one joint
#   "rigid_side": 100% to <side><arg> per connected piece (wrist rings, watch, forearm gauge ride on the forearm)
#   "attach": copy weights from the nearest skinned vertex
#   "attach_piecewise": per connected piece, attach if it sits on an arm, else rigid to the given joint
SIDES = ("Left", "Right")
LEG_BONES = ["Hips", "SpineLower"] + [f"{s}{b}" for s in SIDES for b in ("UpperLeg", "LowerLeg", "FootAnkle")]
TORSO_BONES = ["Hips", "SpineLower", "SpineMiddle", "SpineUpper", "Chest"] + [
    f"{s}{b}" for s in SIDES for b in ("Shoulder", "Scapula", "ArmUpper", "ArmLower", "HandWristTwist", "HandWrist")]
HAND_BONES = [f"{s}Hand{b}" for s in SIDES for b in ("Wrist", "WristTwist")] + [
    f"{s}Hand{f}{k}" for s in SIDES for f in ("Thumb", "Index", "Middle", "Ring", "Little")
    for k in ("Metacarpal", "Proximal", "Intermediate", "Distal") if not (f == "Thumb" and k == "Intermediate")]
FOOT_BONES = [f"{s}{b}" for s in SIDES for b in ("LowerLeg", "FootAnkle", "FootSubtalar", "FootTransverse", "FootBall")]

PARTS = {
    "Astronaut_posed1Z": ("suit_legs", "skin", LEG_BONES),
    "Astronaut_posed59": ("suit_torso_arms", "skin", TORSO_BONES),
    "Astronaut_posed57": ("gloves", "skin", HAND_BONES),
    "Astronaut_posed53": ("boots", "foot", None),
    "Astronaut_posed49": ("boot_soles", "foot", None),
    "Astronaut_posed50": ("boot_straps", "foot", None),
    "Astronaut_posed38": ("boot_details", "foot", None),
    "Astronaut_posed58": ("helmet_shell", "rigid", "Chest"),
    "polySurface1": ("helmet_visor", "rigid", "Chest"),
    "polySurface2": ("helmet_visor_assembly", "rigid", "Chest"),
    "Astronaut_posed34": ("helmet_neck_ring", "rigid", "Chest"),
    "Astronaut_posed45": ("plss", "rigid", "Chest"),
    "Astronaut_posed46": ("plss_ops_details", "rigid", "Chest"),
    "Astronaut_posed36": ("plss_antenna_details", "rigid", "Chest"),
    "Astronaut_posed39": ("plss_small_details", "rigid", "Chest"),
    "Astronaut_posed33": ("umbilical_hoses", "rigid", "Chest"),
    "Astronaut_posed44": ("dcm_fittings", "rigid", "Chest"),
    "Astronaut_posed51": ("rcu", "rigid", "Chest"),
    "Astronaut_posed63": ("rcu_details", "rigid", "Chest"),
    "Astronaut_posed60": ("rcu_straps", "rigid", "Chest"),
    "Astronaut_posed56": ("harness_straps_arm_patches", "attach_piecewise", "Chest"),
    "Astronaut_posed62": ("tether", "rigid", "Chest"),
    "Astronaut_posed52": ("hanging_hook", "rigid", "Chest"),
    "Astronaut_posed55": ("wrist_rings", "rigid_side", "ArmLower"),
    "Astronaut_posed17Z": ("wristwatch", "rigid_side", "ArmLower"),
    "Astronaut_posed54": ("glove_fingertips", "attach", None),
    "Astronaut_posed37": ("thigh_pockets", "attach", None),
    "Astronaut_posed48": ("thigh_pocket_flap", "attach", None),
    "Astronaut_posed47": ("waist_band", "attach", None),
    "Astronaut_posed40": ("forearm_gauge_glass", "rigid_side", "ArmLower"),
    "Astronaut_posed41": ("forearm_gauge_ticks", "rigid_side", "ArmLower"),
    "Astronaut_posed42": ("forearm_gauge_body", "rigid_side", "ArmLower"),
    "Astronaut_posed43": ("forearm_gauge_face", "rigid_side", "ArmLower"),
    "Astronaut_posed61": ("wristwatch_glass", "rigid_side", "ArmLower"),
}

# Materials: the source's NVIDIA Omniverse looks (/Root/Looks/<slot>, OmniPBR / OmniGlass MDL; textures in
# source/textures, mirrored from the public Omniverse S3 bucket by scripts/fetch_textures.py) are copied verbatim, and a UsdPreviewSurface network wired to the same textures is added
# beside each MDL shader so renderers without MDL (Blender, Newton, generic Hydra) get the texture maps too.
# Constants below are the preview fallbacks (used directly by the glass looks) and the meshes' displayColor.
LOOKS = {  # slot: (diffuseColor, roughness, metallic, opacity)
    "Body": ((0.80, 0.79, 0.75), 0.80, 0.0, 1.0),
    "Pants": ((0.80, 0.79, 0.75), 0.80, 0.0, 1.0),
    "Gloves": ((0.55, 0.57, 0.60), 0.55, 0.0, 1.0),
    "Boots": ((0.80, 0.79, 0.76), 0.75, 0.0, 1.0),
    "Helmet": ((0.85, 0.85, 0.83), 0.25, 0.0, 1.0),
    "HelmetVisor": ((1.00, 0.77, 0.34), 0.05, 1.0, 1.0),
    "Backpack": ((0.82, 0.81, 0.77), 0.85, 0.0, 1.0),
    "Backpack_top": ((0.82, 0.81, 0.77), 0.70, 0.0, 1.0),
    "Accessories": ((0.70, 0.70, 0.68), 0.50, 0.0, 1.0),
    "WatchGaugeGlass": ((0.90, 0.92, 0.95), 0.02, 0.0, 0.2),
}
PART_LOOKS = {}

ORIGIN_CM = np.array([0.0, -4.0, -0.6])  # Root joint projected onto the sole plane
ROT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(+90): -Y forward -> +X forward
INFLUENCES = 4
FALLOFF = 6.0


def to_out(p_cm):
    return ((np.asarray(p_cm, float) - ORIGIN_CM) * 0.01) @ ROT.T


def bone_segments():
    """Deforming joint -> (a, b) segment in output frame."""
    kids = {n: [c for c, (p, _) in JOINTS.items() if p == n] for n in JOINTS}
    prefer = {"Hips": "SpineLower", "Chest": "Neck", "LeftArmLower": "LeftHandWristTwist",
              "RightArmLower": "RightHandWristTwist", "LeftHandWrist": "LeftHandPalm", "RightHandWrist": "RightHandPalm",
              "LeftLowerLeg": "LeftFootAnkle", "RightLowerLeg": "RightFootAnkle"}
    seg = {}
    for n, (par, p) in JOINTS.items():
        if n in NON_DEFORMING:
            continue
        a = to_out(p)
        if n.endswith("HandWristTwist"):
            b = to_out(JOINTS[n.replace("WristTwist", "Wrist")][1])
        elif n in prefer:
            b = to_out(JOINTS[prefer[n]][1])
        elif kids[n]:
            b = to_out(JOINTS[kids[n][0]][1])
        elif n == "Head":
            b = a + np.array([0.0, 0.0, 0.15])
        else:  # FootBall
            b = a + (a - to_out(JOINTS[par][1])) * 0.8
        seg[n] = (a, b)
    return seg


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def closest_on_seg(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return a + t[:, None] * ab


def skin_weights(P, bones, seg, jidx, bvh, candidates=8, gate=None, adj=None, smooth_iters=4):
    """Inverse-distance weights over the nearest bones that can 'see' the vertex: the straight line
    from the bone to the vertex must not cross the mesh (keeps weights from jumping between the arm
    and torso, or between fingers)."""
    from mathutils import Vector
    C = np.stack([closest_on_seg(P, *seg[b]) for b in bones], 1)  # (n, nb, 3)
    D = np.linalg.norm(P[:, None] - C, axis=2)
    # a vertex clearly nearer one side's bones may not take the other side's (e.g. the inner edge
    # of the right boot sits near the midline but must never follow the left foot)
    left = np.array([b.startswith("Left") for b in bones])
    right = np.array([b.startswith("Right") for b in bones])
    if left.any() and right.any():
        dl, dr = D[:, left].min(1), D[:, right].min(1)
        D[np.ix_(dl < 0.8 * dr, right)] = np.inf
        D[np.ix_(dr < 0.8 * dl, left)] = np.inf
    order = np.argsort(D, 1)[:, :candidates]
    W = np.zeros_like(D)
    for v in range(len(P)):
        pv = Vector(P[v])
        for j in order[v]:
            d = D[v, j]
            if not np.isfinite(d):
                break
            if d > 1e-4:
                o = Vector(C[v, j])
                hit = bvh.ray_cast(o, (pv - o) / d, d - 0.004)[0]
                if hit is not None:
                    continue
            W[v, j] = 1.0 / max(d, 0.003) ** FALLOFF
        if not W[v].any():
            j = order[v, 0]
            W[v, j] = 1.0
    W /= W.sum(1, keepdims=True)
    if gate is not None:
        G = W * gate
        ok = G.sum(1) > 1e-6
        W[ok] = G[ok] / G[ok].sum(1, keepdims=True)
    if adj is not None:  # smooth along mesh edges only (never across touching surfaces)
        for _ in range(smooth_iters):
            W = 0.5 * W + 0.5 * (adj @ W)
            if gate is not None:
                W *= gate > 0
            W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    top = np.argsort(-W, 1)[:, :INFLUENCES]
    w = np.take_along_axis(W, top, 1)
    w /= w.sum(1, keepdims=True)
    idx = np.array([jidx[b] for b in bones])[top]
    return idx.astype(np.int32), w.astype(np.float32)


def edge_graph(P, prim):
    """Sparse edge-length graph and row-normalised adjacency of a mesh."""
    from scipy.sparse import coo_matrix, diags
    m = UsdGeom.Mesh(prim)
    counts = np.array(m.GetFaceVertexCountsAttr().Get())
    fvi = np.array(m.GetFaceVertexIndicesAttr().Get())
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    nxt = np.arange(len(fvi)) + 1
    nxt[np.cumsum(counts) - 1] = starts
    a, b = fvi, fvi[nxt]
    n = len(P)
    L = coo_matrix((np.linalg.norm(P[a] - P[b], axis=1), (a, b)), shape=(n, n)).tocsr()
    L = L.maximum(L.T)
    A = (L > 0).astype(float)
    return L, diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A


def boundary_loops(prim):
    from collections import Counter
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = UsdGeom.Mesh(prim)
    counts = np.array(m.GetFaceVertexCountsAttr().Get())
    fvi = np.array(m.GetFaceVertexIndicesAttr().Get())
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    nxt = np.arange(len(fvi)) + 1
    nxt[np.cumsum(counts) - 1] = starts
    ec = Counter(zip(np.minimum(fvi, fvi[nxt]).tolist(), np.maximum(fvi, fvi[nxt]).tolist()))
    be = np.array([e for e, c in ec.items() if c == 1])
    n = int(fvi.max()) + 1
    _, lab = connected_components(coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(n, n)), directed=False)
    used = np.unique(be)
    return [used[lab[used] == l] for l in np.unique(lab[used])]


def arm_gate(P, prim, L, bones, seg):
    """Gate arm bones on the torso mesh by surface distance from each wrist opening: arm bones may only
    reach as far up the sleeve as the shoulder, so the torso side next to the hanging hand stays on the spine."""
    from scipy.sparse.csgraph import dijkstra
    gate = np.ones((len(P), len(bones)))
    loops = [l for l in boundary_loops(prim) if len(l) > 15]
    for side, sgn in (("Left", 1), ("Right", -1)):
        wrist = max(loops, key=lambda l: sgn * P[l, 1].mean())  # left = +Y
        g = dijkstra(L, directed=False, indices=wrist, min_only=True)
        shoulder = seg[f"{side}ArmUpper"][0]
        near = np.argsort(np.linalg.norm(P - shoulder, axis=1))[:20]
        thr = np.median(g[near])
        ramp = np.clip((thr + 0.05 - g) / 0.10, 0.0, 1.0)  # 1 on the arm, 0 past the shoulder
        for k, b in enumerate(bones):
            if b.startswith(side) and any(t in b for t in ("ArmUpper", "ArmLower", "Hand")):
                gate[:, k] = ramp
    return gate


def author_look(stage, src_layer, slot):
    """Copy the source's MDL look and add a UsdPreviewSurface network on the same textures."""
    from pxr import UsdShade
    path = Sdf.Path(f"/Astronaut/Looks/{slot}")
    src_path = Sdf.Path(f"/Root/Looks/{slot}")
    if src_layer.GetPrimAtPath(src_path):
        Sdf.CopySpec(src_layer, src_path, stage.GetRootLayer(), path)
        # re-anchor the source's relative texture paths (./textures/...) to the rig's own folder
        out_dir = os.path.dirname(stage.GetRootLayer().realPath)
        for prim in stage.GetPrimAtPath(path).GetAllChildren():
            for attr in prim.GetAttributes():
                v = attr.Get()
                if isinstance(v, Sdf.AssetPath) and v.path.startswith("."):
                    attr.Set(Sdf.AssetPath(os.path.relpath(src_layer.ComputeAbsolutePath(v.path), out_dir)))
    mat = UsdShade.Material.Define(stage, path)
    col, rough, metal, opac = LOOKS[slot]
    ps = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    ps.CreateIdAttr("UsdPreviewSurface")
    inp = lambda n, t, v: ps.CreateInput(n, t).Set(v)  # noqa: E731
    inp("diffuseColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*col))
    inp("roughness", Sdf.ValueTypeNames.Float, rough)
    inp("metallic", Sdf.ValueTypeNames.Float, metal)
    inp("opacity", Sdf.ValueTypeNames.Float, opac)
    mat.CreateSurfaceOutput().ConnectToSource(ps.ConnectableAPI(), "surface")
    mdl = stage.GetPrimAtPath(path.AppendChild("Shader"))
    if not mdl or mdl.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() != "OmniPBR":
        return mat  # glass looks keep the constant preview (gold visor / clear watch glass)
    st = UsdShade.Shader.Define(stage, path.AppendChild("stReader"))
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")

    def tex(name, mdl_input, raw, normal=False):
        a = mdl.GetAttribute(f"inputs:{mdl_input}")
        v = a.Get() if a else None
        if not v or not v.path:
            return None
        t = UsdShade.Shader.Define(stage, path.AppendChild(name))
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(v)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        if normal:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        return t

    def link(t, out, channel, typ):
        if t is not None:
            t.CreateOutput(channel, typ)
            ps.CreateInput(out, typ).ConnectToSource(t.ConnectableAPI(), channel)

    link(tex("BaseColorTex", "diffuse_texture", False), "diffuseColor", "rgb", Sdf.ValueTypeNames.Float3)
    link(tex("NormalTex", "normalmap_texture", True, normal=True), "normal", "rgb", Sdf.ValueTypeNames.Float3)
    if mdl.GetAttribute("inputs:enable_ORM_texture").Get():
        orm = tex("ORMTex", "ORM_texture", True)
        link(orm, "occlusion", "r", Sdf.ValueTypeNames.Float)
        link(orm, "roughness", "g", Sdf.ValueTypeNames.Float)
        link(orm, "metallic", "b", Sdf.ValueTypeNames.Float)
    else:
        link(tex("RoughnessTex", "reflectionroughness_texture", True), "roughness", "r", Sdf.ValueTypeNames.Float)
        link(tex("MetallicTex", "metallic_texture", True), "metallic", "r", Sdf.ValueTypeNames.Float)
    return mat


def cap_holes(P, counts, fvi, N, st_idx, max_loop=32, max_radius=0.016):
    """Close small open boundary loops (the open glove fingertip tubes) with one polygon each.
    Winding runs against the neighbouring faces' boundary edges so the cap faces outward; face-varying
    normals get the cap's Newell normal and UVs reuse each corner vertex's existing st index."""
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    nxt = np.arange(len(fvi)) + 1
    nxt[np.cumsum(counts) - 1] = starts
    a, b = fvi, fvi[nxt]
    from collections import Counter
    und = Counter(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))
    cap_next = {}
    for u, v in zip(a.tolist(), b.tolist()):
        if und[(min(u, v), max(u, v))] == 1:
            cap_next[v] = u
    first_corner = {}
    for c, v in enumerate(fvi.tolist()):
        first_corner.setdefault(v, c)
    new_counts, new_fvi, new_n, new_st, seen = [], [], [], [], set()
    for v0 in cap_next:
        if v0 in seen:
            continue
        cyc, v = [], v0
        while v not in seen and v in cap_next and len(cyc) <= max_loop:
            seen.add(v)
            cyc.append(v)
            v = cap_next[v]
        if v != v0 or len(cyc) < 3 or len(cyc) > max_loop:
            continue
        q = P[cyc]
        if np.linalg.norm(q - q.mean(0), axis=1).mean() > max_radius:
            continue
        n = np.cross(q, np.roll(q, -1, 0)).sum(0)  # Newell normal
        n /= max(np.linalg.norm(n), 1e-12)
        new_counts.append(len(cyc))
        new_fvi += cyc
        new_n += [n] * len(cyc)
        if st_idx is not None:
            new_st += [st_idx[first_corner[x]] for x in cyc]
    if not new_counts:
        return counts, fvi, N, st_idx, 0
    counts = np.r_[counts, new_counts]
    fvi = np.r_[fvi, new_fvi]
    if N is not None:
        N = np.r_[N, np.array(new_n)]
    if st_idx is not None:
        st_idx = np.r_[st_idx, new_st]
    return counts, fvi, N, st_idx, len(new_counts)


def make_bvh(P, prim):
    from mathutils.bvhtree import BVHTree
    m = UsdGeom.Mesh(prim)
    counts = np.array(m.GetFaceVertexCountsAttr().Get())
    fvi = np.array(m.GetFaceVertexIndicesAttr().Get()).tolist()
    polys, o = [], 0
    for c in counts:
        polys.append(fvi[o:o + c])
        o += c
    return BVHTree.FromPolygons(P.tolist(), polys, all_triangles=False)


def components(prim):
    """Vertex index arrays of each connected piece of a mesh."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = UsdGeom.Mesh(prim)
    counts = np.array(m.GetFaceVertexCountsAttr().Get())
    fvi = np.array(m.GetFaceVertexIndicesAttr().Get())
    n = len(m.GetPointsAttr().Get())
    first = np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    g = coo_matrix((np.ones(len(fvi)), (fvi, fvi[first])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return [np.where(lab == i)[0] for i in np.unique(lab)]


def main():
    src = Usd.Stage.Open(SRC)
    names = ordered()
    jidx = {n: i for i, n in enumerate(names)}
    seg = bone_segments()

    meshes = {}
    for prim in src.Traverse():
        if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith("/Root/Astronaut/"):  # skip the studio backdrop
            assert prim.GetName() in PARTS, prim.GetName()
            meshes[prim.GetName()] = prim

    data = {}
    for key, (part, mode, arg) in PARTS.items():
        m = UsdGeom.Mesh(meshes[key])
        xf = np.array(UsdGeom.Xformable(meshes[key]).ComputeLocalToWorldTransform(0))
        P = np.array(m.GetPointsAttr().Get(), float)
        P = to_out((np.c_[P, np.ones(len(P))] @ xf)[:, :3])
        data[key] = {"P": P}
        if mode == "skin":
            L, adj = edge_graph(P, meshes[key])
            gate = arm_gate(P, meshes[key], L, arg, seg) if part == "suit_torso_arms" else None
            data[key]["idx"], data[key]["w"] = skin_weights(P, arg, seg, jidx, make_bvh(P, meshes[key]), gate=gate, adj=adj)
        elif mode == "foot":
            idx = np.zeros((len(P), INFLUENCES), np.int32)
            w = np.zeros((len(P), INFLUENCES), np.float32)
            for side in SIDES:
                ankle = to_out(JOINTS[f"{side}FootAnkle"][1])
                other = to_out(JOINTS[("Right" if side == "Left" else "Left") + "FootAnkle"][1])
                sel = np.linalg.norm(P[:, :2] - ankle[:2], axis=1) < np.linalg.norm(P[:, :2] - other[:2], axis=1)
                t = np.clip((P[sel, 2] - (ankle[2] + 0.02)) / 0.08, 0.0, 1.0)
                t = t * t * (3 - 2 * t)
                idx[sel, 0], idx[sel, 1] = jidx[f"{side}FootAnkle"], jidx[f"{side}LowerLeg"]
                w[sel, 0], w[sel, 1] = 1 - t, t
            data[key]["idx"], data[key]["w"] = idx, w
        elif mode == "rigid_side":
            idx = np.zeros((len(P), INFLUENCES), np.int32)
            w = np.zeros((len(P), INFLUENCES), np.float32)
            for comp in components(meshes[key]):
                side = "Left" if P[comp, 1].mean() > 0 else "Right"
                idx[comp, 0], w[comp, 0] = jidx[f"{side}{arg}"], 1.0
            data[key]["idx"], data[key]["w"] = idx, w
        elif mode == "rigid":
            data[key]["idx"] = np.zeros((len(P), INFLUENCES), np.int32)
            data[key]["idx"][:, 0] = jidx[arg]
            data[key]["w"] = np.zeros((len(P), INFLUENCES), np.float32)
            data[key]["w"][:, 0] = 1.0

    from scipy.spatial import cKDTree
    # Layer consistency: where the trouser top tucks under the torso hem, follow the torso's weights
    # so the two layers move together when the spine bends.
    legs, torso = data["Astronaut_posed1Z"], data["Astronaut_posed59"]
    hem = torso["P"][:, 2].min() + 0.06
    over = np.where(legs["P"][:, 2] > hem - 0.04)[0]
    if len(over):
        _, nn = cKDTree(torso["P"]).query(legs["P"][over])
        t = np.clip((legs["P"][over, 2] - (hem - 0.04)) / 0.04, 0.0, 1.0)[:, None]
        for v, n_, tv in zip(over, nn, t[:, 0]):
            acc = {}
            for j, w in zip(legs["idx"][v], legs["w"][v]):
                acc[j] = acc.get(j, 0.0) + (1 - tv) * w
            for j, w in zip(torso["idx"][n_], torso["w"][n_]):
                acc[j] = acc.get(j, 0.0) + tv * w
            top = sorted(acc.items(), key=lambda kv: -kv[1])[:INFLUENCES]
            tot = sum(w for _, w in top)
            legs["idx"][v] = [j for j, _ in top] + [0] * (INFLUENCES - len(top))
            legs["w"][v] = [w / tot for _, w in top] + [0.0] * (INFLUENCES - len(top))
    skinned = [k for k, v in PARTS.items() if v[1] == "skin"]
    SP = np.concatenate([data[k]["P"] for k in skinned])
    SI = np.concatenate([data[k]["idx"] for k in skinned])
    SW = np.concatenate([data[k]["w"] for k in skinned])
    tree = cKDTree(SP)
    body = np.concatenate([np.full(len(data[k]["P"]), PARTS[k][0] != "gloves") for k in skinned])
    body_tree, body_ids = cKDTree(SP[body]), np.where(body)[0]
    for key, (part, mode, arg) in PARTS.items():
        if mode in ("attach", "attach_piecewise"):
            if part.startswith(("thigh", "waist")):  # never follow the hands hanging next to them
                nn = body_ids[body_tree.query(data[key]["P"])[1]]
            else:
                _, nn = tree.query(data[key]["P"])
            data[key]["idx"], data[key]["w"] = SI[nn].copy(), SW[nn].copy()
        if mode == "attach_piecewise":
            arm = {jidx[n] for n in names if any(k in n for k in ("ArmUpper", "ArmLower", "Hand"))}
            for comp in components(meshes[key]):
                _, c = tree.query(data[key]["P"][comp].mean(0))
                if SI[c, np.argmax(SW[c])] not in arm:
                    data[key]["idx"][comp] = 0
                    data[key]["idx"][comp, 0] = jidx[arg]
                    data[key]["w"][comp] = 0.0
                    data[key]["w"][comp, 0] = 1.0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    stage = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdSkel.Root.Define(stage, "/Astronaut")
    root = stage.GetPrimAtPath("/Astronaut")
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind("component")

    skel = UsdSkel.Skeleton.Define(stage, "/Astronaut/Skeleton")
    jpos = {n: to_out(JOINTS[n][1]) for n in names}
    skel.CreateJointsAttr([usd_path(n) for n in names])
    skel.CreateJointNamesAttr(names)
    import joint_frames
    frames = joint_frames.compute(JOINTS, jpos)
    bind = {}
    for n in names:
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = frames[n], jpos[n]
        bind[n] = M
    # USD matrices are row-vector (transposed) relative to the column-vector math here
    skel.CreateBindTransformsAttr(Vt.Matrix4dArray([Gf.Matrix4d(*bind[n].T.ravel().tolist()) for n in names]))
    rest = []
    for n in names:
        par = JOINTS[n][0]
        L = bind[n] if not par else np.linalg.inv(bind[par]) @ bind[n]
        rest.append(Gf.Matrix4d(*L.T.ravel().tolist()))
    skel.CreateRestTransformsAttr(Vt.Matrix4dArray(rest))
    skel.GetPrim().SetCustomDataByKey("jointConvention", "OpenXR XR_FB_body_tracking + XR_META_body_tracking_full_body names; "
                                      "bind orientations follow XR_EXT_hand_tracking axes (+Z along the bone away from the child, +Y dorsal); "
                                      "see scripts/joint_frames.py")

    from pxr import UsdShade
    looks = {}
    UsdGeom.Scope.Define(stage, "/Astronaut/Looks")
    for slot in LOOKS:
        looks[slot] = author_look(stage, src.GetRootLayer(), slot)

    UsdGeom.Scope.Define(stage, "/Astronaut/Geometry")
    for key, (part, mode, _) in PARTS.items():
        sm = UsdGeom.Mesh(meshes[key])
        mesh = UsdGeom.Mesh.Define(stage, f"/Astronaut/Geometry/{part}")
        P = data[key]["P"]
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
        counts = np.array(sm.GetFaceVertexCountsAttr().Get())
        fvi = np.array(sm.GetFaceVertexIndicesAttr().Get())
        N = sm.GetNormalsAttr().Get()
        N = None if N is None else np.array(N, float) @ ROT.T
        st = UsdGeom.PrimvarsAPI(sm).GetPrimvar("st")
        st_idx = np.array(st.GetIndices()) if st and st.IsIndexed() else None
        if part == "gloves":
            counts, fvi, N, st_idx, ncap = cap_holes(P, counts, fvi, N, st_idx)
            print(f"capped {ncap} open fingertips")
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts.astype(np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(fvi.astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr(sm.GetSubdivisionSchemeAttr().Get() or UsdGeom.Tokens.none)
        if sm.GetOrientationAttr().HasAuthoredValue():
            mesh.CreateOrientationAttr(sm.GetOrientationAttr().Get())
        if N is not None:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.astype(np.float32)))
            mesh.SetNormalsInterpolation(sm.GetNormalsInterpolation())
        lo, hi = P.min(0), P.max(0)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
        if st:
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, st.GetInterpolation())
            pv.Set(st.Get())
            if st_idx is not None:
                pv.SetIndices(Vt.IntArray.FromNumpy(st_idx.astype(np.int32)))
        binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
        binding.CreateSkeletonRel().SetTargets([skel.GetPath()])
        binding.CreateJointIndicesPrimvar(False, INFLUENCES).Set(Vt.IntArray.FromNumpy(data[key]["idx"].ravel()))
        binding.CreateJointWeightsPrimvar(False, INFLUENCES).Set(Vt.FloatArray.FromNumpy(data[key]["w"].ravel()))
        binding.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
        slot = PART_LOOKS.get(part) or meshes[key].GetRelationship("material:binding").GetTargets()[0].name
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(looks[slot])
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*LOOKS[slot][0])])
        mesh.GetPrim().SetCustomDataByKey("sourceMesh", key)
        mesh.GetPrim().SetCustomDataByKey("skinning", mode)
    # bind on the Geometry scope (not the SkelRoot) so physics colliders added under the root are not skinned
    UsdSkel.BindingAPI.Apply(stage.GetPrimAtPath("/Astronaut/Geometry")).CreateSkeletonRel().SetTargets([skel.GetPath()])
    stage.GetRootLayer().Save()

    np.savez_compressed(os.path.join(ROOT, "build", "weights.npz"), joints=np.array(names),
                        **{f"{PARTS[k][0]}__idx": v["idx"] for k, v in data.items()},
                        **{f"{PARTS[k][0]}__w": v["w"] for k, v in data.items()})
    print("wrote", OUT, len(names), "joints", sum(len(v["P"]) for v in data.values()), "verts")


if __name__ == "__main__":
    main()
