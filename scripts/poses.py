"""Pose library (Apollo-style hero poses) and baker: build/hero_<pose>.usd (posed, materials kept).

Poses are a few spine/hip rotations plus two-bone IK targets for hands and feet, in the rig frame
(metres, X forward, Y left, Z up). Feet are kept flat unless a pose gives a world foot rotation; the soles are dropped onto z = `ground`.
"""
import os
import sys

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

sys.path.insert(0, os.path.dirname(__file__))
from rig import Rig, open_render_stage, rot  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
FWD, BACK, OUT_L, OUT_R = (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)

LIMBS = {
    "LeftArm": ("LeftArmUpper", "LeftArmLower", "LeftHandWrist"),
    "RightArm": ("RightArmUpper", "RightArmLower", "RightHandWrist"),
    "LeftLeg": ("LeftUpperLeg", "LeftLowerLeg", "LeftFootAnkle"),
    "RightLeg": ("RightUpperLeg", "RightLowerLeg", "RightFootAnkle"),
}

POSES = {
    # relaxed contrapposto, weight on the right leg
    "stand": dict(
        root=(0.0, 0.0, -0.015),
        rot={"Hips": rot(Z, 4), "SpineMiddle": rot(Z, -6), "Chest": rot(Y, -2)},
        ik={"LeftLeg": ((0.12, 0.19, 0.106), FWD), "RightLeg": ((-0.04, -0.15, 0.106), FWD),
            "LeftArm": ((0.10, 0.36, 0.97), BACK), "RightArm": ((0.04, -0.37, 0.98), BACK)},
    ),
    # the Apollo lunar lope: airborne mid-stride, leaning forward
    "lope": dict(
        root=(0.0, 0.0, -0.02), ground=0.12, foot={"RightLeg": rot(Y, 40), "LeftLeg": rot(Y, -12)},
        rot={"Hips": rot(Y, 6), "SpineMiddle": rot(Y, 6) @ rot(Z, 6), "Chest": rot(Y, 3)},
        ik={"LeftLeg": ((0.40, 0.13, 0.16), FWD), "RightLeg": ((-0.40, -0.11, 0.30), FWD),
            "LeftArm": ((-0.10, 0.38, 1.02), BACK), "RightArm": ((0.30, -0.28, 1.08), BACK)},
    ),
    # T-pose: arms straight out to the sides at shoulder height (90 deg side raise, inside the A7L's ~100 deg),
    # open flat hands palms down, legs straight with the feet a little apart
    "tpose": dict(
        root=(0.0, 0.0, 0.0),
        ik={"LeftLeg": ((0.01, 0.12, 0.106), FWD), "RightLeg": ((0.01, -0.12, 0.106), FWD)},
        hand={"LeftArm": dict(contact="LeftHandWrist", target=(-0.01, 0.80, 1.456), fingers=(0.0, 1.0, 0.0),
                              dorsal=(0.0, 0.0, 1.0), open=1.0),
              "RightArm": dict(contact="RightHandWrist", target=(-0.01, -0.80, 1.456), fingers=(0.0, -1.0, 0.0),
                               dorsal=(0.0, 0.0, 1.0), open=1.0)},
    ),
    # Apollo-style salute: flat right hand, index fingertip at the right brow of the visor, elbow out
    "salute": dict(
        root=(0.0, 0.0, 0.0),
        rot={"Chest": rot(Y, -2)},
        ik={"LeftLeg": ((0.03, 0.13, 0.106), FWD), "RightLeg": ((0.03, -0.13, 0.106), FWD),
            "LeftArm": ((0.04, 0.37, 0.96), BACK), "RightArm": (None, (1.0, -0.55, -0.55))},
        hand={"RightArm": dict(contact="RightHandIndexTip", target=(0.14, -0.165, 1.70),
                               fingers=(-0.15, 1.0, 0.30), dorsal=(-0.35, 0.0, 1.0), open=1.0)},
    ),
    # geology: bending to reach the ground, hand on knee
    "geology": dict(
        root=(-0.04, 0.0, -0.12),
        rot={"Hips": rot(Y, 16), "SpineMiddle": rot(Y, 10) @ rot(Z, -10), "Chest": rot(Y, 5)},
        ik={"LeftLeg": ((0.24, 0.21, 0.106), FWD), "RightLeg": ((-0.20, -0.17, 0.106), FWD),
            "LeftArm": ((0.30, 0.20, 0.62), OUT_L), "RightArm": ((0.48, -0.24, 0.40), BACK)},
    ),
    # walking toward the camera, mid-stance
    "walk": dict(
        root=(0.0, 0.0, -0.03),
        rot={"Hips": rot(Z, -5), "SpineMiddle": rot(Z, 7) @ rot(Y, 4)},
        ik={"LeftLeg": ((0.24, 0.14, 0.106), FWD), "RightLeg": ((-0.24, -0.13, 0.17), FWD),
            "LeftArm": ((-0.06, 0.38, 0.99), BACK), "RightArm": ((0.20, -0.33, 1.01), BACK)},
    ),
}


def _rot_between(a, b):
    from scipy.spatial.transform import Rotation as R
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ax = np.cross(a, b)
    s, c = np.linalg.norm(ax), float(np.dot(a, b))
    return np.eye(3) if s < 1e-9 else R.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


def _slerp_rot(Rm, t):
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(R.from_matrix(Rm).as_rotvec() * t).as_matrix()


def open_hand(rig, side, amount=1.0):
    """World-aligned local rotations that straighten the fingers (and thumb) of one hand in line with
    their knuckles. amount 0..1 (1 = flat hand, fingers together as modelled at the knuckles)."""
    B = rig.bind[:, :3, 3]
    local = {}
    for f in ("Index", "Middle", "Ring", "Little", "Thumb"):
        ks = ("Metacarpal", "Proximal", "Distal", "Tip") if f == "Thumb" else \
            ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")
        ch = [rig.index[f"{side}Hand{f}{k}"] for k in ks]
        target = B[ch[1]] - B[ch[0]]
        A_prev = np.eye(3)
        for i in range(1, len(ch) - 1):
            d = A_prev @ (B[ch[i + 1]] - B[ch[i]])
            A_i = _slerp_rot(_rot_between(d, target), amount) @ A_prev
            local[f"{side}Hand{f}{ks[i]}"] = A_prev.T @ A_i
            A_prev = A_i
    return local


def _hand_frame(fingers, dorsal):
    f = np.asarray(fingers, float) / np.linalg.norm(fingers)
    d = np.asarray(dorsal, float) - np.dot(dorsal, f) * f
    d /= np.linalg.norm(d)
    return np.column_stack([f, d, np.cross(f, d)])


# ---------------------------------------------------------------- collision-aware elbow swivel
# Body proxies in the bind pose (rig frame, metres), rigid with the chest. Sized from the suit mesh: the torso is
# an oval ~35 cm deep x 42 cm wide (two vertical capsules), plus the helmet and the PLSS backpack.
TORSO_CAPSULES = [((0.03, y, 0.98), (0.03, y, 1.36), 0.15) for y in (-0.06, 0.06)]
HELMET_SPHERE = ((0.03, 0.0, 1.655), 0.20)
PLSS_BOX = ((-0.265, 0.0, 1.26), (0.135, 0.24, 0.35))  # centre, half extents
ARM_RADIUS = 0.07  # sleeve clearance radius around the arm bones
NATURAL_ELBOW = {"LeftArm": (-0.45, 0.55, -0.70), "RightArm": (-0.45, -0.55, -0.70)}  # back, out and down


def _seg_seg_dist(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1) if e > 1e-12 else 0.0  # second segment may be a point (sphere)
    s = np.clip((b * t - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def _seg_box_dist(a, b, centre, R, half, n=9):
    pts = a + np.outer(np.linspace(0, 1, n), b - a)
    q = np.abs((pts - centre) @ R) - half
    return float((np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)).min())


def body_clearance(rig, W, segments):
    """Minimum clearance [m] between arm segments [(a, b), ...] (radius ARM_RADIUS) and the body proxies, with
    the proxies carried by the posed chest (W: joint world matrices)."""
    c = rig.index["Chest"]
    M = W[c] @ np.linalg.inv(rig.bind[c])
    X = lambda p: M[:3, :3] @ np.asarray(p, float) + M[:3, 3]  # noqa: E731
    best = np.inf
    for a, b in segments:
        for p0, p1, r in TORSO_CAPSULES:
            best = min(best, _seg_seg_dist(a, b, X(p0), X(p1)) - r - ARM_RADIUS)
        hc, hr = HELMET_SPHERE
        best = min(best, _seg_seg_dist(a, b, X(hc), X(hc)) - hr - ARM_RADIUS)
        bc, bh = PLSS_BOX
        best = min(best, _seg_box_dist(a, b, X(bc), M[:3, :3], np.array(bh)) - ARM_RADIUS)
    return best


def _rom():
    from hand_physics import load_params
    return load_params()["body_rom"]


def cone_violation_deg(u, side_sign, lim):
    """Degrees by which limb direction u (in the parent's world-aligned frame: X forward, Y left, Z up) exceeds a
    direction-dependent cone around hanging-down: forward (flexion), backward (extension), outward (abduction) and
    across the body (adduction) limits, blended elliptically. side_sign = +1 left limb, -1 right limb."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    theta = np.degrees(np.arccos(np.clip(-u[2], -1, 1)))  # elevation from straight down
    fwd, out = u[0], side_sign * u[1]
    if np.hypot(fwd, out) < 1e-9:
        return theta - lim["flexion_deg"] if theta > 90 else 0.0
    psi = np.arctan2(out, fwd)
    a = lim["flexion_deg"] if fwd >= 0 else lim["extension_deg"]
    b = lim["abduction_deg"] if out >= 0 else lim["adduction_deg"]
    tmax = 1.0 / np.sqrt((np.cos(psi) / a) ** 2 + (np.sin(psi) / b) ** 2)
    return float(theta - tmax)


def shoulder_violation_deg(rig, W, side, elbow):
    """Suit shoulder range-of-motion violation for an elbow position (upper arm relative to the posed chest)."""
    c = rig.index["Chest"]
    A = W[c, :3, :3] @ rig.B[c].T  # chest world-aligned rotation
    u = A.T @ (np.asarray(elbow) - W[rig.index[f"{side}ArmUpper"], :3, 3])
    return cone_violation_deg(u, 1.0 if side == "Left" else -1.0, _rom()["shoulder"])


def elbow_position(S, T, lu, ll, pole, rig=None, W=None, natural=None, margin=0.01, samples=36, side=None):
    """Two-bone elbow. Without rig/W this is the classic pole-vector solution. With them, the elbow swivel about
    the shoulder->wrist axis is sampled and the candidate closest to the natural direction is chosen among those
    that clear the body by `margin`; if none does, the one with the most clearance is used."""
    d_min = abs(lu - ll) + 1e-4
    if side is not None:  # suit elbow: the hand cannot come closer to the shoulder than the maximum bend allows
        flex = np.radians(_rom()["elbow"]["flexion_deg"])
        d_min = max(d_min, np.sqrt(lu ** 2 + ll ** 2 - 2 * lu * ll * np.cos(np.pi - flex)))
    d = np.clip(np.linalg.norm(T - S), d_min, lu + ll - 1e-4)
    dirn = (T - S) / np.linalg.norm(T - S)
    ca = (lu ** 2 + d ** 2 - ll ** 2) / (2 * lu * d)
    sa = np.sqrt(max(0.0, 1 - ca ** 2))
    p = np.asarray(pole if natural is None else natural, float)
    p = p - p.dot(dirn) * dirn
    p /= np.linalg.norm(p)
    if rig is None:
        return S + lu * (ca * dirn + sa * p), d, dirn
    q = np.cross(dirn, p)
    Tn = S + dirn * d
    best_free, best_any = None, None
    for k in range(samples):
        phi = 2 * np.pi * k / samples
        phi = phi if phi <= np.pi else phi - 2 * np.pi
        E = S + lu * (ca * dirn + sa * (np.cos(phi) * p + np.sin(phi) * q))
        cl = body_clearance(rig, W, [(S, E), (E, Tn)])
        rom = shoulder_violation_deg(rig, W, side, E) if side else 0.0
        if cl >= margin and rom <= 0.0 and (best_free is None or abs(phi) < best_free[0]):
            best_free = (abs(phi), E)
        score = min(cl - margin, 0.0) * 100.0 - max(rom, 0.0)  # cm of penetration ~ degrees of ROM violation
        if best_any is None or score > best_any[0]:
            best_any = (score, E)
    E = best_free[1] if best_free is not None else best_any[1]
    return E, d, dirn


def solve(rig, pose):
    """Pose keys: root (xyz), rot {joint: R}, ik {limb: (target, pole)}, foot {leg: R},
    hand {arm: dict(contact=joint, target=xyz, fingers=dir, dorsal=dir, open=0..1)} — orients the hand
    (finger direction and back-of-hand direction) and solves the arm so `contact` lands on `target`."""
    T = np.eye(4)
    T[:3, 3] = pose.get("root", (0, 0, 0))
    local = dict(pose.get("rot", {}))
    B = rig.bind[:, :3, 3]
    hands = pose.get("hand", {})
    hand_rot = {}
    ik = dict(pose.get("ik", {}))
    for limb, h in hands.items():
        side = "Left" if limb.startswith("Left") else "Right"
        if h.get("open"):
            local.update(open_hand(rig, side, h["open"]))
        wrist = rig.index[f"{side}HandWrist"]
        f_b = B[rig.index[f"{side}HandMiddleProximal"]] - B[wrist]
        d_b = rig.B[wrist][:, 1]  # bind +Y = back of the hand (OpenXR)
        R_hand = _hand_frame(h["fingers"], h["dorsal"]) @ _hand_frame(f_b, d_b).T
        W0 = rig.fk({k: v for k, v in local.items() if k.startswith(f"{side}Hand")})
        offset = W0[rig.index[h["contact"]], :3, 3] - W0[wrist, :3, 3]
        pole = ik.get(limb, (None, NATURAL_ELBOW.get(limb, (0.0, 0.0, -1.0))))[1]
        ik[limb] = (np.asarray(h["target"], float) - R_hand @ offset, pole)
        hand_rot[limb] = R_hand
    for limb, (target, pole) in ik.items():
        u, l, e = (rig.index[n] for n in LIMBS[limb])
        W = rig.fk(local, T)
        S, Pw = W[u, :3, 3], rig.aligned(W)[rig.parent[u]]
        lu, ll = np.linalg.norm(B[l] - B[u]), np.linalg.norm(B[e] - B[l])
        t = np.asarray(target, float)
        if limb.endswith("Arm") and pose.get("avoid_body", True):
            # collision-aware swivel: an authored pole is the preferred direction, else the natural elbow
            natural = pole if pose.get("ik", {}).get(limb) else NATURAL_ELBOW[limb]
            mid, d, dirn = elbow_position(S, t, lu, ll, pole, rig=rig, W=W, natural=natural, side=limb[:-3])
        else:
            mid, d, dirn = elbow_position(S, t, lu, ll, pole)
        Ru = _rot_between(B[l] - B[u], mid - S)
        Rl = _rot_between(B[e] - B[l], S + dirn * d - mid)
        local[LIMBS[limb][0]] = Pw.T @ Ru
        local[LIMBS[limb][1]] = Ru.T @ Rl
        if limb.endswith("Leg"):  # keep the foot flat (world-aligned, as in the bind pose)
            local[LIMBS[limb][2]] = Rl.T @ pose.get("foot", {}).get(limb, np.eye(3))
        if limb in hand_rot:  # hand orientation relative to the solved forearm
            local[LIMBS[limb][2]] = Rl.T @ hand_rot[limb]
    return rig.fk(local, T)


def bake(rig, name, pose, src):
    W = solve(rig, pose)
    pts = rig.skin(W)
    names = [m["name"] for m in rig.meshes]
    sole = pts[names.index("boot_soles")]
    lift = pose.get("ground", 0.0) - sole[:, 2].min()
    out = os.path.join(ROOT, "build", f"hero_{name}.usd")
    layer = open_render_stage(src).Flatten()  # rig or packaged asset (physics variants off)
    stage = Usd.Stage.Open(layer)
    from fetch_textures import anchor  # the bake lives elsewhere: make texture paths absolute
    anchor(stage)
    for m, P in zip(rig.meshes, pts):
        P = P + np.array([0.0, 0.0, lift])
        prim = stage.GetPrimAtPath(rig.root.AppendPath(f"Geometry/{m['name']}"))
        mesh = UsdGeom.Mesh(prim)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([P.min(0), P.max(0)]).astype(np.float32)))
        mesh.GetNormalsAttr().Clear()
    stage.GetRootLayer().Export(out)
    print("wrote", out)


if __name__ == "__main__":
    src = os.path.join(ROOT, "build", "astronaut_rig.usd")
    if not os.path.exists(src):  # a fresh clone: bake from the committed asset
        src = os.path.join(ROOT, "assets", "astronaut", "astronaut.usda")
    rig = Rig(src)
    for name in sys.argv[1:] or POSES:
        bake(rig, name, POSES[name], src)
