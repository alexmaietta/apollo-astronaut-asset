"""OpenXR-style joint orientations for the bind pose (rig output frame: metres, X forward, Y left, Z up).

Hands follow XR_EXT_hand_tracking exactly: +Z runs along the bone and points away from the fingertip
(so -Z points down the bone toward the tip), +Y points out of the back of the hand / finger nail, and
X = Y x Z (right-handed). The wrist's +Z points from the middle-finger knuckle back toward the wrist; the
palm's +Z runs along the middle metacarpal. Because the bind pose has curled fingers, each finger
segment's frame rotates with the curl about that hand's knuckle axis (the spec's "fully opened hand"
frames are these frames with the fingers straightened).

The body uses the same rule, since XR_FB_body_tracking does not define per-joint axes: +Z along the bone
away from the child, X = Y x Z, and +Y = back of the hand for the arm chain (so a palms-down T-pose gives
+Y up, matching the hand spec), forward for the spine, neck, head and legs, and up for the clavicles and
the forward-pointing foot bones. Root is world-aligned.
"""
import numpy as np

SIDES = ("Left", "Right")
FINGERS = ("Index", "Middle", "Ring", "Little")

# bone direction: joint -> this joint (when not its first child)
AIM = {"Hips": "SpineLower", "Chest": "Neck", "FootAnkle": "FootBall"}


def _unit(v):
    return v / np.linalg.norm(v)


def _frame(z_back, y_hint):
    """Rotation matrix with columns X, Y, Z from a +Z direction and a +Y hint (orthogonalised)."""
    z = _unit(z_back)
    y = y_hint - (y_hint @ z) * z
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("Y hint parallel to bone")
    y = _unit(y)
    return np.column_stack([np.cross(y, z), y, z])


def compute(joints, pos):
    """joints: {name: (parent, ...)}; pos: {name: xyz (output frame)}. Returns {name: 3x3}."""
    kids = {n: [c for c, (p, *_) in joints.items() if p == n] for n in joints}
    fwd, up = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    R = {}

    def toward(n, target):
        return pos[target] - pos[n]

    for side in SIDES:
        h = lambda k: f"{side}Hand{k}"  # noqa: E731
        lateral = np.array([0.0, 1.0 if side == "Left" else -1.0, 0.0])
        wrist_dir = toward(h("Wrist"), h("MiddleProximal"))
        knuckles = pos[h("LittleProximal")] - pos[h("IndexProximal")]
        dorsal = np.cross(knuckles, wrist_dir)
        dorsal = _unit(dorsal if dorsal @ lateral > 0 else -dorsal)  # palms face the body in the bind pose
        a = _unit(knuckles)
        # arm chain: +Y = back of the hand
        R[f"{side}ArmUpper"] = _frame(-toward(f"{side}ArmUpper", f"{side}ArmLower"), dorsal)
        R[f"{side}ArmLower"] = _frame(-toward(f"{side}ArmLower", h("Wrist")), dorsal)
        R[h("WristTwist")] = _frame(-toward(h("WristTwist"), h("Wrist")), dorsal)
        R[h("Wrist")] = _frame(-wrist_dir, dorsal)
        R[h("Palm")] = _frame(-toward(h("MiddleMetacarpal"), h("MiddleProximal")), dorsal)
        for f in FINGERS:
            chain = [h(f + k) for k in ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")]
            for i, n in enumerate(chain):
                d = toward(n, chain[i + 1]) if i < 4 else toward(chain[3], chain[4])
                y = np.cross(_unit(-d), a)  # rotates with the curl about the knuckle axis
                R[n] = _frame(-d, y if y @ dorsal > 0 else -y)
        # thumb: nail faces away from the palm, between the back of the hand and the index side
        tchain = [h("Thumb" + k) for k in ("Metacarpal", "Proximal", "Distal", "Tip")]
        away = _unit(pos[tchain[1]] - pos[h("IndexProximal")])
        nail = _unit(dorsal + away)
        tax = None
        for i, n in enumerate(tchain):
            d = toward(n, tchain[i + 1]) if i < 3 else toward(tchain[2], tchain[3])
            if tax is None:
                tax = _unit(np.cross(nail, -d))  # thumb flexion axis, fixed along the chain
            y = np.cross(_unit(-d), tax)
            R[n] = _frame(-d, y if y @ nail > 0 else -y)
        # clavicle / scapula point laterally: +Y up
        R[f"{side}Shoulder"] = _frame(-toward(f"{side}Shoulder", f"{side}Scapula"), up)
        R[f"{side}Scapula"] = _frame(-toward(f"{side}Scapula", f"{side}ArmUpper"), up)
        # legs: +Y forward; forward-pointing foot bones: +Y up
        R[f"{side}UpperLeg"] = _frame(-toward(f"{side}UpperLeg", f"{side}LowerLeg"), fwd)
        R[f"{side}LowerLeg"] = _frame(-toward(f"{side}LowerLeg", f"{side}FootAnkle"), fwd)
        R[f"{side}FootAnkleTwist"] = _frame(-toward(f"{side}LowerLeg", f"{side}FootAnkle"), fwd)
        R[f"{side}FootAnkle"] = _frame(-toward(f"{side}FootAnkle", f"{side}FootBall"), up)
        R[f"{side}FootSubtalar"] = _frame(-toward(f"{side}FootSubtalar", f"{side}FootTransverse"), up)
        R[f"{side}FootTransverse"] = _frame(-toward(f"{side}FootTransverse", f"{side}FootBall"), up)
        R[f"{side}FootBall"] = R[f"{side}FootTransverse"]
    for n, nxt in (("Hips", "SpineLower"), ("SpineLower", "SpineMiddle"), ("SpineMiddle", "SpineUpper"),
                   ("SpineUpper", "Chest"), ("Chest", "Neck"), ("Neck", "Head")):
        R[n] = _frame(-toward(n, nxt), fwd)
    R["Head"] = R["Neck"]
    R["Root"] = np.eye(3)
    missing = set(joints) - set(R)
    assert not missing, missing
    return R
