"""Astronaut skeleton: OpenXR (XR_FB_body_tracking + XR_META_body_tracking_full_body) joint set.

Positions are bind-pose joint centres in the source asset frame (cm, Z-up, facing -Y,
astronaut's left = +X). Left/right legs are asymmetric because the source pose has the
left foot forward and out.
"""
import json
import os

# name: (parent, (x, y, z))
_BODY = {
    "Root": (None, (0.0, -4.0, 0.0)),
    "Hips": ("Root", (0.0, -5.0, 92.0)),
    "SpineLower": ("Hips", (0.0, -5.0, 100.0)),
    "SpineMiddle": ("SpineLower", (0.0, -5.0, 112.0)),
    "SpineUpper": ("SpineMiddle", (0.0, -5.0, 124.0)),
    "Chest": ("SpineUpper", (0.0, -5.0, 136.0)),
    "Neck": ("Chest", (0.0, -6.0, 152.0)),
    "Head": ("Neck", (0.0, -8.0, 162.0)),
}
_ARM_L = {
    "LeftShoulder": ("Chest", (3.0, -6.0, 148.0)),
    "LeftScapula": ("LeftShoulder", (13.0, -4.0, 147.0)),
    "LeftArmUpper": ("LeftScapula", (25.5, -3.0, 145.0)),
    "LeftArmLower": ("LeftArmUpper", (33.5, -3.0, 119.0)),
    "LeftHandWristTwist": ("LeftArmLower", (35.0, -4.0, 106.0)),
    "LeftHandWrist": ("LeftArmLower", (36.0, -5.0, 96.0)),
    "LeftHandPalm": ("LeftHandWrist", (36.5, -6.5, 90.0)),
    "LeftHandThumbMetacarpal": ("LeftHandWrist", (35.0, -8.0, 94.0)),
    "LeftHandThumbProximal": ("LeftHandThumbMetacarpal", (34.0, -11.0, 91.5)),
    "LeftHandThumbDistal": ("LeftHandThumbProximal", (33.6, -12.0, 89.8)),
    "LeftHandThumbTip": ("LeftHandThumbDistal", (33.4, -12.2, 88.4)),
}
# finger: metacarpal, proximal (MCP), intermediate (PIP), distal (DIP), tip
_FINGERS_L = {
    "Index": [(36.5, -8.0, 94.5), (37.5, -11.0, 88.0), (37.5, -12.0, 84.5), (37.7, -12.4, 82.5), (37.9, -12.5, 80.9)],
    "Middle": [(36.8, -6.0, 94.5), (37.5, -8.5, 87.5), (36.8, -9.0, 83.5), (36.2, -9.0, 81.5), (35.6, -9.0, 80.2)],
    "Ring": [(36.8, -4.0, 94.5), (37.0, -6.0, 88.0), (36.3, -5.8, 84.0), (35.7, -5.6, 82.2), (35.2, -5.4, 81.0)],
    "Little": [(36.5, -2.0, 94.0), (36.5, -3.5, 88.5), (35.6, -2.6, 85.0), (35.0, -2.3, 83.6), (34.4, -2.1, 82.8)],
}
_LEG = {
    "Left": {
        "UpperLeg": (9.0, -6.0, 88.0), "LowerLeg": (17.5, -8.0, 52.0), "FootAnkleTwist": (23.5, -6.5, 16.0),
        "FootAnkle": (24.0, -6.0, 10.0), "FootSubtalar": (24.0, -3.0, 6.0), "FootTransverse": (25.0, -12.0, 5.0),
        "FootBall": (26.0, -20.0, 3.0),
    },
    "Right": {
        "UpperLeg": (-9.0, -4.0, 88.0), "LowerLeg": (-11.5, -3.0, 52.0), "FootAnkleTwist": (-12.5, 1.0, 16.0),
        "FootAnkle": (-13.0, 2.0, 10.0), "FootSubtalar": (-13.0, 4.0, 6.0), "FootTransverse": (-13.5, -6.0, 5.0),
        "FootBall": (-14.0, -13.0, 3.0),
    },
}
_LEG_PARENT = {"UpperLeg": "Hips", "LowerLeg": "UpperLeg", "FootAnkleTwist": "LowerLeg", "FootAnkle": "LowerLeg",
               "FootSubtalar": "FootAnkle", "FootTransverse": "FootSubtalar", "FootBall": "FootTransverse"}

# Fitted finger joints (scripts/fit_hands.py -> data/hand_fit_left.json) replace the template above when present.
_FIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hand_fit_left.json")
if os.path.exists(_FIT) and os.environ.get("ASTRONAUT_HAND_TEMPLATE") != "1":
    with open(_FIT) as _f:
        _fit = json.load(_f)
    for _finger, _pts in _fit.items():
        if _finger == "Thumb":
            for _k, _p in _pts.items():
                _ARM_L[f"LeftHandThumb{_k}"] = (_ARM_L[f"LeftHandThumb{_k}"][0], tuple(_p))
        else:
            _FINGERS_L[_finger] = [tuple(_pts[k]) for k in ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")]

# Helper joints that do not form part of a bone chain (twist/palm/tips get small or no weights).
NON_DEFORMING = {"Root", "LeftHandPalm", "RightHandPalm"} | {
    f"{s}Hand{f}Tip" for s in ("Left", "Right") for f in ("Thumb", "Index", "Middle", "Ring", "Little")}


def _build():
    j = dict(_BODY)
    arm = dict(_ARM_L)
    for f, pts in _FINGERS_L.items():
        names = [f"LeftHand{f}{k}" for k in ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")]
        parent = "LeftHandWrist"
        for n, p in zip(names, pts):
            arm[n] = (parent, p)
            parent = n
    j.update(arm)
    for n, (par, (x, y, z)) in arm.items():  # mirror left arm to right
        j[n.replace("Left", "Right")] = (par.replace("Left", "Right"), (-x, y, z))
    for side, pts in _LEG.items():
        for k, p in pts.items():
            par = _LEG_PARENT[k]
            j[f"{side}{k}"] = (par if par == "Hips" else f"{side}{par}", p)
    return j


JOINTS = _build()


def ordered():
    """Joint names in parent-before-child order."""
    out, done = [], set()
    def visit(n):
        if n in done:
            return
        par = JOINTS[n][0]
        if par:
            visit(par)
        done.add(n)
        out.append(n)
    for n in JOINTS:
        visit(n)
    return out


def usd_path(name):
    """UsdSkel joint path (ancestors/name)."""
    parts = [name]
    while JOINTS[parts[-1]][0]:
        parts.append(JOINTS[parts[-1]][0])
    return "/".join(reversed(parts))


if __name__ == "__main__":
    print(len(JOINTS), "joints")
