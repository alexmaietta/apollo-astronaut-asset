"""OpenXR / Manus hand input -> astronaut physics-hand joint targets.

Input: the 26 XR_EXT_hand_tracking joints per hand as world poses, named like Isaac Lab's Manus/OpenXR devices
("wrist", "palm", "thumb_metacarpal" ... "little_tip"; "pinky_*" is accepted too), or Manus-style per-joint
angles. Output: the wrist pose (for the wrist mount) and one target per physics DoF (hand_physics.py naming).

Angles are measured from bone directions only (-Z of each OpenXR joint frame) about the same axes the physics
joints use, so they don't depend on the operator's hand size. Physics joint coordinates are zero at the
astronaut's bind pose, so targets are the tracked angle minus the bind angle measured the same way, then clipped
to the joint limits (which include the glove's reduced range of motion).
"""
import numpy as np

from hand_physics import FINGERS, signed_angle

OPENXR_JOINTS = ["wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"] + [
    f"{f}_{k}" for f in ("index", "middle", "ring", "little") for k in ("metacarpal", "proximal", "intermediate", "distal", "tip")
] + ["palm"]


def rig_name(side, xr_name):
    """'index_proximal' -> 'LeftHandIndexProximal'."""
    xr_name = xr_name.replace("pinky", "little")
    return f"{side}Hand" + "".join(p.capitalize() for p in xr_name.split("_"))


def _quat_to_R(q_xyzw):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(q_xyzw).as_matrix()


def dof_angles(side, frames):
    """frames: {rig joint name: 3x3 world rotation (OpenXR axes)} -> {physics joint: angle rad}."""
    h = lambda k: f"{side}Hand{k}"  # noqa: E731
    bone = {n: -R[:, 2] for n, R in frames.items()}  # -Z points down the bone toward the tip
    out = {}
    for f in FINGERS:
        m, p, i, d = (h(f + k) for k in ("Metacarpal", "Proximal", "Intermediate", "Distal"))
        out[f"{p}_abd"] = signed_angle(bone[m], bone[p], frames[p][:, 1])
        out[f"{p}_flex"] = signed_angle(bone[m], bone[p], -frames[p][:, 0])
        out[f"{i}_flex"] = signed_angle(bone[p], bone[i], -frames[i][:, 0])
        out[f"{d}_flex"] = signed_angle(bone[i], bone[d], -frames[d][:, 0])
    w, tm, tp, td = (h(k) for k in ("Wrist", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal"))
    out[f"{tm}_abd"] = signed_angle(bone[w], bone[tm], frames[tm][:, 1])
    out[f"{tm}_flex"] = signed_angle(bone[w], bone[tm], -frames[tm][:, 0])
    out[f"{tp}_flex"] = signed_angle(bone[tm], bone[tp], -frames[tp][:, 0])
    out[f"{td}_flex"] = signed_angle(bone[tp], bone[td], -frames[td][:, 0])
    return out


class HandRetargeter:
    def __init__(self, rig, hand_model):
        self.side = hand_model.side
        self.joints = {j["name"]: j for j in hand_model.joints}
        bind = {n: rig.bind[i, :3, :3] for n, i in rig.index.items() if n.startswith(f"{self.side}Hand")}
        self.bind_angles = dof_angles(self.side, bind)

    def targets_from_frames(self, frames):
        ang = dof_angles(self.side, frames)
        out = {}
        for n, j in self.joints.items():
            q = ang[n] - self.bind_angles[n]
            out[n] = float(np.clip(q, np.radians(j["lower_deg"]), np.radians(j["upper_deg"])))
        return out

    def from_openxr(self, joints):
        """joints: {openxr name: (position xyz, quaternion xyzw)} in the sim world frame.
        Returns (wrist 4x4, {physics joint: target rad})."""
        frames = {rig_name(self.side, n): _quat_to_R(q) for n, (p, q) in joints.items()}
        W = np.eye(4)
        p, q = joints["wrist"]
        W[:3, :3], W[:3, 3] = _quat_to_R(q), p
        return W, self.targets_from_frames(frames)

    def from_angles(self, angles_deg):
        """Manus-style angles (degrees, measured from an open flat hand): {physics joint name: angle}.
        Unlisted joints stay at the open-hand pose."""
        out = {}
        for n, j in self.joints.items():
            q = np.radians(angles_deg.get(n, 0.0)) - self.bind_angles[n]
            out[n] = float(np.clip(q, np.radians(j["lower_deg"]), np.radians(j["upper_deg"])))
        return out


def openxr_from_rig(rig, W, side):
    """Synthetic OpenXR hand joints from a posed rig (joint world matrices W): {openxr name: (pos, quat xyzw)}.
    Useful for tests and scripted demos; a live device provides the same structure."""
    from scipy.spatial.transform import Rotation as R
    out = {}
    for n in OPENXR_JOINTS:
        i = rig.index[rig_name(side, n)]
        out[n] = (W[i, :3, 3].copy(), R.from_matrix(W[i, :3, :3]).as_quat())
    return out
