"""Record a motion-captured walk on the astronaut for the Cycles walk GIF.

A CMU mocap walk (subject 35, trial 1: two full gait cycles, heel strike to heel strike, so the GIF loops) is
retargeted onto the OpenXR skeleton: pelvis and chest orientation from frames through the hips / shoulders, the
spine blended between them, and limb bones pointed along the captured bone directions. The root is scaled by the
leg-length ratio. The GPU full-body IK then cleans every frame (feet and hands held, arms out of the suit body,
shoulders inside the A7L range), and the GPU skinning writes the vertices exactly as the Newton viewer draws them.

    python examples/newton/utils/record_walk.py [--out build/walk] [--subject 35 --trial 1 --start 26 --end 283]

Writes <out>/walk.npz (frame times, hips path, mesh layout) and <out>/points_<frame>.npy. Render with
scripts/render_hero.py --walk. Motion data: CMU Graphics Lab Motion Capture Database (http://mocap.cs.cmu.edu),
downloaded to build/mocap/ and not redistributed.
"""
import argparse
import os
import sys

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kernels import REPO  # noqa: E402
from kernels.fullbody_ik import FullBodyIK  # noqa: E402
from kernels.skinning import AstronautRenderer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmu_mocap  # noqa: E402

SIDES = (("Left", "l"), ("Right", "r"))


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ax = np.cross(a, b)
    s, c = np.linalg.norm(ax), float(np.dot(a, b))
    return np.eye(3) if s < 1e-9 else R.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


def frame(lateral, up):
    """Orthonormal frame (forward, left, up) from a left-pointing lateral vector and an approximate up vector."""
    y = lateral / np.linalg.norm(lateral)
    z = up - np.dot(up, y) * y
    z /= np.linalg.norm(z)
    return np.column_stack([np.cross(y, z), y, z])


def slerp(Rm, t):
    return R.from_rotvec(R.from_matrix(Rm).as_rotvec() * t).as_matrix()


class MocapRetarget:
    """CMU joint positions -> rig joint world matrices (world-aligned rotation deltas, see Rig.fk)."""

    # rig bone (parent joint, child joint) -> source (from, to) joint
    LIMBS = [("{s}UpperLeg", "{s}LowerLeg", "{s}hip", "{s}knee"),
             ("{s}LowerLeg", "{s}FootAnkle", "{s}knee", "{s}ankle"),
             ("{s}FootAnkle", "{s}FootBall", "{s}ankle", "{s}ball"),
             ("{s}ArmUpper", "{s}ArmLower", "{s}shoulder", "{s}elbow"),
             ("{s}ArmLower", "{s}HandWrist", "{s}elbow", "{s}wrist"),
             ("{s}HandWrist", "{s}HandMiddleProximal", "{s}wrist", "{s}hand")]

    def __init__(self, rig, P):
        self.rig = rig
        self.B = rig.bind[:, :3, 3]
        J = lambda n: self.B[rig.index[n]]  # noqa: E731
        src = {"hips": P["root"], "spine": P["lowerback"], "chest": P["thorax"], "neck": P["lowerneck"],
               "neck_top": P["upperneck"]}
        for side, s in SIDES:  # ASF bone end points: hip, knee, ankle, ball; shoulder, elbow, wrist, hand
            src.update({f"{side}hip": P[f"{s}hipjoint"], f"{side}knee": P[f"{s}femur"], f"{side}ankle": P[f"{s}tibia"],
                        f"{side}ball": P[f"{s}foot"], f"{side}shoulder": P[f"{s}clavicle"],
                        f"{side}elbow": P[f"{s}humerus"], f"{side}wrist": P[f"{s}radius"], f"{side}hand": P[f"{s}hand"]})
        d = src["hips"][-1] - src["hips"][0]  # walk along +X
        yaw = -np.arctan2(d[1], d[0])
        Rz = R.from_euler("z", yaw).as_matrix()
        self.src = {k: v @ Rz.T for k, v in src.items()}
        S = self.src
        src_leg = (np.linalg.norm(S["Lefthip"] - S["Leftknee"], axis=1).mean()
                   + np.linalg.norm(S["Leftknee"] - S["Leftankle"], axis=1).mean())
        rig_leg = np.linalg.norm(J("LeftUpperLeg") - J("LeftLowerLeg")) + np.linalg.norm(J("LeftLowerLeg") - J("LeftFootAnkle"))
        self.scale = rig_leg / src_leg
        self.F_hips = frame(J("LeftUpperLeg") - J("RightUpperLeg"), J("SpineLower") - J("Hips"))
        self.F_chest = frame(J("LeftArmUpper") - J("RightArmUpper"), J("Neck") - J("Chest"))
        self.origin = S["hips"][0].copy()

    def _parent_R(self, Rw, n):
        rig = self.rig
        p = rig.parent[rig.index[n]]
        while rig.names[p] not in Rw:
            p = rig.parent[p]
        return Rw[rig.names[p]]

    def pose(self, f):
        rig, B, S = self.rig, self.B, {k: v[f] for k, v in self.src.items()}
        idx = rig.index
        Rw = {"Root": np.eye(3)}
        Rw["Hips"] = frame(S["Lefthip"] - S["Righthip"], S["spine"] - S["hips"]) @ self.F_hips.T
        R_chest = frame(S["Leftshoulder"] - S["Rightshoulder"], S["neck"] - S["chest"]) @ self.F_chest.T
        Q = R_chest @ Rw["Hips"].T
        for k, n in enumerate(("SpineLower", "SpineMiddle", "SpineUpper", "Chest")):
            Rw[n] = slerp(Q, (k + 1) / 4) @ Rw["Hips"]

        def point(n, child, a, b):
            Pr = self._parent_R(Rw, n)  # minimal rotation from the parent-carried bind direction: no added twist
            Rw[n] = rot_between(Pr @ (B[idx[child]] - B[idx[n]]), S[b] - S[a]) @ Pr

        point("Neck", "Head", "neck", "neck_top")
        Rw["Head"] = Rw["Neck"]
        for side, _ in SIDES:
            for bone, child, a, b in self.LIMBS:
                point(bone.format(s=side), child.format(s=side), a.format(s=side), b.format(s=side))
        local = {n: self._parent_R(Rw, n).T @ Rn for n, Rn in Rw.items() if n != "Root"}
        T = np.eye(4)
        T[:3, 3] = (S["hips"] - self.origin) * self.scale
        return rig.fk(local, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "build", "walk"))
    ap.add_argument("--subject", type=int, default=35)
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--start", type=int, default=26, help="first frame (120 Hz): a left heel strike")
    ap.add_argument("--end", type=int, default=283, help="the left heel strike two cycles later (exclusive)")
    ap.add_argument("--every", type=int, default=6, help="record every N-th mocap frame (6 = 20 fps)")
    ap.add_argument("--ik-iters", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    asf, amc = cmu_mocap.fetch(a.subject, a.trial, os.path.join(REPO, "build", "mocap"))
    P = cmu_mocap.joint_positions(cmu_mocap.load_asf(asf), cmu_mocap.load_amc(amc))
    device = wp.get_device()
    astro = AstronautRenderer(device)
    rig = astro.rig
    rt = MocapRetarget(rig, P)
    fbik = FullBodyIK(rig, device)
    frames, hips = [], []
    for f in range(a.start, a.end, a.every):
        W = rt.pose(f)
        fbik.set_reference(W, seed=True)  # start at the retarget; the IK only corrects it
        fbik.solve(a.ik_iters)
        W = fbik.rig_world(fbik.body_q().numpy())
        astro.skin(W, ground=True)  # soles on z = 0
        wp.synchronize()
        np.save(os.path.join(a.out, f"points_{f:04d}.npy"), astro.points.numpy())
        frames.append(f)
        hips.append(W[rig.index["Hips"], :3, 3])
    hips = np.array(hips)
    np.savez(os.path.join(a.out, "walk.npz"), frames=np.array(frames), times=(np.array(frames) - a.start) / cmu_mocap.FPS,
             hips=hips, scale=rt.scale, stride=(rt.src["hips"][a.end] - rt.src["hips"][a.start]) * rt.scale,
             mesh_names=np.array([m["name"] for m in rig.meshes]), mesh_counts=np.array([len(m["P"]) for m in rig.meshes]))
    print(f"recorded {len(frames)} frames to {a.out}; scale {rt.scale:.3f}, travel {(hips[-1] - hips[0]).round(2)} m")


if __name__ == "__main__":
    main()
