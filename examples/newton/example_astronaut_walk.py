# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Astronaut Walk
#
# Newton's Unitree G1 walking policy (stock ONNX policy, MuJoCo-Warp) drives
# the skinned Apollo 11 astronaut. Each frame the G1 link rotations are
# retargeted onto the astronaut skeleton as world-space deltas from the G1
# zero pose, applied on top of a straightened neutral stance; the 389k-vertex
# suit is skinned on the GPU with the shoulder correctives and drawn textured
# beside the robot. A GPU full-body IK (Newton IK) cleans each retargeted
# frame: hands and feet stay put while the arms are kept out of the suit's
# torso, the legs apart, and the shoulders inside the A7L range of motion.
# This is a visual retarget (Earth gravity, G1 dynamics), not astronaut physics.
#
# Keys: i/k forward/back, j/l strafe, u/o turn, p reset. Walks forward on its
# own while no key is held. The camera starts on a front view of
# the pair and follows them (untick "Follow camera" to fly freely).
#
# Command (from the astronaut repo root, with the Newton venv):
#   python examples/newton/example_astronaut_walk.py
#   ... --viewer null --num-frames 400 --record build/walk_anim.usda
#
###########################################################################

import argparse
import os
import sys

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

import newton
import newton.examples
from newton.examples.robot.example_robot_policy import Example as PolicyExample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import REPO  # noqa: E402  (also puts scripts/ on sys.path)
from kernels.fullbody_ik import FullBodyIK  # noqa: E402
from kernels.skinning import RIG, AstronautRenderer  # noqa: E402
from poses import body_clearance  # noqa: E402

# astronaut joint -> G1 link driving its rotation (descendants follow their nearest mapped ancestor)
LINK_MAP = {
    "Hips": "pelvis",
    "SpineMiddle": "torso_link",
    "LeftArmUpper": "left_shoulder_yaw_link",
    "LeftArmLower": "left_elbow_link",
    "LeftHandWrist": "left_wrist_yaw_link",
    "RightArmUpper": "right_shoulder_yaw_link",
    "RightArmLower": "right_elbow_link",
    "RightHandWrist": "right_wrist_yaw_link",
    "LeftUpperLeg": "left_hip_yaw_link",
    "LeftLowerLeg": "left_knee_link",
    "LeftFootAnkle": "left_ankle_roll_link",
    "RightUpperLeg": "right_hip_yaw_link",
    "RightLowerLeg": "right_knee_link",
    "RightFootAnkle": "right_ankle_roll_link",
}
# Neutral pose: limb bones straightened toward these world directions (left = +Y), so the
# astronaut's A-pose arms and splayed left leg start from a symmetric stance.
NEUTRAL_DIRS = {
    ("LeftUpperLeg", "LeftLowerLeg"): (0.0, 0.03, -1.0),
    ("LeftLowerLeg", "LeftFootAnkle"): (0.0, 0.0, -1.0),
    ("RightUpperLeg", "RightLowerLeg"): (0.0, -0.03, -1.0),
    ("RightLowerLeg", "RightFootAnkle"): (0.0, 0.0, -1.0),
    ("LeftArmUpper", "LeftArmLower"): (0.0, 0.12, -1.0),
    ("LeftArmLower", "LeftHandWrist"): (0.0, 0.05, -1.0),
    ("RightArmUpper", "RightArmLower"): (0.0, -0.12, -1.0),
    ("RightArmLower", "RightHandWrist"): (0.0, -0.05, -1.0),
}
SIDE_OFFSET = np.array([1.2, 0.0, 0.0])  # beside the robot (the base example spawns the G1 facing +Y)
CAMERA_POS = np.array([1.3, 3.8, 1.3])  # in front of the pair, slightly to one side: both walk toward +Y (the camera)
CAMERA_LOOK_AT = np.array([0.6, 0.0, 0.85])  # midway between the G1 and the astronaut, about chest height


def _rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ax = np.cross(a, b)
    s, c = np.linalg.norm(ax), np.dot(a, b)
    return np.eye(3) if s < 1e-9 else R.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


class G1Retarget:
    def __init__(self, rig, link_index, rest_body_rot, g1_leg_len):
        self.rig, self.link_index, self.rest_rot = rig, link_index, rest_body_rot
        pos = rig.bind[:, :3, 3]
        self.N = np.repeat(np.eye(3)[None], len(rig.names), 0)  # neutral world rotations
        for (j, child), d in NEUTRAL_DIRS.items():
            self.N[rig.index[j]] = _rot_between(pos[rig.index[child]] - pos[rig.index[j]], np.array(d))
        for side in ("Left", "Right"):  # hands keep their pose relative to the straightened forearm
            for i, nm in enumerate(rig.names):
                if nm.startswith(f"{side}Hand"):
                    self.N[i] = self.N[rig.index[f"{side}ArmLower"]]
        self.driver = []  # nearest mapped ancestor-or-self
        for i in range(len(rig.names)):
            k = i
            while k >= 0 and rig.names[k] not in LINK_MAP:
                k = rig.parent[k]
            self.driver.append(LINK_MAP[rig.names[k]] if k >= 0 else None)
        self.scale = np.linalg.norm(pos[rig.index["LeftUpperLeg"]] - pos[rig.index["LeftFootAnkle"]]) / g1_leg_len
        self.hips_i = rig.index["Hips"]
        self.origin = None

    def pose(self, body_q):
        """body_q: (nbody, 7) G1 body poses [p, q_xyzw] -> astronaut joint world matrices."""
        rig = self.rig
        D = {n: R.from_quat(body_q[i, 3:]).as_matrix() @ self.rest_rot[n].T for n, i in self.link_index.items()}
        Wrot = np.array([(D[d] if d else np.eye(3)) @ self.N[i] for i, d in enumerate(self.driver)])
        local = {nm: Wrot[i] if rig.parent[i] < 0 else Wrot[rig.parent[i]].T @ Wrot[i] for i, nm in enumerate(rig.names)}
        W = rig.fk(local)
        pelvis = body_q[self.link_index["pelvis"], :3]
        if self.origin is None:
            self.origin = pelvis.copy()
        target = (pelvis - self.origin) * self.scale + SIDE_OFFSET
        W[:, :3, 3] += (target - W[self.hips_i, :3, 3]) * np.array([1.0, 1.0, 0.0])
        return W


class Example(PolicyExample):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.auto_walk = not args.no_auto_walk
        self._command = wp.vec3(1.0 if self.auto_walk else 0.0, 0.0, 0.0)
        if hasattr(viewer, "is_key_down"):
            orig = viewer.is_key_down

            def is_key_down(key, _orig=orig):
                manual = any(_orig(k) for k in "ijkluo")
                return _orig(key) or (self.auto_walk and not manual and key == "i")

            viewer.is_key_down = is_key_down

        labels = [lab.rsplit("/", 1)[-1] for lab in self.model.body_label]
        link_index = {name: labels.index(name) for name in set(LINK_MAP.values())}
        q0 = np.zeros(self.model.joint_coord_count, np.float32)  # G1 zero pose, identity base
        q0[6] = 1.0
        rest = self.model.state()
        newton.eval_fk(self.model, wp.array(q0, device=self.device), wp.zeros(self.model.joint_dof_count, device=self.device), rest)
        bq = rest.body_q.numpy()
        rest_rot = {n: R.from_quat(bq[i, 3:]).as_matrix() for n, i in link_index.items()}
        g1_leg = np.linalg.norm(bq[labels.index("left_hip_pitch_link"), :3] - bq[labels.index("left_ankle_roll_link"), :3])

        self.follow = True
        self._g1_yaw0 = None
        self._look_at(CAMERA_POS, CAMERA_LOOK_AT)
        self.astro = AstronautRenderer(self.device)
        self.retarget = G1Retarget(self.astro.rig, link_index, rest_rot, g1_leg)
        # full-body IK cleans up the retarget: hands/feet stay where the retarget puts them, arms clear the torso,
        # legs clear each other, shoulders stay in the suit's range of motion (GPU, warm-started each frame)
        self.fbik = FullBodyIK(self.astro.rig, self.device) if args.fullbody_ik else None
        self.ik_iters = args.ik_iters
        self.min_clearance = np.inf
        self.measure = bool(args.test)
        self.record = args.record
        self.frames = []
        self.astro_root = None
        print(f"astronaut: {self.astro.vertex_count} verts, root scale {self.retarget.scale:.3f}")

    def _look_at(self, pos, target):
        """Point the viewer camera from pos at target (viewer yaw is measured from +X toward +Y, pitch up)."""
        if not hasattr(self.viewer, "set_camera"):
            return
        d = np.asarray(target, float) - np.asarray(pos, float)
        yaw = np.degrees(np.arctan2(d[1], d[0]))
        pitch = np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))
        self.viewer.set_camera(wp.vec3(*map(float, pos)), pitch=float(pitch), yaw=float(yaw))

    def _follow_camera(self, pelvis_q):
        """Keep the initial three-quarter front framing on the midpoint between the two walkers, turning with
        the G1's heading."""
        yaw = R.from_quat(pelvis_q[3:]).as_euler("xyz")[2]
        if self._g1_yaw0 is None:
            self._g1_yaw0 = yaw
        rot = R.from_euler("z", yaw - self._g1_yaw0).as_matrix()
        mid = (pelvis_q[:3] + self.astro_root) / 2.0
        target = np.array([mid[0], mid[1], CAMERA_LOOK_AT[2]])
        start_mid = (np.array([0.0, 0.0]) + SIDE_OFFSET[:2]) / 2.0
        offset = rot @ (CAMERA_POS - np.array([start_mid[0], start_mid[1], CAMERA_LOOK_AT[2]]))
        self._look_at(target + offset, target)

    def render(self):
        body_q = self.state_0.body_q.numpy()
        W_ref = self.retarget.pose(body_q)
        if self.fbik is not None:
            self.fbik.set_reference(W_ref, seed=True)  # start each frame at the retarget; IK only corrects it
            self.fbik.solve(self.ik_iters)
            W_ref = self.fbik.rig_world(self.fbik.body_q().numpy())
        if self.measure:  # test metric (CPU), off in normal runs
            rig = self.astro.rig
            for s in ("Left", "Right"):
                u, l, w = (rig.index[f"{s}{n}"] for n in ("ArmUpper", "ArmLower", "HandWrist"))
                self.min_clearance = min(self.min_clearance, body_clearance(
                    rig, W_ref, [(W_ref[u, :3, 3], W_ref[l, :3, 3]), (W_ref[l, :3, 3], W_ref[w, :3, 3])]))
        W = self.astro.skin(W_ref, ground=True)
        self.astro_root = W[self.retarget.hips_i, :3, 3]
        if self.follow:
            self._follow_camera(body_q[self.retarget.link_index["pelvis"]])
        if self.record:
            self.frames.append(W)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.astro.log(self.viewer)
        self.viewer.end_frame()

    def gui(self, ui):
        _, self.follow = ui.checkbox("Follow camera", self.follow)

    def test_final(self):
        print(f"astronaut: minimum arm-torso clearance over the walk {self.min_clearance * 100:.1f} cm "
              f"(full-body IK {'on' if self.fbik else 'off'})")
        super().test_final()
        assert self.astro_root is not None and np.linalg.norm(self.astro_root[:2] - SIDE_OFFSET[:2]) > 0.5, \
            "the astronaut did not walk"

    def save_animation(self, path, fps):
        """UsdSkel animation of the astronaut with the shoulder-corrective weights (scripts/skel_animation.py)."""
        from skel_animation import write_animation
        write_animation(self.astro.rig, self.frames, path, fps, asset=RIG)
        print("astronaut: wrote", path, len(self.frames), "frames (joints + corrective blend-shape weights)")

    @staticmethod
    def create_parser():
        parser = PolicyExample.create_parser()
        parser.add_argument("--no-auto-walk", action="store_true", help="Only walk while keys are held.")
        parser.add_argument("--record", type=str, default=None, help="Save the astronaut motion as a UsdSkel animation.")
        parser.add_argument("--fullbody-ik", action=argparse.BooleanOptionalAction, default=True,
                            help="Clean the retarget with the GPU full-body IK (self-collision, suit range of motion).")
        parser.add_argument("--ik-iters", type=int, default=4, help="Full-body IK iterations per frame (warm-started).")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if args.record:
        example.save_animation(args.record, fps=int(round(1.0 / example.frame_dt / example.decimation)) or 50)
