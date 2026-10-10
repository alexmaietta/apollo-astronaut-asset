# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Astronaut IK
#
# Interactive full-body IK for the textured, skinned Apollo 11 astronaut, in
# the style of Newton's gizmo IK examples (example_ik_h1): drag the transform
# gizmos on the hands, feet and hips and the GPU full-body IK
# (kernels/fullbody_ik.py, Newton IK / Levenberg-Marquardt, one CUDA graph per
# frame) re-solves every frame. Besides the targets it keeps the posture close
# to the chosen start pose, respects joint limits and the A7L shoulder range of
# motion, and keeps the arms and legs out of the suit body (capsule
# self-collision). On release each gizmo snaps back to the solved pose.
#
# GUI: start pose (bind or a gallery pose), collision / suit-ROM toggles, IK
# iterations, live target errors, shoulder margins and arm clearance.
#
# Command (from the astronaut repo root, with the Newton venv):
#   python examples/newton/example_astronaut_ik.py
#   ... --pose stand
#
###########################################################################

import os
import sys

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

import newton
import newton.examples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import REPO  # noqa: E402  (also puts scripts/ on sys.path)
from kernels.fullbody_ik import FullBodyIK, _set_row  # noqa: E402
from kernels.skinning import AstronautRenderer  # noqa: E402
from poses import POSES, body_clearance, shoulder_violation_deg, solve  # noqa: E402

EFFECTORS = ("LeftHandWrist", "RightHandWrist", "LeftFootAnkle", "RightFootAnkle")


def _tf(M):
    return wp.transform(wp.vec3(*map(float, M[:3, 3])), wp.quat(*map(float, R.from_matrix(M[:3, :3]).as_quat())))


def _mat(tf):
    M = np.eye(4)
    M[:3, 3] = np.array(wp.transform_get_translation(tf))
    M[:3, :3] = R.from_quat(np.array(wp.transform_get_rotation(tf))).as_matrix()
    return M


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane(color=(0.42, 0.41, 0.39))
        self.model = builder.finalize()
        self.state = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(3.0, -1.7, 1.35), pitch=-8.0, yaw=150.0)

        self.astro = AstronautRenderer(wp.get_device())
        self.rig = self.astro.rig
        self.ik = FullBodyIK(self.rig, wp.get_device())
        self.pose_names = ["bind", *POSES]
        self.iterations = 8
        self.collisions = True
        self.suit_rom = True
        self._weights = (self.ik.collision.weight, self._cone().weight)
        self.metrics = {}
        self.W = None
        self._start(args.pose if args.pose in self.pose_names else "stand")

    def _cone(self):
        from kernels.fullbody_ik import ShoulderConeObjective
        return next(o for o in self.ik.objectives if isinstance(o, ShoulderConeObjective))

    def _start(self, name):
        """Seed the IK at a pose: it becomes the posture prior, and the gizmos jump to its hands/feet/hips."""
        self.pose = name
        W = self.rig.bind.copy() if name == "bind" else solve(self.rig, POSES[name])
        self.ik.set_reference(W, seed=True)
        # persistent gizmo transforms (the viewer mutates them in place while dragging)
        self.tfs = {n: _tf(W[self.rig.index[n]]) for n in (*EFFECTORS, "Hips")}
        self._solved = {n: _tf(W[self.rig.index[n]]) for n in self.tfs}

    def _set_objective_weights(self):
        self.ik.collision.weight = self._weights[0] if self.collisions else 0.0
        self._cone().weight = self._weights[1] if self.suit_rom else 0.0
        self.ik.__dict__.get("_graphs", {}).clear()  # weights are captured in the solve graph: re-capture

    # ------------------------------------------------------------------ IK loop
    def _push_targets(self):
        for n in EFFECTORS:
            self.ik.set_target(n, _mat(self.tfs[n]))
        _set_row(self.ik.root_pos, 0, np.array(wp.transform_get_translation(self.tfs["Hips"])))

    def step(self):
        self._push_targets()
        self.ik.solve(self.iterations)  # warm-started from the previous frame
        W = self.ik.rig_world(self.ik.body_q().numpy())
        self.W = W
        for n in self.tfs:
            self._solved[n] = _tf(W[self.rig.index[n]])
        rig, m = self.rig, {}
        for n in EFFECTORS:
            m[f"{n} error (mm)"] = 1000 * np.linalg.norm(W[rig.index[n], :3, 3] - np.array(wp.transform_get_translation(self.tfs[n])))
        for s in ("Left", "Right"):
            u, l, w = (rig.index[f"{s}{k}"] for k in ("ArmUpper", "ArmLower", "HandWrist"))
            m[f"{s} shoulder margin (deg)"] = -shoulder_violation_deg(rig, W, s, W[l, :3, 3])
            m[f"{s} arm clearance (cm)"] = 100 * body_clearance(rig, W, [(W[u, :3, 3], W[l, :3, 3]), (W[l, :3, 3], W[w, :3, 3])])
        self.metrics = m
        self.astro.skin(W)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for n, tf in self.tfs.items():
            self.viewer.log_gizmo(f"target_{n}", tf, snap_to=self._solved[n])
        self.viewer.log_state(self.state)
        self.astro.log(self.viewer)
        self.viewer.end_frame()

    def gui(self, ui):
        changed, idx = ui.combo("Start pose", self.pose_names.index(self.pose), self.pose_names)
        if changed:
            self._start(self.pose_names[idx])
        if ui.button("Reset gizmos to pose"):
            self._start(self.pose)
        c1, self.collisions = ui.checkbox("Body self-collision", self.collisions)
        c2, self.suit_rom = ui.checkbox("A7L shoulder range of motion", self.suit_rom)
        if c1 or c2:
            self._set_objective_weights()
        _, self.iterations = ui.slider_int("IK iterations / frame", self.iterations, 1, 30)
        for k, v in self.metrics.items():
            bad = (k.endswith("(mm)") and v > 10.0) or (not k.endswith("(mm)") and v < 0.0)
            ui.text(f"{k}: {v:+.1f}" + ("  (!)" if bad else ""))

    def test_final(self):
        # drag the right hand a quarter of the way toward its shoulder and 5 cm up (reachable from any start pose,
        # including a fully extended T-pose arm), then check the solve
        start = _mat(self.tfs["RightHandWrist"])
        shoulder = self.W[self.rig.index["RightArmUpper"], :3, 3]
        delta = 0.25 * (shoulder - start[:3, 3]) + np.array([0.0, 0.0, 0.05])
        feet = {n: self.W[self.rig.index[n], :3, 3].copy() for n in ("LeftFootAnkle", "RightFootAnkle")}
        for k in range(1, 61):
            M = start.copy()
            M[:3, 3] += delta * k / 60
            self.tfs["RightHandWrist"] = _tf(M)
            self.step()
        m = self.metrics
        assert np.isfinite(self.astro.points.numpy()).all()
        assert m["RightHandWrist error (mm)"] < 15.0, m
        assert all(np.linalg.norm(self.W[self.rig.index[n], :3, 3] - p) < 0.01 for n, p in feet.items()), "feet slid"
        assert m["Right arm clearance (cm)"] > -1.5 and m["Right shoulder margin (deg)"] > -2.0, m
        print("ik: dragged right hand reached within %.1f mm; feet held; arm clearance %.1f cm; shoulder margin %.1f deg"
              % (m["RightHandWrist error (mm)"], m["Right arm clearance (cm)"], m["Right shoulder margin (deg)"]))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--pose", default="stand", help="Start pose: bind or a gallery pose.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
