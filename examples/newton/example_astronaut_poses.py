# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Astronaut Poses
#
# An interactive pose editor for the textured, skinned Apollo 11 astronaut in
# Newton's viewer. Pick a pose from the library (the bind pose, the gallery
# poses from the README -- stand, lope, salute, geology, walk -- and any you
# saved), then adjust it: root offset, hip/spine/chest bend-twist sliders, hand
# and foot target positions (two-bone IK with the collision-aware elbow and the
# A7L shoulder/elbow limits), and per-hand finger grip. Optionally refine with
# the GPU full-body IK. The panel shows the shoulder range-of-motion margin and
# arm-to-torso clearance. Save poses by name to data/user_poses.json and export
# a posed USD (build/hero_<name>.usd) for the Cycles hero renderer.
#
# Command (from the astronaut repo root, with the Newton venv):
#   python examples/newton/example_astronaut_poses.py
#   ... --pose salute
#
###########################################################################

import copy
import json
import os
import sys

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

import newton
import newton.examples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import REPO  # noqa: E402  (also puts scripts/ on sys.path)
from kernels.skinning import RIG, AstronautRenderer  # noqa: E402
from poses import LIMBS, POSES, bake, body_clearance, shoulder_violation_deg, solve  # noqa: E402

USER_POSES = os.path.join(REPO, "data", "user_poses.json")
ROT_JOINTS = ("Hips", "SpineMiddle", "Chest")
ROT_LABELS = ("bend (fwd +)", "side bend", "twist")
FINGERS = ("Index", "Middle", "Ring", "Little")


# ------------------------------------------------------------------ pose <-> editable JSON
def pose_to_spec(pose):
    """poses.py pose dict (rotation matrices, tuples) -> editable JSON-able spec."""
    spec = {"root": list(map(float, pose.get("root", (0.0, 0.0, 0.0)))), "rot": {}, "ik": {}, "hand": {}, "foot": {},
            "grip": {"Left": 0.0, "Right": 0.0}, "ground": float(pose.get("ground", 0.0))}
    for j, M in pose.get("rot", {}).items():
        # world-aligned rotation as (bend about Y, side bend about X, twist about Z) in degrees
        y, x, z = R.from_matrix(M).as_euler("YXZ", degrees=True)
        spec["rot"][j] = [float(y), float(x), float(z)]
    for limb, (target, pole) in pose.get("ik", {}).items():
        spec["ik"][limb] = {"target": None if target is None else list(map(float, target)), "pole": list(map(float, pole))}
    for limb, h in pose.get("hand", {}).items():
        spec["hand"][limb] = {k: (list(map(float, v)) if isinstance(v, (tuple, list, np.ndarray)) else v) for k, v in h.items()}
    for limb, M in pose.get("foot", {}).items():
        spec["foot"][limb] = list(map(float, R.from_matrix(M).as_euler("YXZ", degrees=True)))
    return spec


def spec_to_pose(spec, rig):
    pose = {"root": tuple(spec["root"]), "ground": spec.get("ground", 0.0), "rot": {}, "ik": {}, "hand": {}, "foot": {}}
    for j, (y, x, z) in spec["rot"].items():
        pose["rot"][j] = R.from_euler("YXZ", [y, x, z], degrees=True).as_matrix()
    for limb, e in spec["ik"].items():
        pose["ik"][limb] = (None if e["target"] is None else tuple(e["target"]), tuple(e["pole"]))
    for limb, h in spec["hand"].items():
        pose["hand"][limb] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in h.items()}
    for limb, a in spec.get("foot", {}).items():
        pose["foot"][limb] = R.from_euler("YXZ", a, degrees=True).as_matrix()
    for side, g in spec.get("grip", {}).items():  # finger curl about each joint's flexion axis (-X, OpenXR)
        if abs(g) < 1e-4:
            continue
        for f in FINGERS:
            for k, deg in (("Proximal", 60), ("Intermediate", 70), ("Distal", 45)):
                n = f"{side}Hand{f}{k}"
                pose["rot"][n] = R.from_rotvec(-rig.B[rig.index[n]][:, 0] * np.radians(deg * g)).as_matrix()
        for k, deg in (("ThumbMetacarpal", 25), ("ThumbProximal", 30), ("ThumbDistal", 30)):
            n = f"{side}Hand{k}"
            pose["rot"][n] = R.from_rotvec(-rig.B[rig.index[n]][:, 0] * np.radians(deg * g)).as_matrix()
    return pose


def load_library(rig):
    lib = {"bind": pose_to_spec({})}
    for name, pose in POSES.items():
        lib[name] = pose_to_spec(pose)
    if os.path.exists(USER_POSES):
        with open(USER_POSES) as f:
            lib.update(json.load(f))
    return lib


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
        self.viewer.set_camera(wp.vec3(3.3, -1.6, 1.3), pitch=-6.0, yaw=154.0)

        self.astro = AstronautRenderer(wp.get_device())
        self.rig = self.astro.rig
        self.library = load_library(self.rig)
        self.names = list(self.library)
        self.current = args.pose if args.pose in self.library else "stand"
        self.spec = copy.deepcopy(self.library[self.current])
        self.use_fbik = args.fullbody_ik
        self.fbik = None
        self.save_name = "my_pose"
        self.status = ""
        self.dirty = True
        self.metrics = {}
        self.W = None

    # ------------------------------------------------------------------ solve + draw
    def _solve(self):
        pose = spec_to_pose(self.spec, self.rig)
        W = solve(self.rig, pose)
        if self.use_fbik:
            if self.fbik is None:
                from kernels.fullbody_ik import FullBodyIK
                self.fbik = FullBodyIK(self.rig, wp.get_device())
            self.fbik.set_reference(W, seed=True)  # start at the edited pose; the IK only corrects it
            self.fbik.solve(20)
            W = self._keep_fingers(W, self.fbik.rig_world(self.fbik.body_q().numpy()))
        self.W = W
        rig = self.rig
        m = {}
        for s in ("Left", "Right"):
            u, l, w = (rig.index[f"{s}{n}"] for n in ("ArmUpper", "ArmLower", "HandWrist"))
            m[f"{s} shoulder margin (deg)"] = -shoulder_violation_deg(rig, W, s, W[l, :3, 3])
            m[f"{s} arm clearance (cm)"] = 100 * body_clearance(rig, W, [(W[u, :3, 3], W[l, :3, 3]), (W[l, :3, 3], W[w, :3, 3])])
        self.metrics = m
        self.astro.skin(W, ground=True)
        self.dirty = False

    def _keep_fingers(self, W_ref, W_body):
        """Full-body IK poses the wrists; carry the edited finger rotations over from the two-bone solution."""
        rig, W = self.rig, W_body.copy()
        for side in ("Left", "Right"):
            wi = rig.index[f"{side}HandWrist"]
            M = W[wi] @ np.linalg.inv(W_ref[wi])
            for i, n in enumerate(rig.names):
                if n.startswith(f"{side}Hand") and n != f"{side}HandWrist":
                    W[i] = M @ W_ref[i]
        return W

    def step(self):
        if self.dirty:
            self._solve()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.astro.log(self.viewer)
        self.viewer.end_frame()

    # ------------------------------------------------------------------ editor GUI
    def _load(self, name):
        self.current, self.spec, self.dirty = name, copy.deepcopy(self.library[name]), True

    def _slider(self, ui, label, value, lo, hi):
        changed, v = ui.slider_float(label, float(value), lo, hi)
        self.dirty |= changed
        return v

    def gui(self, ui):
        changed, idx = ui.combo("Pose", self.names.index(self.current), self.names)
        if changed:
            self._load(self.names[idx])
        if ui.button("Reset to library pose"):
            self._load(self.current)
        changed, self.use_fbik = ui.checkbox("Refine with full-body IK (GPU)", self.use_fbik)
        self.dirty |= changed
        for k, v in self.metrics.items():
            ui.text(f"{k}: {v:+.1f}" + ("  (!)" if v < 0 else ""))

        if ui.collapsing_header("Body"):
            for a, lab in enumerate(("root x", "root y", "root z")):
                self.spec["root"][a] = self._slider(ui, f"{lab}##root", self.spec["root"][a], -0.5, 0.5)
            for j in ROT_JOINTS:
                ang = self.spec["rot"].setdefault(j, [0.0, 0.0, 0.0])
                for a, lab in enumerate(ROT_LABELS):
                    ang[a] = self._slider(ui, f"{j} {lab}##{j}{a}", ang[a], -60.0, 60.0)

        if ui.collapsing_header("Hands and feet (IK targets)"):
            for limb in LIMBS:
                side = limb[:-3]
                hand = self.spec["hand"].get(limb)
                entry = self.spec["ik"].get(limb)
                if hand is not None:
                    t = hand["target"]
                elif entry is not None and entry["target"] is not None:
                    t = entry["target"]
                else:
                    on = False
                    changed, on = ui.checkbox(f"Pose {limb}##en{limb}", on)
                    if changed and on:  # start from where the limb currently is
                        end = {"LeftArm": "LeftHandWrist", "RightArm": "RightHandWrist", "LeftLeg": "LeftFootAnkle",
                               "RightLeg": "RightFootAnkle"}[limb]
                        p = self.W[self.rig.index[end], :3, 3].tolist() if self.W is not None else [0.0, 0.0, 1.0]
                        pole = [1.0, 0.0, 0.0] if limb.endswith("Leg") else [-0.45, 0.55 if side == "Left" else -0.55, -0.7]
                        self.spec["ik"][limb] = {"target": p, "pole": pole}
                        self.dirty = True
                    continue
                ui.text(limb + (" (hand target = index fingertip contact)" if hand is not None else ""))
                for a, lab in enumerate("xyz"):
                    t[a] = self._slider(ui, f"{limb} {lab}##{limb}{a}", t[a], -0.8 if a < 2 else 0.0, 0.8 if a < 2 else 2.0)
                if ui.button(f"Release {limb}##rel{limb}"):
                    self.spec["ik"].pop(limb, None)
                    self.spec["hand"].pop(limb, None)
                    self.dirty = True
            for side in ("Left", "Right"):
                self.spec["grip"][side] = self._slider(ui, f"{side} grip##grip{side}", self.spec["grip"].get(side, 0.0), -0.5, 1.0)

        if ui.collapsing_header("Save / export"):
            _, self.save_name = ui.input_text("Name", self.save_name)
            if ui.button("Save pose to data/user_poses.json"):
                self.status = self.save(self.save_name)
            if ui.button("Export posed USD for Cycles (build/hero_<name>.usd)"):
                self.status = self.export(self.save_name)
            if self.status:
                ui.text(self.status)

    # ------------------------------------------------------------------ persistence
    def save(self, name, path=USER_POSES):
        name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name.strip()) or "my_pose"
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        data[name] = copy.deepcopy(self.spec)
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
        self.library[name] = copy.deepcopy(self.spec)
        if name not in self.names:
            self.names.append(name)
        self.current = name
        return f"saved '{name}' to {os.path.relpath(path, REPO)}"

    def export(self, name):
        bake(self.rig, name, spec_to_pose(self.spec, self.rig), RIG)
        return f"wrote build/hero_{name}.usd (render: blender -b --python scripts/render_hero.py)"

    def test_final(self):
        import tempfile
        for name in self.names:  # every library pose solves and skins cleanly
            self._load(name)
            self._solve()
            pts = self.astro.points.numpy()
            assert np.isfinite(pts).all(), name
            assert self.metrics["Left arm clearance (cm)"] > -5 and self.metrics["Right arm clearance (cm)"] > -5, (name, self.metrics)
        with tempfile.TemporaryDirectory() as d:  # save/load round trip
            p = os.path.join(d, "poses.json")
            self._load("salute")
            self.spec["grip"]["Left"] = 0.5
            self.save("roundtrip", p)
            with open(p) as f:
                back = json.load(f)["roundtrip"]
            W1 = solve(self.rig, spec_to_pose(self.spec, self.rig))
            W2 = solve(self.rig, spec_to_pose(back, self.rig))
            assert np.abs(W1 - W2).max() < 1e-6
        print(f"poses: {len(self.names)} library poses solve; save/load round trip ok")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--pose", default="stand", help="Pose to start from (library or saved name).")
        parser.add_argument("--fullbody-ik", action="store_true", help="Start with full-body IK refinement on.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
