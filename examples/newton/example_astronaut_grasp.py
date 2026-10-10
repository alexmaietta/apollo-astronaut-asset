# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Astronaut Grasp
#
# The astronaut's physics hands (SimReady asset, Physics variant "mujoco":
# 20-DoF articulations, capsule fingers, A7L glove springs that scale with
# suit pressure) pick up an Apollo-style sample carrier by its handle under
# lunar gravity. Control runs entirely on the GPU: a device-side task clock
# drives the grip (bind-relative finger angles, the Manus "ergonomics" input
# path) and the wrist mount (the tracked wrist pose), and the whole step --
# control, collision, solver substeps, contact forces and per-finger haptics
# -- is one CUDA graph. A live OpenXR / Manus hand stream plugs into
# kernels.hand_control.HandControlGPU.write_openxr_targets instead of the task kernel.
#
# Haptics: per-finger normal grip force against the carrier, in Isaac Lab's
# glove format (thumb, index, middle, ring, pinky), ready for a Manus glove via
# any HapticFeedbackReceiver. The glove is skinned from the simulated finger
# bodies on the GPU; the arm follows the wrist.
#
# GUI: scripted task on/off, manual grip, suit pressure, hand colliders.
#
# Command (from the astronaut repo root, after scripts/package_simready.py):
#   python examples/newton/example_astronaut_grasp.py
#   ... --pressure 0      # unpressurized glove for comparison
#
###########################################################################

import argparse
import os
import sys

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ModelFlags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import REPO  # noqa: E402  (also puts scripts/ on sys.path)
from kernels.hand_control import HandControlGPU, scatter_mounts_kernel, task_kernel  # noqa: E402
from hand_physics import HandModel, load_params  # noqa: E402
from kernels.haptics import CHANNELS, FingerHaptics  # noqa: E402
from kernels.skinning import ASSET, AstronautRenderer  # noqa: E402
from kernels.openxr_adapter import HandRetargeter  # noqa: E402
from poses import solve  # noqa: E402

G_MOON = 1.62
SIDE = "Right"
CARRIER_MASS = 1.2  # [kg] sample carrier (lunar weight ~1.9 N)
HANDLE_RADIUS = 0.010  # [m] tool-style handle; fits the gap between the closed glove fingers and the palm
LIFT = 0.15  # [m]
GRIP_OPEN = -0.4  # the operator starts with the hand opened a little
# full-grip finger flexion relative to the bind pose [deg] (the operator's power grasp)
GRIP_DEG = {"Proximal_flex": 55.0, "Intermediate_flex": 60.0, "Distal_flex": 35.0,
            "ThumbMetacarpal_flex": 40.0, "ThumbProximal_flex": 35.0, "ThumbDistal_flex": 35.0}
FINGER_SCALE = {"Index": 1.0, "Middle": 1.0, "Ring": 1.0, "Little": 1.0, "Thumb": 1.0}  # per-finger grip strength
HANDLE_MU = 0.9  # handle friction (a low value acts like a free-turning handle sleeve: the hook grip carries the load)
HANDLE_TO_PALM = 0.15
HANDLE_HALF = 0.085  # [m] half length of the handle; the posts stand at its ends  # handle position: fraction of the way from the closed fingers' middles to the palm box
# scripted task (seconds): close the hand, lift, hold, set down, open
T_CLOSE, T_LIFT, T_HOLD, T_LOWER, T_OPEN, T_END = 0.5, 2.0, 3.5, 5.5, 7.0, 8.0


def grip_gain(joint_name):
    scale = next((v for f, v in FINGER_SCALE.items() if f"Hand{f}" in joint_name), 1.0)
    for key in ("ThumbMetacarpal_flex", "ThumbProximal_flex", "ThumbDistal_flex"):
        if joint_name.endswith(key):
            return GRIP_DEG[key] * scale
    if "Thumb" in joint_name or joint_name.endswith("_abd"):
        return 0.0
    return scale * next((v for k, v in GRIP_DEG.items() if joint_name.endswith(k)), 0.0)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        FINGER_SCALE["Thumb"] = args.thumb
        self.lift = args.lift
        self.params = load_params(pressure_psi=args.pressure)
        self.pressure = float(args.pressure)
        self.show_colliders = False
        device = wp.get_device()

        from pxr import Usd

        stage = Usd.Stage.Open(args.usd)
        stage.GetDefaultPrim().GetVariantSet("Physics").SetVariantSelection("mujoco")
        stage.GetDefaultPrim().GetVariantSet("BodyColliders").SetVariantSelection(args.body_colliders)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -G_MOON))
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.005
        builder.add_usd(stage, ignore_paths=[".*/Geometry.*", ".*/Looks.*"], collapse_fixed_joints=False,
                        enable_self_collisions=False, hide_collision_shapes=True)
        hand_bodies = builder.body_count

        self.astro = AstronautRenderer(device)
        rig = self.astro.rig
        self.hands = {s: HandModel(rig, s, self.params) for s in ("Left", "Right")}
        self.hand_T = self._hand_transform(rig, np.asarray(args.hand_offset, float))
        self.carrier = self._add_carrier(builder, rig)
        builder.add_ground_plane(color=(0.42, 0.41, 0.39))
        self.model = builder.finalize()

        labels = [s.rsplit("/", 1)[-1] for s in builder.joint_label]
        qs, ds = self.model.joint_q_start.numpy(), self.model.joint_qd_start.numpy()
        joints = [j for s in ("Left", "Right") for j in self.hands[s].joints]
        self.ctl = HandControlGPU([self.hands[s] for s in ("Left", "Right")],
                                  [HandRetargeter(rig, self.hands[s]) for s in ("Left", "Right")],
                                  {j["name"]: int(qs[labels.index(j["name"])]) for j in joints}, device)
        gains = [np.radians(grip_gain(r["name"])) if r["name"].startswith(SIDE) else 0.0 for r in self.ctl.rows]
        self.grip_gain = wp.array(np.array(gains, np.float32), device=device)
        # the operator opens the fingers but not the thumb, which only wraps over the handle
        self.open_scale = wp.array(np.array([0.0 if "Thumb" in r["name"] else 1.0 for r in self.ctl.rows], np.float32),
                                   device=device)
        self.mount = labels.index(f"{SIDE}WristMount")
        Xp = self.model.joint_X_p.numpy()
        Xp[self.mount] = self._apply_hand_T(Xp[self.mount])  # start pose of the tracked wrist (hand and carrier)
        self.model.joint_X_p.assign(Xp)
        self.mount_X0 = wp.transform(*Xp[self.mount])
        self.suit_dofs = np.array([int(ds[labels.index(j["name"])]) for j in joints])
        if args.servo_kp is not None:  # override the asset's teleop servo gains (operator grip strength)
            ke, kd = self.model.joint_target_ke.numpy(), self.model.joint_target_kd.numpy()
            ke[self.suit_dofs] = args.servo_kp
            kd[self.suit_dofs] = args.servo_kp * 0.02
            self.model.joint_target_ke.assign(ke)
            self.model.joint_target_kd.assign(kd)
        self.k_ref = np.array([self.params["joints"][j["kind"]]["suit_stiffness_nm_per_rad"] for j in joints])
        self._apply_pressure()

        # pose-driven suit body colliders (BodyColliders variant): mount joint -> skeleton joint
        self.collider_mounts, self.collider_joints = [], []
        for k, path in enumerate(builder.joint_label):
            prim = stage.GetPrimAtPath(path)
            if "/BodyColliders/" in path and prim and prim.GetCustomDataByKey("astronaut:skinJoint"):
                self.collider_mounts.append(k)
                self.collider_joints.append(rig.index[prim.GetCustomDataByKey("astronaut:skinJoint")])
        n = max(len(self.collider_mounts), 1)
        self.collider_X = wp.array(np.tile(np.array([0, 0, 0, 0, 0, 0, 1], np.float32), (n, 1)), dtype=wp.transform, device=device)
        self.collider_mount_idx = wp.array(np.array(self.collider_mounts or [0], np.int32), device=device)
        self.body_names = [s.rsplit("/", 1)[-1] for s in builder.body_label]
        self.astro.bind_hands({n: i for i, n in enumerate(self.body_names[:hand_bodies])})
        self.wrist_body = self.body_names.index(f"{SIDE}HandWrist")

        self.mujoco_contacts = args.mujoco_contacts
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast", cone="elliptic", impratio=20.0,
            iterations=100, ls_iterations=50, njmax=600, nconmax=400, use_mujoco_contacts=self.mujoco_contacts)
        self.haptics = FingerHaptics(self.model, self.model.body_label, [self.carrier])
        # start in the opened hand the task begins with (the bind pose's fingers would overlap the handle)
        q = self.model.joint_q.numpy()
        for r, gain, o in zip(self.ctl.rows, self.grip_gain.numpy(), self.open_scale.numpy()):
            q[r["coord"]] = np.clip(GRIP_OPEN * o * gain, r["lower"], r["upper"])
        self.model.joint_q.assign(q)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        if self.collider_mounts:
            Xp = self.model.joint_X_p.numpy()
            self.collider_X.assign(Xp[self.collider_mounts])  # authored bind-pose mounts until the first IK update
        self._check_initial_overlap()

        self.clock = wp.zeros(1, dtype=float, device=device)
        self.auto = wp.array([1], dtype=wp.int32, device=device)
        self.manual_grip = wp.array([GRIP_OPEN], dtype=float, device=device)
        self._auto_host, self._grip_host = True, GRIP_OPEN
        self.peak = np.zeros(len(CHANNELS))
        self._hold_sum, self._hold_n = np.zeros(len(CHANNELS)), 0
        self.hold_force = None
        self.carrier_z0 = float(self.state_0.body_q.numpy()[self.carrier][2])
        self.lift_at_hold = None
        self._frames = 0
        self._arm_W, self._arm_wrist = None, None
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.85, -1.05, 1.05), pitch=-15.0, yaw=125.0)
        self.capture()

    def _hand_transform(self, rig, offset):
        """Rigid transform that moves the hand (and the carrier under it) from the bind pose by `offset` [m, world].
        The hand also rolls about the forward axis (the handle axis) by the extra arm abduction, so the wrist stays
        straight and the handle level."""
        P = lambda n: rig.bind[rig.index[n], :3, 3]  # noqa: E731
        w0, sh = P(f"{SIDE}HandWrist"), P(f"{SIDE}ArmUpper")
        if np.any(offset):  # swing the arm about the shoulder (the bind arm is nearly straight: a plain translation
            # would put the wrist out of reach), keeping the elbow slightly bent at 97% of the bind reach
            v = w0 + offset - sh
            offset = sh + v * (0.97 * np.linalg.norm(w0 - sh) / np.linalg.norm(v)) - w0
        ang = lambda v: np.arctan2(v[1], -v[2])  # noqa: E731  (side angle of the arm, seen along X)
        roll = ang(w0 + offset - sh) - ang(w0 - sh)
        c, s_ = np.cos(roll), np.sin(roll)
        T = np.eye(4)
        T[:3, :3] = [[1, 0, 0], [0, c, -s_], [0, s_, c]]
        T[:3, 3] = w0 + offset - T[:3, :3] @ w0
        return T

    def _apply_hand_T(self, xform):
        from scipy.spatial.transform import Rotation as R
        p = self.hand_T[:3, :3] @ xform[:3] + self.hand_T[:3, 3]
        q = (R.from_matrix(self.hand_T[:3, :3]) * R.from_quat(xform[3:])).as_quat()
        return np.concatenate([p, q]).astype(np.float32)

    def _add_carrier(self, builder, rig):
        """Sample carrier whose handle sits between the closed glove fingers and the palm, on a pedestal."""
        from scipy.spatial.transform import Rotation as R
        hand = self.hands[SIDE]
        P = lambda n: rig.bind[rig.index[n], :3, 3]  # noqa: E731
        axis = P(f"{SIDE}HandIndexProximal") - P(f"{SIDE}HandLittleProximal")
        axis /= np.linalg.norm(axis)
        # closed-hand forward kinematics of the scripted grip: the handle goes 15% of the way from the closed
        # fingers' middles (all four, so it is centred under the hand) toward the palm box
        local = {}
        for b in hand.bodies:
            n = b["name"]
            for k, deg in GRIP_DEG.items():
                if n.endswith(k.split("_")[0]) and ("Thumb" in k) == ("Thumb" in n):
                    local[n] = R.from_rotvec(-rig.B[rig.index[n]][:, 0] * np.radians(deg)).as_matrix()
        W = rig.fk(local)
        mid = np.mean([(W[rig.index[f"{SIDE}Hand{f}Intermediate"], :3, 3] + W[rig.index[f"{SIDE}Hand{f}Distal"], :3, 3]) / 2
                       for f in ("Index", "Middle", "Ring", "Little")], 0)
        palm = hand.bodies[0]["colliders"][0]["center"]
        handle_c = mid + HANDLE_TO_PALM * (palm - mid)
        handle_c = self.hand_T[:3, :3] @ handle_c + self.hand_T[:3, 3]
        axis = self.hand_T[:3, :3] @ axis
        self.handle_c = handle_c
        box_half, drop = np.array([0.09, 0.045, 0.05]), 0.11
        box_c = handle_c - np.array([0.0, 0.0, drop + box_half[2]])
        body = builder.add_body(xform=wp.transform(wp.vec3(*box_c), wp.quat_identity()), label="sample_carrier")
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.9, ke=2.0e4, kd=2.0e2)
        builder.add_shape_box(body, hx=box_half[0], hy=box_half[1], hz=box_half[2], cfg=cfg)
        q_handle = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*axis))
        builder.add_shape_capsule(body, xform=wp.transform(wp.vec3(*(handle_c - box_c)), q_handle),
                                  radius=HANDLE_RADIUS, half_height=HANDLE_HALF,
                                  cfg=newton.ModelBuilder.ShapeConfig(mu=HANDLE_MU, ke=2.0e4, kd=2.0e2))
        for s in (-1, 1):  # posts outside the gloved hand's ~10 cm width
            post = handle_c - box_c + s * HANDLE_HALF * axis - [0.0, 0.0, drop / 2]
            builder.add_shape_box(body, xform=wp.transform(wp.vec3(*post), wp.quat_identity()),
                                  hx=0.006, hy=0.006, hz=drop / 2, cfg=cfg)
        builder.body_mass[body] = CARRIER_MASS
        builder.body_inv_mass[body] = 1.0 / CARRIER_MASS
        top = box_c[2] - box_half[2]
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(box_c[0], box_c[1], top / 2), wp.quat_identity()),
                              hx=0.09, hy=0.07, hz=top / 2, cfg=cfg, label="pedestal")
        return body

    def _update_colliders(self, W):
        """Body-collider mounts follow the skeleton (read by the captured graph on the next step)."""
        if not self.collider_mounts:
            return
        from scipy.spatial.transform import Rotation as R
        M = W[self.collider_joints]
        X = np.concatenate([M[:, :3, 3], R.from_matrix(M[:, :3, :3]).as_quat()], 1).astype(np.float32)
        self.collider_X.assign(X)

    def _check_initial_overlap(self, tol=0.001):
        """Fail loudly if the scene starts interpenetrating (that launches the carrier)."""
        self.collision_pipeline.collide(self.state_0, self.contacts)
        n = int(self.contacts.rigid_contact_count.numpy()[0])
        if n == 0:
            return
        body = self.model.shape_body.numpy()
        s0, s1 = self.contacts.rigid_contact_shape0.numpy()[:n], self.contacts.rigid_contact_shape1.numpy()[:n]
        p0, p1 = self.contacts.rigid_contact_point0.numpy()[:n], self.contacts.rigid_contact_point1.numpy()[:n]
        nrm, bq = self.contacts.rigid_contact_normal.numpy()[:n], self.state_0.body_q.numpy()

        def world(b, p):
            if b < 0:
                return np.asarray(p)
            return bq[b][:3] + np.array(wp.quat_rotate(wp.quat(*bq[b][3:]), wp.vec3(*p)))

        worst = 0.0
        for k in range(n):
            b0, b1 = body[s0[k]], body[s1[k]]
            if self.carrier in (b0, b1) and -1 not in (b0, b1):
                worst = min(worst, float(np.dot(world(b1, p1[k]) - world(b0, p0[k]), nrm[k])))
        if worst < -tol:
            raise RuntimeError(f"the hand starts {-worst * 1000:.1f} mm inside the sample carrier")

    def _apply_pressure(self):
        k = self.model.mujoco.dof_passive_stiffness.numpy()
        k[self.suit_dofs] = self.k_ref * self.pressure / self.params["reference_pressure_psi"]
        self.model.mujoco.dof_passive_stiffness.assign(k)
        if hasattr(self, "solver"):
            self.solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

    # ------------------------------------------------------------------ sim loop (one CUDA graph per frame)
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.launch(task_kernel, dim=self.ctl.n, inputs=[
            self.clock, self.frame_dt, self.auto, self.manual_grip, T_CLOSE, T_LIFT, T_LOWER, T_OPEN, T_END, self.lift,
            GRIP_OPEN, self.grip_gain, self.open_scale, self.ctl.lower, self.ctl.upper, self.ctl.coord, self.mount,
            self.mount_X0],
            outputs=[self.control.joint_target_q, self.model.joint_X_p])
        if self.collider_mounts:
            wp.launch(scatter_mounts_kernel, dim=len(self.collider_mounts), inputs=[self.collider_X, self.collider_mount_idx],
                      outputs=[self.model.joint_X_p])
        self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)  # mounts moved (graph-capturable)
        if not self.mujoco_contacts:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None if self.mujoco_contacts else self.contacts,
                             self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)
        self.haptics.update(self.state_0, self.contacts)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._frames += 1
        t = self._frames * self.frame_dt % T_END if self._auto_host else 0.0
        if self._auto_host and T_HOLD <= t < T_LOWER and self.lift_at_hold is None:
            v = self.haptics.values["right"]  # mean grip while holding (contact transients excluded)
            self.peak = np.maximum(self.peak, v)
            self._hold_sum += v
            self._hold_n += 1
        if self._auto_host and self.lift_at_hold is None and t >= T_HOLD + 1.0:
            self.lift_at_hold = float(self.state_0.body_q.numpy()[self.carrier][2]) - self.carrier_z0
            self.hold_force = self._hold_sum / max(self._hold_n, 1)
            print(f"grasp: carrier lifted {self.lift_at_hold * 100:.1f} cm at {self.pressure:.1f} psi; mean grip while "
                  "holding [N] " + ", ".join(f"{c} {f:.1f}" for c, f in zip(CHANNELS, self.hold_force)))

    def render(self):
        rig = self.astro.rig
        wrist = self.state_0.body_q.numpy()[self.wrist_body]  # one transform for the arm IK
        if self._arm_W is None or np.linalg.norm(wrist[:3] - self._arm_wrist) > 0.002:
            # arm follows the wrist; the elbow swivel avoids the suit body (poses.elbow_position)
            from scipy.spatial.transform import Rotation as R
            bindW = rig.bind[rig.index[f"{SIDE}HandWrist"]]
            Rw = R.from_quat(wrist[3:]).as_matrix() @ bindW[:3, :3].T  # simulated wrist rotation from the bind pose
            pose = dict(ik={"LeftLeg": ((0.02, 0.24, 0.106), (1, 0, 0)), "RightLeg": ((-0.06, -0.13, 0.106), (1, 0, 0))},
                        hand={f"{SIDE}Arm": dict(contact=f"{SIDE}HandWrist", target=tuple(wrist[:3]),
                                                 fingers=tuple(Rw @ (rig.bind[rig.index[f"{SIDE}HandMiddleProximal"], :3, 3] - bindW[:3, 3])),
                                                 dorsal=tuple(Rw @ bindW[:3, 1]))})
            self._arm_W, self._arm_wrist = solve(rig, pose), wrist[:3].copy()
            self._update_colliders(self._arm_W)
        self.astro.skin(self._arm_W, body_q=self.state_0.body_q)  # hands posed from the bodies on the GPU
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.astro.log(self.viewer, hidden=self.show_colliders)
        self.viewer.end_frame()

    def gui(self, ui):
        changed, self._auto_host = ui.checkbox("Scripted pick-and-place", self._auto_host)
        if changed:
            self.auto.assign(np.array([int(self._auto_host)], np.int32))
        if not self._auto_host:
            ui.text("Grip")
            changed, self._grip_host = ui.slider_float("##grip", self._grip_host, GRIP_OPEN, 1.0)
            if changed:
                self.manual_grip.assign(np.array([self._grip_host], np.float32))
        ui.text("Suit pressure (psi)")
        changed, p = ui.slider_float("##pressure", self.pressure, 0.0, 4.3)
        if changed:
            self.pressure = p
            self._apply_pressure()
        changed, self.show_colliders = ui.checkbox("Show hand colliders (hide suit)", self.show_colliders)
        if changed and hasattr(self.viewer, "show_collision"):
            self.viewer.show_collision = self.show_colliders
        ui.text("Haptics, right glove [N]")
        for c, f in zip(CHANNELS, self.haptics.values["right"]):
            ui.text(f"  {c:7s} {f:5.2f} " + "#" * int(min(f, 10.0) * 3))

    def test_final(self):
        assert self.lift_at_hold is not None and self.lift_at_hold > 0.05, f"carrier not lifted ({self.lift_at_hold})"
        assert (self.hold_force > 0.2).sum() >= 2, f"too few fingers gripping: {self.hold_force}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--usd", default=ASSET, help="Path to the astronaut SimReady asset.")
        parser.add_argument("--pressure", type=float, default=3.7, help="Suit pressure [psi] (glove stiffness scale).")
        parser.add_argument("--mujoco-contacts", action=argparse.BooleanOptionalAction, default=True,
                            help="MuJoCo collision/contacts (default; robust grasps) or Newton's pipeline (--no-mujoco-contacts).")
        parser.add_argument("--substeps", type=int, default=10, help="Solver substeps per 60 Hz frame.")
        parser.add_argument("--body-colliders", default="capsules", choices=["none", "capsules"],
                            help="BodyColliders variant: pose-driven suit proxies the hands and objects collide with.")
        parser.add_argument("--lift", type=float, default=LIFT, help="Lift height of the scripted task [m].")
        parser.add_argument("--hand-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                            help="Start the hand (and carrier) this far from the bind pose [m, world: +X forward, "
                                 "-Y out to the right], swung about the shoulder so the arm can reach; the hand "
                                 "rolls with the arm's abduction.")
        parser.add_argument("--thumb", type=float, default=1.0,
                            help="Thumb grip scale: 0 = hook grip (the fingers carry the handle, the thumb stays open).")
        parser.add_argument("--servo-kp", type=float, default=None,
                            help="Override the teleop finger servo gain [N*m/rad] (operator grip strength).")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
