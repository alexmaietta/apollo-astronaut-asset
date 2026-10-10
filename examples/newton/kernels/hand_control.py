"""GPU (Warp) control kernels for the astronaut hands: everything in the control loop stays on the device, so a
whole step (task schedule -> joint targets -> wrist mount -> solver substeps) can be captured in one CUDA graph.

- `HandControlGPU.write_angle_targets`: bind-relative finger angles (Manus "ergonomics"-style input, or a scripted
  grip) -> clipped joint targets.
- `HandControlGPU.write_openxr_targets`: 26 OpenXR joint transforms per hand (a live glove stream copied to the
  device) -> joint targets, the same math as openxr_adapter.py but in a kernel.
- `task_kernel`: a scripted pick-and-place schedule driven by a device-side clock.
- `haptic_channels_kernel`: per-finger normal force from SensorContact's force matrices.
"""
import numpy as np
import warp as wp

from .openxr_adapter import OPENXR_JOINTS, rig_name


@wp.func
def smooth01(x: float):
    y = wp.clamp(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


@wp.kernel
def task_kernel(
    clock: wp.array[float], dt: float, auto: wp.array[wp.int32], manual_grip: wp.array[float],
    t_close: float, t_lift: float, t_lower: float, t_open: float, t_end: float, lift_height: float, grip_open: float,
    grip_gain: wp.array[float], open_scale: wp.array[float], lower: wp.array[float], upper: wp.array[float],
    coord: wp.array[wp.int32],
    mount_joint: int, mount_X0: wp.transform,
    joint_target_q: wp.array[float], joint_X_p: wp.array[wp.transform],
):
    d = wp.tid()
    t = clock[0]
    grip = manual_grip[0]
    lift = float(0.0)
    if auto[0] != 0:
        grip = grip_open + (1.0 - grip_open) * smooth01((t - t_close) / 1.0)
        grip = grip - (grip - grip_open) * smooth01((t - t_open) / 0.8)
        lift = lift_height * (smooth01((t - t_lift) / 1.2) - smooth01((t - t_lower) / 1.2))
    g = grip
    if g < 0.0:
        g = g * open_scale[d]  # per-DoF opening (e.g. the thumb does not open, it only wraps)
    joint_target_q[coord[d]] = wp.clamp(g * grip_gain[d], lower[d], upper[d])
    if d == 0:  # one thread moves the wrist mount (the tracked wrist) and advances the clock
        p = wp.transform_get_translation(mount_X0) + wp.vec3(0.0, 0.0, lift)
        joint_X_p[mount_joint] = wp.transform(p, wp.transform_get_rotation(mount_X0))
        tn = t + dt
        if auto[0] != 0 and tn > t_end:
            tn = 0.0
        clock[0] = tn


@wp.kernel
def scatter_mounts_kernel(poses: wp.array[wp.transform], mount_joint: wp.array[wp.int32], joint_X_p: wp.array[wp.transform]):
    """Move pose-driven mount joints (body-collider proxies) to their skeleton joints' world poses."""
    i = wp.tid()
    joint_X_p[mount_joint[i]] = poses[i]


@wp.func
def signed_angle(d0: wp.vec3, d1: wp.vec3, axis: wp.vec3):
    a = wp.normalize(axis)
    p0 = wp.normalize(d0 - wp.dot(d0, a) * a)
    p1 = wp.normalize(d1 - wp.dot(d1, a) * a)
    return wp.atan2(wp.dot(wp.cross(p0, p1), a), wp.dot(p0, p1))


@wp.kernel
def openxr_targets_kernel(
    frames: wp.array[wp.transform], parent: wp.array[wp.int32], child: wp.array[wp.int32], axis_col: wp.array[wp.int32],
    axis_sign: wp.array[float], bind: wp.array[float], lower: wp.array[float], upper: wp.array[float],
    coord: wp.array[wp.int32], joint_target_q: wp.array[float],
):
    d = wp.tid()
    qp = wp.transform_get_rotation(frames[parent[d]])
    qc = wp.transform_get_rotation(frames[child[d]])
    bone_p = -wp.quat_rotate(qp, wp.vec3(0.0, 0.0, 1.0))  # -Z runs down the bone
    bone_c = -wp.quat_rotate(qc, wp.vec3(0.0, 0.0, 1.0))
    ax = wp.vec3(1.0, 0.0, 0.0)
    if axis_col[d] == 1:
        ax = wp.vec3(0.0, 1.0, 0.0)
    axis = axis_sign[d] * wp.quat_rotate(qc, ax)
    q = signed_angle(bone_p, bone_c, axis) - bind[d]
    joint_target_q[coord[d]] = wp.clamp(q, lower[d], upper[d])


@wp.kernel
def angle_targets_kernel(angles: wp.array[float], bind: wp.array[float], lower: wp.array[float], upper: wp.array[float],
                         coord: wp.array[wp.int32], joint_target_q: wp.array[float]):
    d = wp.tid()
    joint_target_q[coord[d]] = wp.clamp(angles[d] - bind[d], lower[d], upper[d])


@wp.kernel
def haptic_channels_kernel(force: wp.array2d[wp.vec3], friction: wp.array2d[wp.vec3], row_channel: wp.array[wp.int32],
                           n_cols: int, out: wp.array[float]):
    r = wp.tid()
    f = wp.vec3(0.0)
    for j in range(n_cols):
        f += force[r, j] - friction[r, j]
    wp.atomic_add(out, row_channel[r], wp.length(f))


class HandControlGPU:
    """Per-DoF tables for one or both hands, on the device."""

    def __init__(self, hands, retargeters, coord_of, device):
        self.device = device
        rows, frame_names = [], []
        for hand, rt in zip(hands, retargeters):
            side = hand.side
            names = [rig_name(side, n) for n in OPENXR_JOINTS]
            base = len(frame_names)
            frame_names += names
            fi = {n: base + i for i, n in enumerate(names)}
            h = lambda k: f"{side}Hand{k}"  # noqa: E731
            for j in hand.joints:
                n = j["name"]
                child = n.rsplit("_", 1)[0]  # e.g. RightHandIndexProximal
                parent_of = {"ThumbMetacarpal": "Wrist", "ThumbProximal": "ThumbMetacarpal", "ThumbDistal": "ThumbProximal"}
                for f in ("Index", "Middle", "Ring", "Little"):
                    parent_of.update({f + "Proximal": f + "Metacarpal", f + "Intermediate": f + "Proximal",
                                      f + "Distal": f + "Intermediate"})
                par = h(parent_of[child[len(h("")):]])  # same parent bones as openxr_adapter.dof_angles
                spread = n.endswith("_abd")
                rows.append(dict(parent=fi[par], child=fi[child], col=1 if spread else 0, sign=1.0 if spread else -1.0,
                                 bind=rt.bind_angles[n], lower=np.radians(j["lower_deg"]), upper=np.radians(j["upper_deg"]),
                                 coord=coord_of[n], name=n))
        self.rows = rows
        self.frame_names = frame_names
        a = lambda k, dt: wp.array(np.array([r[k] for r in rows], dtype=dt), device=device)  # noqa: E731
        self.parent, self.child, self.col = a("parent", np.int32), a("child", np.int32), a("col", np.int32)
        self.sign, self.bind = a("sign", np.float32), a("bind", np.float32)
        self.lower, self.upper, self.coord = a("lower", np.float32), a("upper", np.float32), a("coord", np.int32)
        self.n = len(rows)

    def write_openxr_targets(self, frames, joint_target_q):
        """frames: wp.array[wp.transform] in self.frame_names order (both hands' 26 OpenXR joints)."""
        wp.launch(openxr_targets_kernel, dim=self.n, inputs=[frames, self.parent, self.child, self.col, self.sign,
                  self.bind, self.lower, self.upper, self.coord], outputs=[joint_target_q], device=self.device)

    def write_angle_targets(self, angles, joint_target_q):
        """angles: wp.array[float] per DoF (radians, measured from an open flat hand)."""
        wp.launch(angle_targets_kernel, dim=self.n, inputs=[angles, self.bind, self.lower, self.upper, self.coord],
                  outputs=[joint_target_q], device=self.device)
