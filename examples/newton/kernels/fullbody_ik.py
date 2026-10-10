"""GPU full-body IK for the astronaut (Newton IK: batched Levenberg-Marquardt on Warp).

A kinematic Newton model of the skeleton: free root at the hips, 3-DoF spine joints (lower/middle/upper/chest),
3-DoF shoulders, hinge elbows, 3-DoF wrists, 3-DoF hips, hinge knees, 3-DoF ankles (46 DoF). Each body sits at its
rig joint's bind frame (OpenXR axes), so a solved body pose is that skeleton joint's world matrix.

Objectives (all evaluated on the GPU):
- hand / foot position and orientation targets,
- a low-weight orientation prior toward a reference pose (e.g. a retargeted motion frame), so the solver corrects
  the reference instead of inventing a new pose,
- joint limits (elbow/knee bend, spine, twist) from data/suit_params.json body_rom,
- the A7L shoulder range of motion as a direction-dependent cone (custom kernel),
- body self-collision: capsule-capsule penetration between limbs and the torso / each other (custom kernel).
"""
import numpy as np
import warp as wp

import newton
import newton.ik as ik

from hand_physics import load_params
from poses import TORSO_CAPSULES

SIDES = ("Left", "Right")


@wp.func
def _softplus(x: float, s: float):
    return wp.log(1.0 + wp.exp(x / s)) * s


@wp.func
def _seg_seg(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1) + 1.0e-9
    e = wp.dot(d2, d2) + 1.0e-9
    f = wp.dot(d2, r)
    c = wp.dot(d1, r)
    b = wp.dot(d1, d2)
    den = a * e - b * b
    s = float(0.0)
    if den > 1.0e-9:
        s = wp.clamp((b * f - c * e) / den, 0.0, 1.0)
    t = wp.clamp((b * s + f) / e, 0.0, 1.0)
    s = wp.clamp((b * t - c) / a, 0.0, 1.0)
    return wp.length(p1 + d1 * s - (p2 + d2 * t))


@wp.func
def _closest(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1) + 1.0e-9
    e = wp.dot(d2, d2) + 1.0e-9
    f = wp.dot(d2, r)
    c = wp.dot(d1, r)
    b = wp.dot(d1, d2)
    den = a * e - b * b
    s = float(0.0)
    if den > 1.0e-9:
        s = wp.clamp((b * f - c * e) / den, 0.0, 1.0)
    t = wp.clamp((b * s + f) / e, 0.0, 1.0)
    s = wp.clamp((b * t - c) / a, 0.0, 1.0)
    return p1 + d1 * s, p2 + d2 * t


@wp.kernel
def _collision_jacobian(
    body_q: wp.array2d[wp.transform], joint_S_s: wp.array2d[wp.spatial_vector], body_a: wp.array[wp.int32],
    a0: wp.array[wp.vec3], a1: wp.array[wp.vec3], ra: wp.array[float], body_b: wp.array[wp.int32],
    b0: wp.array[wp.vec3], b1: wp.array[wp.vec3], rb: wp.array[float], affects_a: wp.array2d[wp.uint8],
    affects_b: wp.array2d[wp.uint8], margin: float, weight: float, start_idx: int, jacobian: wp.array3d[float],
):
    # d r / d q = r'(d) * n . (v_A(pA) - v_B(pB)): closest-point distance derivative (envelope theorem)
    row, k, dof = wp.tid()
    ta = body_q[row, body_a[k]]
    tb = body_q[row, body_b[k]]
    pa, pb = _closest(wp.transform_point(ta, a0[k]), wp.transform_point(ta, a1[k]),
                      wp.transform_point(tb, b0[k]), wp.transform_point(tb, b1[k]))
    delta = pa - pb
    d = wp.length(delta) + 1.0e-9
    n = delta / d
    x = (ra[k] + rb[k] + margin - d) / 0.01
    slope = -weight / (1.0 + wp.exp(-x))  # d softplus / d d
    S = joint_S_s[row, dof]
    v = wp.vec3(S[0], S[1], S[2])
    w = wp.vec3(S[3], S[4], S[5])
    vel = wp.vec3(0.0)
    if affects_a[k, dof] != 0:
        vel = vel + v + wp.cross(w, pa)
    if affects_b[k, dof] != 0:
        vel = vel - (v + wp.cross(w, pb))
    jacobian[row, start_idx + k, dof] = slope * wp.dot(n, vel)


@wp.kernel
def _collision_residuals(
    body_q: wp.array2d[wp.transform], body_a: wp.array[wp.int32], a0: wp.array[wp.vec3], a1: wp.array[wp.vec3],
    ra: wp.array[float], body_b: wp.array[wp.int32], b0: wp.array[wp.vec3], b1: wp.array[wp.vec3], rb: wp.array[float],
    margin: float, weight: float, start_idx: int, residuals: wp.array2d[float],
):
    row, k = wp.tid()
    ta = body_q[row, body_a[k]]
    tb = body_q[row, body_b[k]]
    d = _seg_seg(wp.transform_point(ta, a0[k]), wp.transform_point(ta, a1[k]),
                 wp.transform_point(tb, b0[k]), wp.transform_point(tb, b1[k]))
    residuals[row, start_idx + k] = weight * _softplus(ra[k] + rb[k] + margin - d, 0.01)


@wp.kernel
def _cone_residuals(
    body_q: wp.array2d[wp.transform], chest: int, chest_bind_rot: wp.quat, upper: wp.array[wp.int32],
    lower: wp.array[wp.int32], side_sign: wp.array[float], flex: float, ext: float, abd: float, add: float,
    weight: float, start_idx: int, residuals: wp.array2d[float],
):
    row, k = wp.tid()
    tc = body_q[row, chest]
    aligned = wp.transform_get_rotation(tc) * wp.quat_inverse(chest_bind_rot)  # chest world-aligned rotation
    u = wp.quat_rotate_inv(aligned, wp.transform_get_translation(body_q[row, lower[k]]) -
                           wp.transform_get_translation(body_q[row, upper[k]]))
    u = wp.normalize(u)
    theta = wp.acos(wp.clamp(-u[2], -0.9999, 0.9999))
    fwd = u[0]
    out = side_sign[k] * u[1]
    a = flex
    if fwd < 0.0:
        a = ext
    b = abd
    if out < 0.0:
        b = add
    h = wp.sqrt(fwd * fwd + out * out) + 1.0e-6
    cp = fwd / h
    sp = out / h
    tmax = 1.0 / wp.sqrt((cp / a) * (cp / a) + (sp / b) * (sp / b))
    residuals[row, start_idx + k] = weight * _softplus(theta - tmax, 0.03)


@wp.kernel
def _jac_row_fill(q_grad: wp.array2d[float], n_dofs: int, row: int, jacobian: wp.array3d[float]):
    b = wp.tid()
    for j in range(n_dofs):
        jacobian[b, row, j] = q_grad[b, j]


class AutodiffRows(ik.IKObjective):
    """Autodiff Jacobian for a multi-row custom objective: one seed vector per residual row is back-propagated
    through the solver's recorded tape (forward kinematics + this objective's residual kernel)."""

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        self._seeds = []
        for k in range(self.residual_dim()):
            e = np.zeros((self.n_batch, self.total_residuals), np.float32)
            e[:, self.residual_offset + k] = 1.0
            self._seeds.append(wp.array(e.ravel(), device=self.device))

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        for k, seed in enumerate(self._seeds):
            tape.backward(grads={tape.outputs[0]: seed})
            wp.launch(_jac_row_fill, dim=self.n_batch, inputs=[tape.gradients[dq_dof], model.joint_dof_count, start_idx + k],
                      outputs=[jacobian], device=self.device)
            tape.zero()


@wp.kernel
def _cone_jacobian(
    body_q: wp.array2d[wp.transform], joint_S_s: wp.array2d[wp.spatial_vector], chest: int, chest_bind_rot: wp.quat,
    upper: wp.array[wp.int32], lower: wp.array[wp.int32], side_sign: wp.array[float], aff_chest: wp.array[wp.uint8],
    aff_upper: wp.array2d[wp.uint8], aff_lower: wp.array2d[wp.uint8], flex: float, ext: float, abd: float, add: float,
    weight: float, start_idx: int, jacobian: wp.array3d[float],
):
    # u = A^T (p_lower - p_upper) in the chest's world-aligned frame; du = A^T (dx - w_chest x x); then chain rule
    # through the elevation theta = acos(-u_z) and the elliptical limit tmax(psi), psi = atan2(s u_y, u_x)
    row, k, dof = wp.tid()
    A = wp.transform_get_rotation(body_q[row, chest]) * wp.quat_inverse(chest_bind_rot)
    pu = wp.transform_get_translation(body_q[row, upper[k]])
    pl = wp.transform_get_translation(body_q[row, lower[k]])
    x = pl - pu
    S = joint_S_s[row, dof]
    v = wp.vec3(S[0], S[1], S[2])
    w = wp.vec3(S[3], S[4], S[5])
    dx = wp.vec3(0.0)
    if aff_lower[k, dof] != 0:
        dx = dx + v + wp.cross(w, pl)
    if aff_upper[k, dof] != 0:
        dx = dx - (v + wp.cross(w, pu))
    if aff_chest[dof] != 0:
        dx = dx - wp.cross(w, x)
    u = wp.quat_rotate_inv(A, x)
    du = wp.quat_rotate_inv(A, dx)
    n = wp.length(u) + 1.0e-9
    uh = u / n
    duh = (du - uh * wp.dot(uh, du)) / n
    uz = wp.clamp(-uh[2], -0.9999, 0.9999)
    theta = wp.acos(uz)
    dtheta = duh[2] / wp.sqrt(1.0 - uz * uz)  # d acos(-z) = dz / sqrt(1 - z^2)
    fwd = uh[0]
    out = side_sign[k] * uh[1]
    a = flex
    if fwd < 0.0:
        a = ext
    b = abd
    if out < 0.0:
        b = add
    h2 = fwd * fwd + out * out + 1.0e-9
    h = wp.sqrt(h2)
    cp = fwd / h
    sp = out / h
    g = (cp / a) * (cp / a) + (sp / b) * (sp / b)
    tmax = 1.0 / wp.sqrt(g)
    dpsi = (fwd * side_sign[k] * duh[1] - out * duh[0]) / h2
    dtmax = sp * cp * (1.0 / (a * a) - 1.0 / (b * b)) / (g * wp.sqrt(g)) * dpsi
    xs = (theta - tmax) / 0.03
    jacobian[row, start_idx + k, dof] = weight / (1.0 + wp.exp(-xs)) * (dtheta - dtmax)


class CapsuleCollisionObjective(AutodiffRows):
    def __init__(self, pairs, device, margin=0.005, weight=20.0, affects=None):
        super().__init__()
        self.affects_a = wp.array(np.array([affects[p[0]] for p in pairs], np.uint8), device=device)
        self.affects_b = wp.array(np.array([affects[p[4]] for p in pairs], np.uint8), device=device)
        def f(i, dt, typ=None):
            data = np.array([p[i] for p in pairs], dtype=dt)
            return wp.array(data, dtype=typ, device=device) if typ else wp.array(data, device=device)
        self.body_a, self.a0, self.a1, self.ra = f(0, np.int32), f(1, np.float32, wp.vec3), f(2, np.float32, wp.vec3), f(3, np.float32)
        self.body_b, self.b0, self.b1, self.rb = f(4, np.int32), f(5, np.float32, wp.vec3), f(6, np.float32, wp.vec3), f(7, np.float32)
        self.n, self.margin, self.weight, self.device = len(pairs), margin, weight, device

    def residual_dim(self):
        return self.n

    def supports_analytic(self):
        return True

    def init_buffers(self, model, jacobian_mode):
        pass  # analytic Jacobian: no autodiff seeds needed

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_collision_jacobian, dim=(body_q.shape[0], self.n, model.joint_dof_count), inputs=[
            body_q, joint_S_s, self.body_a, self.a0, self.a1, self.ra, self.body_b, self.b0, self.b1, self.rb,
            self.affects_a, self.affects_b, self.margin, self.weight, start_idx], outputs=[jacobian], device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_collision_residuals, dim=(body_q.shape[0], self.n), inputs=[
            body_q, self.body_a, self.a0, self.a1, self.ra, self.body_b, self.b0, self.b1, self.rb, self.margin,
            self.weight, start_idx], outputs=[residuals], device=self.device)


class ShoulderConeObjective(AutodiffRows):
    def __init__(self, chest, chest_bind_rot, arms, rom, device, weight=5.0, affects=None):
        super().__init__()
        self.aff_chest = wp.array(affects[chest], device=device)
        self.aff_upper = wp.array(np.array([affects[a[0]] for a in arms], np.uint8), device=device)
        self.aff_lower = wp.array(np.array([affects[a[1]] for a in arms], np.uint8), device=device)
        self.chest, self.chest_bind_rot = chest, chest_bind_rot
        self.upper = wp.array(np.array([a[0] for a in arms], np.int32), device=device)
        self.lower = wp.array(np.array([a[1] for a in arms], np.int32), device=device)
        self.sign = wp.array(np.array([a[2] for a in arms], np.float32), device=device)
        self.lim = [np.radians(rom[k]) for k in ("flexion_deg", "extension_deg", "abduction_deg", "adduction_deg")]
        self.n, self.weight, self.device = len(arms), weight, device

    def residual_dim(self):
        return self.n

    def supports_analytic(self):
        return True

    def init_buffers(self, model, jacobian_mode):
        pass

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(_cone_jacobian, dim=(body_q.shape[0], self.n, model.joint_dof_count), inputs=[
            body_q, joint_S_s, self.chest, self.chest_bind_rot, self.upper, self.lower, self.sign, self.aff_chest,
            self.aff_upper, self.aff_lower, *self.lim, self.weight, start_idx], outputs=[jacobian], device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(_cone_residuals, dim=(body_q.shape[0], self.n), inputs=[
            body_q, self.chest, self.chest_bind_rot, self.upper, self.lower, self.sign, *self.lim, self.weight,
            start_idx], outputs=[residuals], device=self.device)


def _tf(M):
    from scipy.spatial.transform import Rotation as R
    return wp.transform(wp.vec3(*M[:3, 3]), wp.quat(*R.from_matrix(M[:3, :3]).as_quat()))


class FullBodyIK:
    BODIES = ["Hips", "SpineLower", "SpineMiddle", "SpineUpper", "Chest"] + [
        f"{s}{b}" for s in SIDES for b in ("ArmUpper", "ArmLower", "HandWrist", "UpperLeg", "LowerLeg", "FootAnkle")]

    def _make_solver(self, optimizer):
        if optimizer == "lm":  # Gauss-Newton/LM: per-row autodiff Jacobians for the custom objectives (slower)
            return ik.IKSolver(model=self.model, n_problems=self.n, objectives=self.objectives,
                               optimizer=ik.IKOptimizer.LM, jacobian_mode=ik.IKJacobianType.MIXED)
        return ik.IKSolver(model=self.model, n_problems=self.n, objectives=self.objectives,
                           optimizer=ik.IKOptimizer.LBFGS, jacobian_mode=ik.IKJacobianType.MIXED, h0_scale=1.0,
                           line_search_alphas=[0.01, 0.1, 0.5, 0.75, 1.0], history_len=12)

    def __init__(self, rig, device=None, n_problems=1, params=None, optimizer="lm"):
        self.rig, self.device, self.n = rig, device or wp.get_device(), n_problems
        rom = (params or load_params())["body_rom"]
        B = {n: rig.bind[rig.index[n]] for n in self.BODIES}
        parent = {"SpineLower": "Hips", "SpineMiddle": "SpineLower", "SpineUpper": "SpineMiddle", "Chest": "SpineUpper"}
        for s in SIDES:
            parent.update({f"{s}ArmUpper": "Chest", f"{s}ArmLower": f"{s}ArmUpper", f"{s}HandWrist": f"{s}ArmLower",
                           f"{s}UpperLeg": "Hips", f"{s}LowerLeg": f"{s}UpperLeg", f"{s}FootAnkle": f"{s}LowerLeg"})
        cfg = newton.ModelBuilder.JointDofConfig
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        self.body = {}
        joints = []
        sp = np.radians([rom["spine"]["flexion_deg"], rom["spine"]["lateral_deg"], rom["spine"]["twist_deg"]]) / 4 + 0.05
        for n in self.BODIES:
            self.body[n] = b.add_link(xform=_tf(B[n]), label=n)
            if n == "Hips":
                joints.append(b.add_joint_free(self.body[n], label="root"))
                continue
            p = parent[n]
            rel = np.linalg.inv(B[p]) @ B[n]
            kw = dict(parent=self.body[p], child=self.body[n], parent_xform=_tf(rel), child_xform=wp.transform_identity(),
                      label=n)
            if n.endswith("ArmLower") or n.endswith("LowerLeg"):  # hinge about the true bend-plane normal
                knee = n.endswith("LowerLeg")
                end = n.replace("ArmLower", "HandWrist").replace("LowerLeg", "FootAnkle")
                up = _unit(B[n][:3, 3] - B[p][:3, 3])
                lo = _unit(B[end][:3, 3] - B[n][:3, 3])
                c = np.cross(up, lo)
                if np.linalg.norm(c) > 0.05:
                    world_axis = _unit(c)
                else:  # nearly straight at bind: lateral axis made perpendicular to both bones
                    y = np.array([0.0, 1.0, 0.0])
                    for v in (up, lo):
                        y = y - (y @ v) * v
                    world_axis = _unit(y)
                # sign: + = flexion (forearm swings forward, shin swings backward)
                from scipy.spatial.transform import Rotation as Rr
                moved = Rr.from_rotvec(world_axis * 0.2).as_matrix() @ lo
                if (moved[0] - lo[0]) * (-1.0 if knee else 1.0) < 0:
                    world_axis = -world_axis
                axis = B[n][:3, :3].T @ world_axis
                lim = np.radians(rom["knee"]["flexion_deg"] if knee else rom["elbow"]["flexion_deg"])
                joints.append(b.add_joint_revolute(axis=wp.vec3(*axis), limit_lower=-0.35, limit_upper=float(lim), **kw))
            elif n.startswith(("Spine", "Chest")):
                joints.append(b.add_joint_d6(angular_axes=[cfg(axis=a, limit_lower=-l, limit_upper=l) for a, l in
                                                           zip(newton.Axis, sp)], **kw))
            elif n.endswith("FootAnkle"):  # 3 DoF so any foot orientation is reachable; limited
                joints.append(b.add_joint_d6(angular_axes=[cfg(axis=a, limit_lower=-0.7, limit_upper=0.7)
                                                           for a in newton.Axis], **kw))
            else:  # shoulders, wrists, hips: 3 rotational DoF (the shoulder cone objective shapes the swing)
                lim = 2.6 if n.endswith(("ArmUpper", "UpperLeg")) else 1.2
                joints.append(b.add_joint_d6(angular_axes=[cfg(axis=a, limit_lower=-lim, limit_upper=lim)
                                                           for a in newton.Axis], **kw))
        b.add_articulation(joints, label="astronaut")
        self.model = b.finalize(device=self.device)
        self.index = self.body
        self.coords = self.model.joint_coord_count
        # objectives
        self.targets_pos, self.targets_rot = {}, {}
        objs = []
        for n, w_pos, w_rot in [(f"{s}HandWrist", 10.0, 1.0) for s in SIDES] + [(f"{s}FootAnkle", 10.0, 2.0) for s in SIDES]:
            tp = wp.array(np.tile(B[n][:3, 3], (n_problems, 1)).astype(np.float32), dtype=wp.vec3, device=self.device)
            tr = wp.array(np.tile(np.array(_tf(B[n])[3:7] if False else _quat4(B[n])), (n_problems, 1)).astype(np.float32),
                          dtype=wp.vec4, device=self.device)
            objs.append(ik.IKObjectivePosition(link_index=self.body[n], link_offset=wp.vec3(0.0, 0.0, 0.0),
                                               target_positions=tp, weight=w_pos))
            objs.append(ik.IKObjectiveRotation(link_index=self.body[n], link_offset_rotation=wp.quat_identity(),
                                               target_rotations=tr, weight=w_rot))
            self.targets_pos[n], self.targets_rot[n] = tp, tr
        self.prior_rot = {}
        for n in self.BODIES:  # orientation prior toward the reference pose
            if n.endswith(("HandWrist", "FootAnkle")):
                continue
            tr = wp.array(np.tile(_quat4(B[n]), (n_problems, 1)).astype(np.float32), dtype=wp.vec4, device=self.device)
            objs.append(ik.IKObjectiveRotation(link_index=self.body[n], link_offset_rotation=wp.quat_identity(),
                                               target_rotations=tr, weight=0.3))
            self.prior_rot[n] = tr
        self.root_pos = wp.array(np.tile(B["Hips"][:3, 3], (n_problems, 1)).astype(np.float32), dtype=wp.vec3, device=self.device)
        objs.append(ik.IKObjectivePosition(link_index=self.body["Hips"], link_offset=wp.vec3(0.0, 0.0, 0.0),
                                           target_positions=self.root_pos, weight=1.0))
        objs.append(ik.IKObjectiveJointLimit(joint_limit_lower=self.model.joint_limit_lower,
                                             joint_limit_upper=self.model.joint_limit_upper, weight=10.0))
        chest_rot = _quat_of(B["Chest"])
        objs.append(ShoulderConeObjective(self.body["Chest"], chest_rot,
                                          [(self.body[f"{s}ArmUpper"], self.body[f"{s}ArmLower"], 1.0 if s == "Left" else -1.0)
                                           for s in SIDES], rom["shoulder"], self.device, affects=self._affects()))
        self.collision = CapsuleCollisionObjective(self._collision_pairs(B), self.device, affects=self._affects())
        objs.append(self.collision)
        self.objectives = objs
        self.solver = self._make_solver(optimizer)
        self.q = wp.array(np.tile(self.model.joint_q.numpy(), (n_problems, 1)), dtype=float, device=self.device)
        self.q_out = wp.zeros_like(self.q)

    def _affects(self):
        """affects[body] = per-dof mask of the dofs that move the body (its joint and all ancestors)."""
        m = self.model
        jp, jc = m.joint_parent.numpy(), m.joint_child.numpy()
        qd0 = m.joint_qd_start.numpy()
        joint_of = {int(c): j for j, c in enumerate(jc)}
        out = {}
        for body in range(m.body_count):
            mask = np.zeros(m.joint_dof_count, np.uint8)
            b = body
            while b >= 0 and b in joint_of:
                j = joint_of[b]
                mask[qd0[j]:qd0[j + 1]] = 1
                b = int(jp[j])
            out[body] = mask
        return out

    def _collision_pairs(self, B):
        """Capsules in each body's local frame: (body, a, b, r)."""
        def loc(n, pts):
            M = np.linalg.inv(B[n])
            return [M[:3, :3] @ np.asarray(p) + M[:3, 3] for p in pts]
        cap = {}
        for i, (a, b_, r) in enumerate(TORSO_CAPSULES):
            cap[f"Torso{i}"] = ("Chest", *loc("Chest", [a, b_]), r)
        for s in SIDES:
            P = lambda n: B[n][:3, 3]  # noqa: E731
            cap[f"{s}UpperArm"] = (f"{s}ArmUpper", *loc(f"{s}ArmUpper", [P(f"{s}ArmUpper"), P(f"{s}ArmLower")]), 0.07)
            cap[f"{s}Forearm"] = (f"{s}ArmLower", *loc(f"{s}ArmLower", [P(f"{s}ArmLower"), P(f"{s}HandWrist")]), 0.06)
            hand_c = P(f"{s}HandWrist") + (B[f"{s}HandWrist"][:3, :3] @ np.array([0, 0, -0.07]))
            cap[f"{s}Hand"] = (f"{s}HandWrist", *loc(f"{s}HandWrist", [hand_c, hand_c]), 0.06)
            cap[f"{s}Thigh"] = (f"{s}UpperLeg", *loc(f"{s}UpperLeg", [P(f"{s}UpperLeg"), P(f"{s}LowerLeg")]), 0.085)
            cap[f"{s}Shin"] = (f"{s}LowerLeg", *loc(f"{s}LowerLeg", [P(f"{s}LowerLeg"), P(f"{s}FootAnkle")]), 0.07)
        pairs = []
        for s in SIDES:
            o = "Right" if s == "Left" else "Left"
            for limb in (f"{s}UpperArm", f"{s}Forearm", f"{s}Hand"):
                for torso in ("Torso0", "Torso1"):
                    if limb.endswith("UpperArm"):  # the shoulder joint sits at the torso edge: check the elbow end only
                        pass
                    pairs.append((limb, torso))
            pairs += [(f"{s}Hand", f"{s}Thigh"), (f"{s}Forearm", f"{s}Thigh"), (f"{s}Hand", f"{o}Hand")]
        pairs += [("LeftThigh", "RightThigh"), ("LeftShin", "RightShin")]
        out = []
        for a, b_ in pairs:
            A, Bc = cap[a], cap[b_]
            a0, a1 = A[1], A[2]
            if a.endswith("UpperArm"):  # start the upper-arm capsule 40% down from the shoulder (the joint touches the torso)
                a0 = a0 + 0.4 * (a1 - a0)
            out.append((self.body[A[0]], a0, a1, A[3], self.body[Bc[0]], Bc[1], Bc[2], Bc[3]))
        return out

    # ------------------------------------------------------------------ API
    def seed_from(self, W, problem=0):
        """Initial joint coordinates from rig world matrices W, so a solve starts at the reference and only corrects
        it. Joints are visited parent-first while tracking the frames actually achieved: the free root takes the
        world pose; a D6 decomposes its local rotation into intrinsic angles about its axes; before a hinge (elbow,
        knee) the parent ball joint is twisted about its own bone axis so the hinge axis is perpendicular to the
        child bone (that twist does not move the elbow/knee), then the hinge angle reproduces the child bone
        direction exactly and the next joint (wrist/ankle) absorbs the remaining rotation."""
        from scipy.spatial.transform import Rotation as R
        m = self.model
        q = self.q.numpy()
        qs, qd0 = m.joint_q_start.numpy(), m.joint_qd_start.numpy()
        jp, jc = m.joint_parent.numpy(), m.joint_child.numpy()
        Xp = m.joint_X_p.numpy()
        axis = m.joint_axis.numpy()
        name_of = {v: k for k, v in self.body.items()}
        ref = {b: W[self.rig.index[n]][:3, :3] for b, n in name_of.items()}
        hinge_child = {int(jp[j]): j for j in range(m.joint_count) if qd0[j + 1] - qd0[j] == 1}
        ach = {}  # achieved world rotation per body

        def rot(ax, ang):
            return R.from_rotvec(np.asarray(ax) * ang).as_matrix()

        for j in range(m.joint_count):
            c, p = int(jc[j]), int(jp[j])
            axes = axis[qd0[j]:qd0[j + 1]]
            if p < 0:  # free root
                q[problem, qs[j]:qs[j] + 3] = W[self.rig.index[name_of[c]], :3, 3]
                q[problem, qs[j] + 3:qs[j] + 7] = R.from_matrix(ref[c]).as_quat()
                ach[c] = ref[c]
                continue
            Pp = ach[p] @ R.from_quat(Xp[j][3:]).as_matrix()  # parent-side joint frame (world)
            target = ref[c]
            if c in hinge_child:  # this ball joint feeds a hinge: twist about the bone so the hinge can bend in-plane
                h = hinge_child[c]
                bone = -target[:, 2]  # -Z runs down the bone
                a_w = target @ R.from_quat(Xp[h][3:]).as_matrix() @ axis[qd0[h]]
                d = -ref[int(jc[h])][:, 2]
                a_par = (a_w @ bone) * bone
                a_perp = a_w - a_par
                A, B, C = a_perp @ d, np.cross(bone, a_perp) @ d, a_par @ d
                rr = np.hypot(A, B)
                if rr > 1e-9:
                    base = np.arctan2(B, A)
                    off = np.arccos(np.clip(-C / rr, -1.0, 1.0))
                    phi = min((base + off, base - off), key=lambda x: abs((x + np.pi) % (2 * np.pi) - np.pi))
                    target = rot(bone, phi) @ target
            Rl = Pp.T @ target
            if len(axes) == 1:  # hinge: angle that maps the parent-side bone onto the reference child bone
                a_w = Pp @ axes[0]
                d0 = Pp @ np.array([0.0, 0.0, -1.0])
                d1 = -ref[c][:, 2]
                p0, p1 = d0 - (d0 @ a_w) * a_w, d1 - (d1 @ a_w) * a_w
                ang = np.arctan2(np.cross(p0, p1) @ a_w, p0 @ p1)
                q[problem, qs[j]] = ang
                ach[c] = Pp @ rot(axes[0], ang)
            elif len(axes) == 3:  # D6 with local X, Y, Z: intrinsic X-Y-Z angles
                ang = R.from_matrix(Rl).as_euler("XYZ")
                q[problem, qs[j]:qs[j] + 3] = ang
                ach[c] = Pp @ R.from_euler("XYZ", ang).as_matrix()
            else:  # 2-axis (ankles): sequential projection
                a1, a2 = axes
                t1 = np.arctan2(*_axis_angle_parts(Rl, a1))
                rest = rot(a1, t1).T @ Rl
                t2 = np.arctan2(*_axis_angle_parts(rest, a2))
                q[problem, qs[j]:qs[j] + 2] = [t1, t2]
                ach[c] = Pp @ rot(a1, t1) @ rot(a2, t2)
        self.q.assign(q)

    def set_reference(self, W, problem=0, seed=False):
        """Reference pose (rig joint world matrices): orientation prior, root position, and default targets."""
        for n, arr in self.prior_rot.items():
            _set_row(arr, problem, _quat4(W[self.rig.index[n]]))
        _set_row(self.root_pos, problem, W[self.rig.index["Hips"], :3, 3])
        for n in self.targets_pos:
            self.set_target(n, W[self.rig.index[n]], problem)
        if seed:  # far from the previous solution (a new pose): start at the reference itself
            self.seed_from(W, problem)
            return
        # start the solve with the root at the reference hips pose (free-joint coords = world pose): LM then only
        # makes small corrections; joint angles stay warm-started from the previous solve
        q = self.q.numpy()
        H = W[self.rig.index["Hips"]]
        q[problem, :3], q[problem, 3:7] = H[:3, 3], _quat4(H)
        self.q.assign(q)

    def set_target(self, name, M, problem=0):
        _set_row(self.targets_pos[name], problem, M[:3, 3])
        _set_row(self.targets_rot[name], problem, _quat4(M))

    def _solve_once(self, iterations):
        self.solver.step(self.q, self.q_out, iterations=iterations)
        wp.copy(self.q, self.q_out)  # warm start the next solve

    def solve(self, iterations=20):
        """Run the LM solve; on CUDA the fixed-iteration solve is captured once as a graph (hundreds of small kernel
        launches per solve otherwise dominate the cost). Targets/priors are device arrays the graph reads."""
        if not wp.get_device(self.device).is_cuda:
            self._solve_once(iterations)
            return self.q_out
        graphs = self.__dict__.setdefault("_graphs", {})
        if iterations not in graphs:
            self._solve_once(iterations)  # warm-up (kernel compilation) outside the capture
            with wp.ScopedCapture(device=self.device) as cap:
                self._solve_once(iterations)
            graphs[iterations] = cap.graph
        wp.capture_launch(graphs[iterations])
        return self.q_out

    def body_q(self):
        if not hasattr(self, "_state"):
            self._state = self.model.state()
        wp.copy(self.model.joint_q, self.q_out.flatten()[: self.coords])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self._state)
        return self._state.body_q

    def rig_world(self, body_q_np):
        """Rig joint world matrices from solved body poses (joints without a body follow their parent rigidly)."""
        from scipy.spatial.transform import Rotation as R
        rig = self.rig
        W = rig.bind.copy()
        for i, n in enumerate(rig.names):
            if n in self.body:
                q = body_q_np[self.body[n]]
                W[i] = np.eye(4)
                W[i, :3, :3], W[i, :3, 3] = R.from_quat(q[3:]).as_matrix(), q[:3]
            elif rig.parent[i] >= 0:
                p = rig.parent[i]
                W[i] = W[p] @ np.linalg.inv(rig.bind[p]) @ rig.bind[i]
        return W


def _unit(v):
    return v / np.linalg.norm(v)


def _axis_angle_parts(Rm, a):
    """(sin, cos) of the rotation about unit axis a that best matches Rm (swing-twist twist component)."""
    from scipy.spatial.transform import Rotation as R
    x, y, z, w = R.from_matrix(Rm).as_quat()
    proj = np.dot([x, y, z], a)
    ang = 2.0 * np.arctan2(proj, w)
    return np.sin(ang), np.cos(ang)


def _quat_of(M):
    from scipy.spatial.transform import Rotation as R
    return wp.quat(*R.from_matrix(M[:3, :3]).as_quat())


def _quat4(M):
    from scipy.spatial.transform import Rotation as R
    return R.from_matrix(M[:3, :3]).as_quat().astype(np.float32)


def _set_row(arr, row, value):
    if arr.shape[0] == 1:  # single problem: no read-back
        arr.assign(np.asarray(value, np.float32).reshape(1, -1))
        return
    a = arr.numpy()
    a[row] = value
    arr.assign(a)
