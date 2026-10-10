"""Articulated physics hands for the astronaut, derived from the rig (bind pose, metres, rig frame).

Per hand: 21 bodies, 20 revolute DoF (Allegro/Shadow-style primitives, see newton-assets wonik_allegro/usd_structured):
  Wrist (palm, box collider; metacarpals II-V are rigid with it)
  <F>ProximalBase (MCP spread helper) -> <F>Proximal -> <F>Intermediate -> <F>Distal   for Index/Middle/Ring/Little
  ThumbMetacarpalBase (CMC spread helper) -> ThumbMetacarpal -> ThumbProximal -> ThumbDistal
Finger segments are capsules whose radii are measured from the glove vertices skinned to that segment.

Joint conventions (shared with openxr_adapter.py): flexion joints rotate about -X of the child's OpenXR frame
(positive = curl toward the palm); spread joints rotate about the OpenXR +Y (back-of-hand) axis. Physics joint
coordinates are zero at the bind pose; `bind_deg` is each DoF's bind angle measured from an open hand, so
anatomical angle = q + bind_deg. Suit/glove parameters come from data/suit_params.json.
"""
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "data", "suit_params.json")
FINGERS = ("Index", "Middle", "Ring", "Little")
SIDES = ("Left", "Right")


def load_params(path=PARAMS, **overrides):
    with open(path) as f:
        p = json.load(f)
    p.update(overrides)
    return p


def _unit(v):
    return v / np.linalg.norm(v)


def signed_angle(d0, d1, axis):
    """Angle (rad) rotating d0 to d1 about axis, after projecting both onto the plane normal to axis."""
    a = _unit(axis)
    p0 = _unit(d0 - (d0 @ a) * a)
    p1 = _unit(d1 - (d1 @ a) * a)
    return float(np.arctan2(np.cross(p0, p1) @ a, p0 @ p1))


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1), t


def segment_radii(G, segments, default=0.011, distractors=()):
    """Glove radius per segment: median of least-squares circle fits (fit_hands.circle_centre) to the owned glove
    points in thin slabs at 30/50/70% along the segment. Circle fits are robust to partial rings and off-centre axes."""
    from fit_hands import circle_centre
    D = np.stack([_seg_dist(G, a, b)[0] for a, b in list(segments) + list(distractors)], 1)
    owner = np.argmin(D, 1)  # distractor segments (palm rays) claim palm/webbing points
    radii = []
    for k, (a, b) in enumerate(segments):
        ax = _unit(b - a)
        Q = G[owner == k]
        rs = []
        for t in (0.3, 0.5, 0.7):
            c = a + (b - a) * t
            d = Q - c
            along = d @ ax
            radial = np.linalg.norm(d - along[:, None] * ax, axis=1)
            m = (np.abs(along) < 0.003) & (radial < 0.025)
            if m.sum() >= 8:
                _, r, _ = circle_centre(Q[m], c, ax)
                if 0.005 < r < 0.025:
                    rs.append(r)
        radii.append(float(np.median(rs)) if rs else default)
    return radii


class HandModel:
    def __init__(self, rig, side, params):
        self.side, self.params = side, params
        idx = rig.index
        pos = {n: rig.bind[i, :3, 3] for n, i in idx.items()}
        rot = {n: rig.bind[i, :3, :3] for n, i in idx.items()}
        h = lambda k: f"{side}Hand{k}"  # noqa: E731
        # glove vertices (incl. fingertip caps) and their dominant joints, for collider sizing
        G, J = [], []
        for m in rig.meshes:
            if m["name"] in ("gloves", "glove_fingertips"):
                sel = (m["P"][:, 1] > 0) if side == "Left" else (m["P"][:, 1] < 0)
                G.append(m["P"][sel])
                J.append(m["idx"][sel][np.arange(sel.sum()), np.argmax(m["w"][sel], 1)])
        G, J = np.concatenate(G), np.concatenate(J)
        names = np.array(rig.names)
        scale = params.get("collider_radius_scale", 1.0)

        self.bodies, self.joints = [], []

        def body(name, parent, joint_name, colliders, helper=False):
            self.bodies.append(dict(name=name, parent=parent, R=rot[joint_name], p=pos[joint_name],
                                    colliders=colliders, helper=helper, skin_joint=joint_name))

        def joint(name, parent, child, axis, anchor_joint, kind, bind_rad):
            lim = np.array(self.params["joints"][kind]["limits_deg"], float) * self.params.get("rom_scale", 1.0)
            bind_deg = np.degrees(bind_rad)
            lo, hi = lim - bind_deg
            lo, hi = min(lo, 0.0), max(hi, 0.0)  # the bind pose must stay inside the limits
            self.joints.append(dict(name=name, parent=parent, child=child, axis=_unit(axis), anchor=pos[anchor_joint],
                                    kind=kind, bind_deg=float(bind_deg), lower_deg=float(lo), upper_deg=float(hi)))

        tch = [h("Thumb" + k) for k in ("Metacarpal", "Proximal", "Distal", "Tip")]
        chains = {f: [h(f + k) for k in ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")] for f in FINGERS}
        segs = {}
        for f, ch in chains.items():
            for i in range(1, 4):
                segs[ch[i]] = (pos[ch[i]], pos[ch[i + 1]])
        for i in range(3):
            segs[tch[i]] = (pos[tch[i]], pos[tch[i + 1]])
        palm_rays = [(pos[h("Wrist")], pos[chains[f][1]] - 0.004 * _unit(pos[chains[f][1]] - pos[h("Wrist")]))
                     for f in FINGERS]
        radius = dict(zip(segs, segment_radii(G, list(segs.values()), distractors=palm_rays)))
        # cap finger segments at 1.25x the median finger radius (knuckle padding at the hand edges inflates fits)
        ks = [k for k in radius if "Thumb" not in k]
        cap = 1.25 * np.median([radius[k] for k in ks])
        for k in ks:
            radius[k] = min(radius[k], cap)
        radius = {k: r * scale for k, r in radius.items()}

        def capsule(name):
            a, b = segs[name]
            r = radius[name]
            if name.endswith("Distal"):  # stop the cap so the fingertip surface sits just past the Tip joint
                b = b - _unit(b - a) * max(r - 0.004, 0.0)
            return dict(type="capsule", a=a, b=b, r=r)

        # palm: box in the wrist frame spanning wrist -> knuckles and the knuckle row, thickness from the glove
        Rw, pw = rot[h("Wrist")], pos[h("Wrist")]
        knuck = np.array([(pos[chains[f][1]] - pw) @ Rw for f in FINGERS])
        loc = (G - pw) @ Rw
        # wrist joint at z = 0, knuckles at negative z (fingers along -Z); start 2.5 cm past the wrist so the
        # glove's round wrist cuff does not inflate the palm thickness
        zlo, zhi = knuck[:, 2].min(), -0.025
        xlo, xhi = knuck[:, 0].min() - 0.012, knuck[:, 0].max() + 0.012
        thumb_j = [idx[n] for n in tch]
        m = (loc[:, 2] > zlo) & (loc[:, 2] < zhi) & (loc[:, 0] > xlo) & (loc[:, 0] < xhi) & ~np.isin(J, thumb_j)
        ylo, yhi = np.percentile(loc[m, 1], 15), np.percentile(loc[m, 1], 85)  # skin of the padded glove, not its folds
        lo, hi = np.array([xlo, ylo, zlo]), np.array([xhi, yhi, zhi])
        body(h("Wrist"), None, h("Wrist"),
             [dict(type="box", center=pw + Rw @ ((lo + hi) / 2), R=Rw, half=(hi - lo) / 2)])

        for f, ch in chains.items():
            d = {k: pos[ch[i + 1]] - pos[ch[i]] for i, k in enumerate(("meta", "prox", "inter", "dist"))}
            base = h(f + "ProximalBase")
            self.bodies.append(dict(name=base, parent=h("Wrist"), R=rot[ch[1]], p=pos[ch[1]], colliders=[],
                                    helper=True, skin_joint=None))
            body(ch[1], base, ch[1], [capsule(ch[1])])
            body(ch[2], ch[1], ch[2], [capsule(ch[2])])
            body(ch[3], ch[2], ch[3], [capsule(ch[3])])
            Xp, Yp = rot[ch[1]][:, 0], rot[ch[1]][:, 1]
            # spread: the bind pose is neutral (metacarpal directions are not measured, so no open-hand reference)
            joint(f"{ch[1]}_abd", h("Wrist"), base, Yp, ch[1], "mcp_abd", 0.0)
            joint(f"{ch[1]}_flex", base, ch[1], -Xp, ch[1], "mcp_flex", signed_angle(d["meta"], d["prox"], -Xp))
            Xi = rot[ch[2]][:, 0]
            joint(f"{ch[2]}_flex", ch[1], ch[2], -Xi, ch[2], "pip_flex", signed_angle(d["prox"], d["inter"], -Xi))
            Xd = rot[ch[3]][:, 0]
            joint(f"{ch[3]}_flex", ch[2], ch[3], -Xd, ch[3], "dip_flex", signed_angle(d["inter"], d["dist"], -Xd))

        tbase = h("ThumbMetacarpalBase")
        self.bodies.append(dict(name=tbase, parent=h("Wrist"), R=rot[tch[0]], p=pos[tch[0]], colliders=[],
                                helper=True, skin_joint=None))
        body(tch[0], tbase, tch[0], [capsule(tch[0])])
        body(tch[1], tch[0], tch[1], [capsule(tch[1])])
        body(tch[2], tch[1], tch[2], [capsule(tch[2])])
        Xt, Yt = rot[tch[0]][:, 0], rot[tch[0]][:, 1]
        joint(f"{tch[0]}_abd", h("Wrist"), tbase, Yt, tch[0], "thumb_cmc_abd", 0.0)
        joint(f"{tch[0]}_flex", tbase, tch[0], -Xt, tch[0], "thumb_cmc_flex", 0.0)
        dt = [pos[tch[i + 1]] - pos[tch[i]] for i in range(3)]
        X1, X2 = rot[tch[1]][:, 0], rot[tch[2]][:, 0]
        joint(f"{tch[1]}_flex", tch[0], tch[1], -X1, tch[1], "thumb_mcp_flex", signed_angle(dt[0], dt[1], -X1))
        joint(f"{tch[2]}_flex", tch[1], tch[2], -X2, tch[2], "thumb_ip_flex", signed_angle(dt[1], dt[2], -X2))

        self._assign_mass()
        self.body_index = {b["name"]: i for i, b in enumerate(self.bodies)}

    def _assign_mass(self):
        total = self.params["mass"]["hand_and_glove_kg"]
        helper_mass = 0.005
        vols = []
        for b in self.bodies:
            v = 0.0
            for c in b["colliders"]:
                if c["type"] == "capsule":
                    L = np.linalg.norm(c["b"] - c["a"])
                    v += np.pi * c["r"] ** 2 * L + 4 / 3 * np.pi * c["r"] ** 3
                else:
                    v += 8 * np.prod(c["half"])
            vols.append(v)
        vols = np.array(vols)
        n_help = sum(b["helper"] for b in self.bodies)
        rest = total - helper_mass * n_help
        for b, v in zip(self.bodies, vols):
            b["mass"] = helper_mass if b["helper"] else float(rest * v / vols.sum())

    def summary(self):
        lines = [f"{self.side} hand: {len(self.bodies)} bodies, {len(self.joints)} DoF, "
                 f"{sum(b['mass'] for b in self.bodies):.3f} kg"]
        for b in self.bodies:
            for c in b["colliders"]:
                if c["type"] == "capsule":
                    lines.append(f"  {b['name']:28s} capsule r={c['r'] * 1000:4.1f} mm L={np.linalg.norm(c['b'] - c['a']) * 1000:5.1f} mm  m={b['mass'] * 1000:5.1f} g")
                else:
                    lines.append(f"  {b['name']:28s} box half={np.round(c['half'] * 1000, 1)} mm  m={b['mass'] * 1000:5.1f} g")
        for j in self.joints:
            lines.append(f"  {j['name']:34s} bind {j['bind_deg']:6.1f} deg  limits [{j['lower_deg']:6.1f}, {j['upper_deg']:6.1f}]")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from rig import Rig
    rig = Rig(os.path.join(ROOT, "build", "astronaut_rig.usd"))
    for s in SIDES:
        print(HandModel(rig, s, load_params()).summary())
