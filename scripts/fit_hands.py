"""Fit finger joint positions to the glove geometry (source frame, cm).

Starting from the template chains in skeleton.py, each finger's centreline is re-centred on the glove surface
(cross-section centroids of the points that belong to that finger), the tip is set where the finger's points
end, the knuckle (MCP) is kept from the template projected onto the fitted centreline, and PIP/DIP are placed
at anatomical phalanx proportions along the curve. Prints the fitted table for skeleton.py and writes
build/hand_fit.json. Both hands are fitted; the result is compared against a mirror of the left hand.
"""
import json
import os
import sys

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
os.environ["ASTRONAUT_HAND_TEMPLATE"] = "1"  # always fit from the template, never from a previous fit
from skeleton import _ARM_L, _FINGERS_L  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# fraction of MCP->tip arc length at PIP and DIP (proximal : intermediate : distal ~ 0.49 : 0.29 : 0.22)
PHALANX = {"Index": (0.49, 0.78), "Middle": (0.49, 0.78), "Ring": (0.50, 0.78), "Little": (0.50, 0.77)}
THUMB_SPLIT = 0.55  # thumb proximal->tip arc fraction at the IP joint


def glove_points(side):
    s = Usd.Stage.Open(os.path.join(ROOT, "source", "astronaut.usd"))
    pts = []
    for name in ("Astronaut_posed57", "Astronaut_posed54"):  # glove + blue fingertip caps
        prim = next(p for p in s.Traverse() if p.GetName() == name)
        P = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        pts.append(P[P[:, 0] * side > 0])
    return np.concatenate(pts)


def resample(poly, step=0.25):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, poly[:, k]) for k in range(3)], 1)


def at_arclength(poly, frac):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = frac * s[-1]
    return np.array([np.interp(t, s, poly[:, k]) for k in range(3)])


def circle_centre(Q, p, axis):
    """Least-squares (Kasa) circle fit of points Q projected on the plane through p perpendicular to axis.
    Robust to partial rings, unlike a centroid. Returns (centre, radius, rms residual) or None."""
    axis = axis / np.linalg.norm(axis)
    u = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    x, y = (Q - p) @ u, (Q - p) @ v
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9))
    res = np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))
    return p + cx * u + cy * v, r, res


def fit_side(side):
    G = glove_points(side)
    mirror = np.array([side, 1.0, 1.0])
    chains = {f: np.array(p) * mirror for f, p in _FINGERS_L.items()}
    chains["Thumb"] = np.array([_ARM_L[f"LeftHandThumb{k}"][1] for k in ("Metacarpal", "Proximal", "Distal", "Tip")]) * mirror
    report = {}
    for _ in range(4):
        trees = {f: cKDTree(resample(c[1:], 0.2)) for f, c in chains.items()}
        dist = np.stack([trees[f].query(G)[0] for f in chains], 1)
        owner = np.array(list(chains))[np.argmin(dist, 1)]
        for f, c in chains.items():
            Q = G[owner == f]
            rep = []
            # knuckles distal of the MCP: circle-fit centre across the local bone direction
            for i in range(2, len(c) - 1):
                axis = c[i + 1] - c[i - 1]
                ax = axis / np.linalg.norm(axis)
                d = Q - c[i]
                m = (np.abs(d @ ax) < 0.3) & (np.linalg.norm(d - (d @ ax)[:, None] * ax, axis=1) < 1.6)
                if m.sum() >= 8:
                    centre, r, res = circle_centre(Q[m], c[i], axis)
                    if 0.5 < r < 1.7 and np.linalg.norm(centre - c[i]) < 1.4:
                        c[i] = centre
                    rep.append((round(r, 2), round(res, 2)))
            # tip: last own point along the distal direction, pulled back by the finger radius and centred
            ax = c[-2] - c[-3]
            ax /= np.linalg.norm(ax)
            near = Q[np.linalg.norm(Q - c[-2], axis=1) < 3.0]
            if len(near):
                end = ((near - c[-2]) @ ax).max()
                t = c[-2] + ax * max(end - 0.5, 0.3)
                d = near - t
                m = (np.abs(d @ ax) < 0.35) & (np.linalg.norm(d - (d @ ax)[:, None] * ax, axis=1) < 1.4)
                if m.sum() >= 6:
                    centre, r, res = circle_centre(near[m], t, ax)
                    if r < 1.3 and np.linalg.norm(centre - t) < 0.8:
                        t = centre
                c[-1] = t
            report[f] = rep
    out = {}
    for f, c in chains.items():
        names = ("Metacarpal", "Proximal", "Distal", "Tip") if f == "Thumb" else \
            ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")
        out[f] = dict(zip(names, c))
        out[f]["_fit"] = report[f]
    return out


if __name__ == "__main__":
    L, R = fit_side(1.0), fit_side(-1.0)
    worst = max(np.linalg.norm(L[f][k] * np.array([-1, 1, 1]) - R[f][k]) for f in L for k in L[f] if k[0] != "_")
    print(f"left vs mirrored right: max joint difference {worst:.2f} cm")
    for f in L:
        print(f, "circle radius/rms cm (L, R):", L[f].pop("_fit"), R[f].pop("_fit"))
    with open(os.path.join(ROOT, "build", "hand_fit.json"), "w") as fh:
        json.dump({f: {k: [round(float(x), 2) for x in v] for k, v in L[f].items()} for f in L}, fh, indent=1)
