"""Minimal reader for the CMU Graphics Lab Motion Capture Database (http://mocap.cs.cmu.edu), ASF/AMC format.

The CMU data is free to use, including in commercial products, but may not be resold as data; this repo only ships
renders of the retargeted motion. `fetch()` downloads a subject's skeleton and trial into build/mocap/.

    skel = load_asf("07.asf"); frames = load_amc("07_01.amc")
    P = joint_positions(skel, frames)   # {bone: (n_frames, 3)} end points, metres, Z-up (CMU is Y-up)
"""
import os
import urllib.request

import numpy as np
from scipy.spatial.transform import Rotation as R

URL = "http://mocap.cs.cmu.edu/subjects/{subject:02d}/{name}"
FPS = 120.0


def fetch(subject, trial, out):
    os.makedirs(out, exist_ok=True)
    paths = []
    for name in (f"{subject:02d}.asf", f"{subject:02d}_{trial:02d}.amc"):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            urllib.request.urlretrieve(URL.format(subject=subject, name=name), p)
        paths.append(p)
    return paths


def load_asf(path):
    """Bones: {name: dict(direction, length, C (axis rotation), dof, parent)}, plus root axis and length scale."""
    lines = [ln.strip() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    skel, scale, i, bone = {"root": dict(direction=np.zeros(3), length=0.0, dof=["tx", "ty", "tz", "rx", "ry", "rz"],
                                        C=np.eye(3), parent=None)}, 1.0, 0, None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(":units"):
            i += 1
            while not lines[i].startswith(":"):
                k, v = lines[i].split()[:2]
                if k == "length":
                    scale = 0.0254 / float(v)  # CMU: inches / length unit -> metres
                i += 1
            continue
        if ln.startswith(":root"):
            i += 1
            while not lines[i].startswith(":"):
                t = lines[i].split()
                if t[0] == "orientation":
                    skel["root"]["C"] = R.from_euler("xyz", [float(x) for x in t[1:4]], degrees=True).as_matrix()
                i += 1
            continue
        if ln == "begin":
            bone = dict(dof=[], C=np.eye(3), parent=None)
        elif ln == "end" and bone is not None:
            skel[bone["name"]] = bone
            bone = None
        elif bone is not None:
            t = ln.split()
            if t[0] == "name":
                bone["name"] = t[1]
            elif t[0] == "direction":
                bone["direction"] = np.array([float(x) for x in t[1:4]])
            elif t[0] == "length":
                bone["length"] = float(t[1])
            elif t[0] == "axis":
                bone["C"] = R.from_euler("xyz", [float(x) for x in t[1:4]], degrees=True).as_matrix()
            elif t[0] == "dof":
                bone["dof"] = [d.lower() for d in t[1:]]
        elif ln == ":hierarchy":
            i += 2  # skip "begin"
            while lines[i] != "end":
                t = lines[i].split()
                for child in t[1:]:
                    skel[child]["parent"] = t[0]
                i += 1
        i += 1
    skel["_scale"] = scale
    return skel


def load_amc(path):
    """List of frames, each {bone: [values in dof order]}."""
    frames, cur = [], None
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln[0] in "#:":
            continue
        if ln.isdigit():
            cur = {}
            frames.append(cur)
        else:
            t = ln.split()
            cur[t[0]] = [float(x) for x in t[1:]]
    return frames


def joint_positions(skel, frames):
    """Bone end positions {bone: (n, 3)} in metres, Z-up (CMU Y-up rotated +90 deg about X), and root rotations."""
    order, seen = [], {"root"}
    while len(order) < len([k for k in skel if not k.startswith("_")]) - 1:
        for k, b in skel.items():
            if k.startswith("_") or k in seen or b["parent"] not in seen:
                continue
            order.append(k)
            seen.add(k)
    yup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # (x, y, z)_Y-up -> (x, -z, y)_Z-up
    s = skel["_scale"]
    out = {k: np.zeros((len(frames), 3)) for k in ["root", *order]}
    for f, fr in enumerate(frames):
        rv = dict(zip(skel["root"]["dof"], fr["root"]))
        M = {"root": skel["root"]["C"] @ R.from_euler("xyz", [rv["rx"], rv["ry"], rv["rz"]], degrees=True).as_matrix()
             @ skel["root"]["C"].T}
        pos = {"root": np.array([rv["tx"], rv["ty"], rv["tz"]]) * s}
        for k in order:
            b = skel[k]
            vals = dict(zip(b["dof"], fr.get(k, [])))
            rot = R.from_euler("xyz", [vals.get("rx", 0.0), vals.get("ry", 0.0), vals.get("rz", 0.0)], degrees=True)
            M[k] = M[b["parent"]] @ b["C"] @ rot.as_matrix() @ b["C"].T
            pos[k] = pos[b["parent"]] + b["length"] * s * (M[k] @ b["direction"])
        for k, p in pos.items():
            out[k][f] = yup @ p
    return out
