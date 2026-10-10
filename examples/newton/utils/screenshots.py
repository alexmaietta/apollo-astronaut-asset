"""Headless Newton GL-viewer screenshots of the astronaut examples (for the README).

    python examples/newton/utils/screenshots.py [--out renders/newton] [--only ik grasp]

Writes <out>/newton_<example>.png. scripts/gallery.py converts them for docs/images.
"""
import argparse
import math
import os
import sys

import numpy as np
import warp as wp
from PIL import Image

import newton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kernels import REPO  # noqa: E402

# example: (example args, frames to step first, camera position, look-at target)
SHOTS = {
    "ik": (["--pose", "salute"], 30, (2.6, -2.1, 1.45), (0.0, 0.0, 1.0)),
    "poses": (["--pose", "geology"], 10, (2.7, -1.9, 1.2), (0.1, 0.0, 0.8)),
    "walk": ([], 240, None, None),  # the example's follow camera
    "grasp": (["--thumb", "0", "--lift", "0.08", "--body-colliders", "none", "--hand-offset", "0.15", "-0.15", "0.02"], 240,
              (0.95, -1.1, 1.05), (0.2, -0.49, 0.85)),
}


def look(viewer, pos, target):
    d = np.subtract(target, pos)
    d /= np.linalg.norm(d)
    viewer.set_camera(wp.vec3(*pos), math.degrees(math.asin(d[2])), math.degrees(math.atan2(d[1], d[0])))


def shoot(name, out, width, height):
    import importlib
    mod = importlib.import_module(f"example_astronaut_{name}")
    ex_args, frames, pos, tgt = SHOTS[name]
    args = mod.Example.create_parser().parse_args(["--viewer", "gl", "--headless", *ex_args])
    viewer = newton.viewer.ViewerGL(headless=True, width=width, height=height)
    ex = mod.Example(viewer, args)
    for _ in range(frames):
        ex.step()
    if pos is not None:
        look(viewer, pos, tgt)
    ex.render()
    path = os.path.join(out, f"newton_{name}.png")
    Image.fromarray(viewer.get_frame().numpy()).save(path)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "renders", "newton"))
    ap.add_argument("--only", nargs="*", default=list(SHOTS))
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if len(a.only) == 1:
        shoot(a.only[0], a.out, a.width, a.height)
        return
    import subprocess
    for name in a.only:  # one process (and GL context) per example
        subprocess.run([sys.executable, os.path.abspath(__file__), "--out", a.out, "--only", name,
                        "--width", str(a.width), "--height", str(a.height)], check=True)


if __name__ == "__main__":
    main()
