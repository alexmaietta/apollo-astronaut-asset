"""Record the physics grasp (example_astronaut_grasp.py, headless) for the Cycles close-ups and GIF.

Runs the scripted pick-and-place and saves, every `--every` frames, the GPU-skinned astronaut vertices exactly as
the Newton viewer draws them (hands posed from the simulated finger bodies, arm IK, correctives), the sample
carrier pose and the carrier / pedestal shapes, plus the per-finger haptic forces.

    python examples/newton/utils/record_grasp.py [--out build/grasp] [--every 5] [-- <example args>]

The README shots use a hook grip (the four fingers carry the handle, the thumb stays open, as with a stiff pressurized
glove), the hand starting out to the side and forward, an 8 cm lift and no suit body colliders (default example
args below); the recording covers closing, lifting
and holding (0-5 s).

Writes <out>/grasp.npz (frame times, carrier poses, shapes, haptics, mesh layout) and <out>/points_<frame>.npy
(float32, all rig vertices in rig mesh order). Render with scripts/render_hero.py --grasp.
"""
import argparse
import os
import sys

import numpy as np
import warp as wp

import newton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import example_astronaut_grasp as grasp  # noqa: E402
from kernels import REPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "build", "grasp"))
    ap.add_argument("--every", type=int, default=5, help="record every N-th 60 Hz frame (5 = 12 fps)")
    ap.add_argument("--seconds", type=float, default=5.0, help="length of the recording (close, lift, hold)")
    ap.add_argument("example_args", nargs="*", default=["--thumb", "0", "--lift", "0.08", "--body-colliders", "none",
                                                        "--hand-offset", "0.15", "-0.15", "0.02"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    args = grasp.Example.create_parser().parse_args(a.example_args)
    viewer = newton.viewer.ViewerNull(num_frames=1 << 30)
    ex = grasp.Example(viewer, args)
    m = ex.model
    body, stype = m.shape_body.numpy(), m.shape_type.numpy()
    labels = list(getattr(m, "shape_label", [""] * len(body)))
    keep = [i for i in range(len(body)) if body[i] == ex.carrier or str(labels[i]).endswith("pedestal")]
    shapes = dict(shape_body=body[keep], shape_type=stype[keep], shape_xform=m.shape_transform.numpy()[keep],
                  shape_scale=m.shape_scale.numpy()[keep], carrier=ex.carrier,
                  geo_box=int(newton.GeoType.BOX), geo_capsule=int(newton.GeoType.CAPSULE))
    times, carrier_q, haptics, frames = [], [], [], []
    n = int(round(a.seconds * ex.fps))
    for f in range(n + 1):
        if f % a.every == 0:
            ex.render()  # skins on the GPU from the current state
            wp.synchronize()
            np.save(os.path.join(a.out, f"points_{f:04d}.npy"), ex.astro.points.numpy())
            frames.append(f)
            times.append(f * ex.frame_dt)
            carrier_q.append(ex.state_0.body_q.numpy()[ex.carrier])
            haptics.append(ex.haptics.values["right"].copy())
        ex.step()
    rig = ex.astro.rig
    np.savez(os.path.join(a.out, "grasp.npz"), frames=np.array(frames), times=np.array(times),
             carrier_q=np.array(carrier_q), haptics=np.array(haptics), handle_c=ex.handle_c,
             mesh_names=np.array([mm["name"] for mm in rig.meshes]), mesh_counts=np.array([len(mm["P"]) for mm in rig.meshes]),
             **shapes)
    print(f"recorded {len(frames)} frames to {a.out}")


if __name__ == "__main__":
    main()
