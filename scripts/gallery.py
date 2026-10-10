"""README images: convert the renders to web-sized files in docs/images/.

    .venv/bin/python scripts/gallery.py [--gif-width 560] [--gif-fps 12]

renders/hero/<shot>.png        -> docs/images/<shot>.jpg; hero_full + hero_dcm + hero_rcu -> hero.jpg (the banner);
                                  thumbnail.jpg from hero_full
renders/grasp/still_<t>.png    -> docs/images/grasp_<t>.jpg
renders/grasp/gif/*.png        -> docs/images/grasp.gif (looping, the last frame held)
renders/walk/*.png             -> docs/images/walk.gif (two gait cycles, seamless loop)
renders/newton/newton_*.png    -> docs/images/newton_*.jpg
docs/images is only replaced once every input has been read.
"""
import argparse
import glob
import os
import shutil
import tempfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "renders")


def jpg(src, dst, max_side, quality=88):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(dst, quality=quality, optimize=True, progressive=True)


def gif(frames, dst, width, fps, hold_s=1.0):
    ims = []
    for f in frames:
        with Image.open(f) as im:
            im = im.convert("RGB")
            ims.append(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS))
    # one shared adaptive palette (from a strip of all frames) keeps the loop free of palette flicker
    strip = Image.new("RGB", (width, ims[0].height * len(ims[::6])))
    for i, im in enumerate(ims[::6]):
        strip.paste(im, (0, i * im.height))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for im in ims]
    dur = [round(1000 / fps)] * len(q)
    dur[-1] += round(1000 * hold_s)
    q[0].save(dst, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True, disposal=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "images"))
    ap.add_argument("--gif-width", type=int, default=560)
    ap.add_argument("--gif-fps", type=float, default=12.0)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    for f in sorted(glob.glob(os.path.join(R, "hero", "*.png"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if not name.startswith("hero_"):
            jpg(f, os.path.join(tmp, f"{name}.jpg"), 1600)
    panels = [os.path.join(R, "hero", f"hero_{p}.png") for p in ("full", "dcm", "rcu")]
    if all(os.path.exists(p) for p in panels):
        ims = [Image.open(p).convert("RGB") for p in panels]
        banner = Image.new("RGB", (sum(im.width for im in ims), max(im.height for im in ims)))
        x = 0
        for im in ims:
            banner.paste(im, (x, 0))
            x += im.width
        banner.save(os.path.join(tmp, "hero.jpg"), quality=90, optimize=True, progressive=True)
        w, h = ims[0].size  # square crop of the full figure for the asset thumbnail
        ims[0].crop(((w - h) // 2, 0, (w + h) // 2, h)).resize((512, 512), Image.LANCZOS) \
            .save(os.path.join(tmp, "thumbnail.jpg"), quality=90)
    for f in sorted(glob.glob(os.path.join(R, "grasp", "still_*.png"))):
        t = os.path.basename(f)[len("still_"):-len(".png")]
        jpg(f, os.path.join(tmp, f"grasp_{t}.jpg"), 900)
    frames = sorted(glob.glob(os.path.join(R, "grasp", "gif", "*.png")))
    if frames:
        gif(frames, os.path.join(tmp, "grasp.gif"), a.gif_width, a.gif_fps)
    walk = sorted(glob.glob(os.path.join(R, "walk", "*.png")))
    if walk:
        gif(walk, os.path.join(tmp, "walk.gif"), 640, 20.0, hold_s=0.0)
    for f in sorted(glob.glob(os.path.join(R, "newton", "newton_*.png"))):
        jpg(f, os.path.join(tmp, os.path.basename(f)[:-4] + ".jpg"), 1600)
    os.makedirs(a.out, exist_ok=True)
    for f in sorted(os.listdir(tmp)):
        shutil.move(os.path.join(tmp, f), os.path.join(a.out, f))
        print(f"{f:28s} {os.path.getsize(os.path.join(a.out, f)) / 1e3:7.0f} kB")


if __name__ == "__main__":
    main()
