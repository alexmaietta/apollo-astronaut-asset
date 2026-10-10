"""Keep the astronaut's textures in the repo (source/textures/, Git LFS), mirrored from NVIDIA's public S3 bucket.

NVIDIA's Omniverse Astronaut sample references its textures on S3. This script downloads them into
source/textures/<path below ".../Rendering/Astronaut/">, records each file's upstream URL in
data/texture_sources.json, and rewrites source/astronaut.usd to use the local relative paths ("./textures/...").

    python scripts/fetch_textures.py                 # download missing files, localize the source USD
    python scripts/fetch_textures.py --verify        # re-download nothing; check every manifest file exists
    python scripts/fetch_textures.py --from <usd>    # take the URLs from another copy (e.g. the database's)
"""
import concurrent.futures as cf
import json
import os
import shutil
import sys
import urllib.request

from pxr import Sdf, Usd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "source", "astronaut.usd")
TEXTURES = os.path.join(ROOT, "source", "textures")
MANIFEST = os.path.join(ROOT, "data", "texture_sources.json")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "astronaut", "textures")  # earlier download cache, reused
MARKER = "/Rendering/Astronaut/"


def rel_for(url):
    return url.split(MARKER, 1)[1] if MARKER in url else os.path.basename(url)


def asset_attrs(stage):
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            v = attr.Get()
            if isinstance(v, Sdf.AssetPath) and v.path:
                yield attr, v


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {}


def fetch(rel, url):
    dst = os.path.join(TEXTURES, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return rel, "present"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cached = os.path.join(CACHE, rel)
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        shutil.copyfile(cached, dst)
        return rel, "copied from cache"
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    return rel, "downloaded"


def anchor(stage):
    """Rewrite every resolvable asset path on a stage to its absolute resolved path (for flattened bakes that are
    written to a different folder, e.g. Blender inputs). Returns the number rewritten."""
    n = 0
    for attr, v in asset_attrs(stage):
        if v.resolvedPath and v.resolvedPath != v.path:
            attr.Set(Sdf.AssetPath(v.resolvedPath))
            n += 1
    return n


def main():
    manifest = load_manifest()
    src = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else SOURCE
    stage = Usd.Stage.Open(src)
    for _attr, v in asset_attrs(stage):
        if v.path.startswith("http"):
            manifest[rel_for(v.path)] = v.path
    if "--verify" in sys.argv:
        missing = [r for r in manifest if not os.path.exists(os.path.join(TEXTURES, r))]
        print(f"{len(manifest) - len(missing)}/{len(manifest)} textures present", *missing, sep="\n  ")
        sys.exit(1 if missing else 0)
    with cf.ThreadPoolExecutor(8) as ex:
        for rel, status in ex.map(lambda kv: fetch(*kv), manifest.items()):
            print(f"{status:18s} {rel}")
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(dict(sorted(manifest.items())), f, indent=1)
    # localize the repo's source USD
    layer = Sdf.Layer.FindOrOpen(SOURCE)
    stage = Usd.Stage.Open(layer)
    n = 0
    for attr, v in asset_attrs(stage):
        if v.path.startswith("http"):
            attr.Set(Sdf.AssetPath("./textures/" + rel_for(v.path)))
            n += 1
    layer.Save()
    print(f"{len(manifest)} textures in {TEXTURES}; {n} asset paths in source/astronaut.usd now local")


if __name__ == "__main__":
    main()
