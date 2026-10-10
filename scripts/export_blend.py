"""Rigged Blender file from the packaged asset: UsdSkel -> armature, shoulder correctives -> shape keys,
UsdPreviewSurface looks -> Principled BSDF with the textures packed.

    blender -b --factory-startup -P scripts/export_blend.py -- [--usd build/asset/astronaut/astronaut.usda]
        [--out build/blender/astronaut.blend]

The physics variants are switched off first (a flattened, render-only copy of the asset), so only the skinned
suit is imported. The corrective shape keys import at weight 0; their RBF driver lives in the USD skeleton's
customData["astronaut:correctives"] and is not recreated as Blender drivers.
"""
import os
import sys
import tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    usd = os.path.join(ROOT, "build", "asset", "astronaut", "astronaut.usda")
    out = os.path.join(ROOT, "build", "blender", "astronaut.blend")
    for i, a in enumerate(argv):
        if a == "--usd":
            usd = argv[i + 1]
        elif a == "--out":
            out = argv[i + 1]
    from pxr import Usd
    stage = Usd.Stage.Open(usd)
    root = stage.GetDefaultPrim()
    root.GetVariantSet("Physics").SetVariantSelection("none")
    root.GetVariantSet("BodyColliders").SetVariantSelection("none")
    tmp = os.path.join(tempfile.mkdtemp(), "astronaut_render.usdc")
    layer = stage.Flatten()  # asset paths are anchored to the asset, so the copy resolves its textures
    layer.Export(tmp)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=tmp, import_skeletons=True, import_blendshapes=True, import_usd_preview=True,
                          set_material_blend=True)
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            for p in o.data.polygons:
                p.use_smooth = True
    arm = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if arm is not None:
        arm.name = "astronaut_skeleton"
        arm.data.display_type = "STICK"
    bpy.ops.file.pack_all()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    keys = sum(len(o.data.shape_keys.key_blocks) - 1 for o in meshes if o.data.shape_keys)
    print(f"wrote {out}: {len(meshes)} meshes, armature {arm.name if arm else None} "
          f"({len(arm.data.bones) if arm else 0} bones), {keys} shape keys, {len(bpy.data.images)} images")


if __name__ == "__main__":
    main()
