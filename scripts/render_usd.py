"""Clay render of a (Z-up, metres, +X forward) USD from front, side and 3/4 views.
blender -b --python scripts/render_usd.py -- <file.usd> <out_prefix>"""
import sys
import bpy
from mathutils import Vector
src, out = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=src, import_skeletons=False)
sc = bpy.context.scene
for o in sc.objects:
    if o.type == "MESH":
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
bpy.ops.object.shade_smooth()
sc.render.engine = "BLENDER_WORKBENCH"; sc.display.shading.show_cavity = True
sc.render.resolution_x = sc.render.resolution_y = 700
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = float(sys.argv[-1]) if len(sys.argv) > sys.argv.index("--") + 3 else 2.3
for name, d in [("front", (1, 0, 0)), ("side", (0, -1, 0)), ("q", (1, -1.1, 0.4))]:
    d = Vector(d).normalized(); cam.location = Vector((0, 0, 1.0)) + d * 4
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{out}_{name}.png"; bpy.ops.render.render(write_still=True)
