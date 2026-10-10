"""Render a filmstrip of an animated UsdSkel file (Blender imports the skeleton + animation).
blender -b --python scripts/render_anim.py -- <anim.usda> <out_prefix> <first> <last> <step>"""
import sys
import bpy
from mathutils import Vector
src, out, first, last, step = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=src, import_skeletons=True, set_frame_range=True)
sc = bpy.context.scene
arm = next(o for o in sc.objects if o.type == "ARMATURE")
for o in sc.objects:
    if o.type == "MESH":
        o.select_set(True); bpy.context.view_layer.objects.active = o
bpy.ops.object.shade_smooth()
sc.render.engine = "BLENDER_WORKBENCH"; sc.display.shading.show_cavity = True
sc.render.resolution_x, sc.render.resolution_y = 500, 700
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = 2.6
bpy.ops.mesh.primitive_plane_add(size=40)
for f in range(int(first), int(last) + 1, int(step)):
    sc.frame_set(f)
    hips = arm.matrix_world @ arm.pose.bones["Hips"].head if "Hips" in arm.pose.bones else Vector((0, 0, 1))
    d = Vector((1, -0.25, 0.12)).normalized()
    c = Vector((hips.x, hips.y, 1.05))
    cam.location = c + d * 6; cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{out}_{f:04d}.png"; bpy.ops.render.render(write_still=True)
