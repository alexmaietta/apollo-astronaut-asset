"""Render the skeleton over the source mesh (Blender, workbench) for visual checks.
blender -b --python scripts/render_skeleton.py -- <astronaut.usd> <outdir>"""
import sys, os
import bpy
from mathutils import Vector
sys.path.insert(0, os.path.dirname(__file__))
from skeleton import JOINTS

src, out = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=src)
sc = bpy.context.scene
ghost = bpy.data.materials.new("ghost"); ghost.diffuse_color = (0.8, 0.8, 0.8, 0.25)
for o in sc.objects:
    if o.type == "MESH":
        o.data.materials.clear(); o.data.materials.append(ghost)
red = bpy.data.materials.new("j"); red.diffuse_color = (1, 0.1, 0.05, 1)
blue = bpy.data.materials.new("b"); blue.diffuse_color = (0.1, 0.3, 1, 1)
s = 0.01  # cm -> m (Blender import converts to metres)
for n, (par, p) in JOINTS.items():
    r = 0.006 if "Hand" in n else 0.015
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=Vector(p) * s, segments=12, ring_count=8)
    o = bpy.context.object; o.data.materials.append(red); o.show_in_front = True
    if par:
        a, b = Vector(JOINTS[par][1]) * s, Vector(p) * s
        bpy.ops.mesh.primitive_cylinder_add(radius=r * 0.4, depth=(b - a).length, location=(a + b) / 2, vertices=8)
        c = bpy.context.object; c.rotation_mode = "QUATERNION"
        c.rotation_quaternion = (b - a).to_track_quat("Z", "Y"); c.data.materials.append(blue)
sc.render.engine = "BLENDER_WORKBENCH"
sc.display.shading.color_type = "MATERIAL"; sc.display.shading.show_xray = True; sc.display.shading.xray_alpha = 0.35
sc.render.resolution_x = sc.render.resolution_y = 900
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"
views = [("front", (0, -1, 0), (0, 0, 1.0), 2.2), ("side", (1, 0, 0), (0, 0, 1.0), 2.2),
         ("handL_out", (1, 0, 0), (0.36, -0.06, 0.88), 0.28), ("handL_front", (0, -1, 0), (0.36, -0.06, 0.88), 0.28),
         ("feet_top", (0, 0, 1), (0.05, -0.08, 0.05), 0.7)]
for name, d, c, scale in views:
    d = Vector(d); cam.data.ortho_scale = scale
    cam.location = Vector(c) + d * 3
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y" if abs(d.z) < 0.9 else "X").to_euler()
    sc.render.filepath = f"{out}/skel_{name}.png"; bpy.ops.render.render(write_still=True)
