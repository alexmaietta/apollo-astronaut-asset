"""Cycles hero renders of the posed astronaut on a procedural lunar surface.

    blender -b --factory-startup -P scripts/render_hero.py -- <shot> [<shot> ...] [--samples N] [--scale PCT]
    blender -b --factory-startup -P scripts/render_hero.py -- --grasp [--stills-only | --gif-only] [--samples N]

Shots are defined in SHOTS (pose file build/hero_<pose>.usd from scripts/poses.py). Output: renders/hero/<shot>.png.
--grasp renders the physics grasp recorded by examples/newton/utils/record_grasp.py (build/grasp): the simulated
vertices are written into the imported meshes frame by frame, and the sample carrier and pedestal are rebuilt from
the Newton shapes. Output: renders/grasp/still_<t>.png (GRASP_STILLS) and renders/grasp/gif/<frame>.png.
The USD materials (UsdPreviewSurface slots) are upgraded by name to procedural Cycles materials:
NVIDIA's Omniverse textures (source/textures), lunar dust graded by height, gold LEVA visor.
"""
import math
import os
import random
import sys

import bpy
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGOLITH = (0.30, 0.285, 0.265)

# name: pose, camera position, look-at, focal (mm), sun (azimuth, elevation) deg, resolution, extras
SHOTS = {
    # README banner, three panels on black (studio lighting: key, rim and fill, no ground or dust); gallery.py joins
    # them: the full figure, then macro close-ups of the DCM hose connectors and the RCU name tag
    "hero_full": dict(pose="stand", cam=(5.0, -3.3, 1.05), look=(0.0, 0.0, 1.03), lens=92, res=(600, 1100), studio=True),
    "hero_dcm": dict(pose="stand", cam=(0.78, -0.52, 1.0), look=(0.21, 0.0, 1.12), lens=90, res=(1000, 1100), studio=True,
                     fstop=4.0),
    "hero_rcu": dict(pose="stand", cam=(0.64, -0.36, 1.47), look=(0.26, -0.02, 1.34), lens=95, res=(1000, 1100),
                     studio=True, fstop=4.0),
    # gallery: all 1600 x 1200
    "stand": dict(pose="stand", cam=(3.4, -2.2, 0.75), look=(0, 0, 1.05), lens=45, sun=(-35, 22), res=(1600, 1200)),
    "lope": dict(pose="lope", cam=(1.2, -5.2, 0.9), look=(0, 0, 0.95), lens=70, sun=(-60, 18), res=(1600, 1200)),
    "salute": dict(pose="salute", cam=(4.2, -1.3, 1.25), look=(0, 0.55, 1.1), lens=50, sun=(-25, 25), res=(1600, 1200), flag=True),
    "geology": dict(pose="geology", cam=(3.4, -2.5, 0.8), look=(0.2, -0.05, 0.8), lens=50, sun=(-80, 15), res=(1600, 1200), rocks_near=True),
    "visor": dict(pose="stand", cam=(2.1, -1.05, 1.66), look=(0.12, 0.0, 1.56), lens=100, sun=(-30, 20), res=(1600, 1200)),
    "walk": dict(pose="walk", cam=(9.5, 1.3, 0.45), look=(0, 0.5, 1.0), lens=85, sun=(-30, 16), res=(1600, 1200), earth=(191, 8.0)),
    "glove": dict(pose="stand", cam=(0.75, -1.05, 0.95), look=(0.07, -0.37, 0.9), lens=100, sun=(-35, 22), res=(1600, 1200)),
    "plss": dict(pose="stand", cam=(-3.2, 2.3, 1.35), look=(-0.1, 0.0, 1.15), lens=55, sun=(150, 24), res=(1600, 1200)),
}


def node_tree(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return nt, bsdf


def dusted(nt, color, dust_top, amount=0.85):
    """Base colour mixed toward regolith grey below dust_top metres (world Z), broken up with noise."""
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.interpolation_type = "SMOOTHSTEP"
    rng.inputs["From Min"].default_value = dust_top
    rng.inputs["From Max"].default_value = 0.0
    rng.inputs["To Max"].default_value = amount
    nt.links.new(sep.outputs["Z"], rng.inputs["Value"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 6.0
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(rng.outputs["Result"], mul.inputs[0])
    nt.links.new(noise.outputs["Fac"], mul.inputs[1])
    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    mul2.inputs[1].default_value = 1.6
    nt.links.new(mul.outputs[0], mul2.inputs[0])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.clamp_factor = True
    mix.inputs["A"].default_value = (*color, 1)
    mix.inputs["B"].default_value = (*REGOLITH, 1)
    nt.links.new(mul2.outputs[0], mix.inputs["Factor"])
    return mix.outputs["Result"], mul2.outputs[0]


def solid(name, color, rough, metal=0.0, coat=0.0, dust_top=0.0, transmission=0.0, ior=1.45):
    mat = bpy.data.materials.new(name)
    nt, bsdf = node_tree(mat)
    if dust_top > 0:
        col, _ = dusted(nt, color, dust_top)
        nt.links.new(col, bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["IOR"].default_value = ior
    return mat


DUST = {"Boots": (0.45, 0.85), "Pants": (0.75, 0.6), "Body": (0.75, 0.45), "Accessories": (0.6, 0.4),
        "Gloves": (0.0, 0.0), "Backpack": (0.0, 0.0), "Backpack_top": (0.0, 0.0), "Helmet": (0.0, 0.0)}


def dust_overlay(mat, dust_top, amount):
    """Mix lunar dust over a textured material's base colour, graded by height (world Z) and broken up by noise."""
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None or amount <= 0 or not bsdf.inputs["Base Color"].is_linked:
        return
    src = bsdf.inputs["Base Color"].links[0].from_socket
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.interpolation_type = "SMOOTHSTEP"
    rng.inputs["From Min"].default_value = dust_top
    rng.inputs["From Max"].default_value = 0.0
    rng.inputs["To Max"].default_value = amount
    nt.links.new(sep.outputs["Z"], rng.inputs["Value"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 6.0
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(rng.outputs["Result"], mul.inputs[0])
    nt.links.new(noise.outputs["Fac"], mul.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.clamp_factor = True
    nt.links.new(src, mix.inputs["A"])
    mix.inputs["B"].default_value = (*REGOLITH, 1)
    nt.links.new(mul.outputs[0], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])


def astronaut_materials(dust=True):
    """Keep the imported textured materials (NVIDIA's textures via UsdPreviewSurface); add lunar dust, and use
    Cycles-native gold for the LEVA visor and clear glass for the watch (the source uses MDL OmniGlass)."""
    gold = solid("LEVAGold", (1.0, 0.76, 0.33), 0.035, metal=1.0)
    glass = solid("Glass", (0.95, 0.97, 1.0), 0.02, transmission=1.0, ior=1.5)
    done = set()
    for o in bpy.context.scene.objects:
        if o.type != "MESH" or not o.data.materials:
            continue
        mat = o.data.materials[0]
        base = mat.name.split(".")[0]
        if base == "HelmetVisor":
            o.data.materials[0] = gold
        elif base == "WatchGaugeGlass":
            o.data.materials[0] = glass
        elif dust and mat.name not in done and base in DUST:
            dust_overlay(mat, *DUST[base])
            done.add(mat.name)


def terrain(rocks_near=False):
    import numpy as np
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=900, y_subdivisions=900, size=110)
    g = bpy.context.object
    g.name = "Regolith"
    co = np.empty(len(g.data.vertices) * 3)
    g.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    rng = random.Random(3)
    for _ in range(45):  # craters: bowl + raised rim, kept away from the astronaut
        r = 0.4 + 6.0 * rng.random() ** 3
        a, dist = rng.uniform(0, 2 * math.pi), rng.uniform(3.0 + r, 42.0)
        cx, cy = dist * math.cos(a), dist * math.sin(a)
        q = np.hypot(co[:, 0] - cx, co[:, 1] - cy) / r
        depth = 0.18 * r
        co[:, 2] += np.where(q < 1.0, -depth * (1 - q ** 2), 0.0) + 0.25 * depth * np.exp(-((q - 1.0) / 0.25) ** 2)
    g.data.vertices.foreach_set("co", co.ravel())
    g.data.update()
    vg = g.vertex_groups.new(name="away")
    for v in g.data.vertices:  # flat pad under the astronaut, fading back to flat at the tile edge
        r = Vector((v.co.x, v.co.y)).length
        e = max(abs(v.co.x), abs(v.co.y))
        vg.add([v.index], min(1.0, max(0.0, (r - 1.0) / 3.0), max(0.0, (54.0 - e) / 10.0)), "REPLACE")
    for size, strength, depth in ((9.0, 0.6, 2), (1.6, 0.10, 3), (0.35, 0.025, 2)):  # mid_level 0.5: zero-mean
        tex = bpy.data.textures.new(f"clouds{size}", "CLOUDS")
        tex.noise_scale = size
        tex.noise_depth = depth
        d = g.modifiers.new(f"disp{size}", "DISPLACE")
        d.texture = tex
        d.strength = strength
        d.mid_level = 0.5
        d.texture_coords = "GLOBAL"
        d.vertex_group = "away"
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("RegolithMat")
    nt, bsdf = node_tree(mat)
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 3.0
    n.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.11, 0.105, 0.10, 1)
    ramp.color_ramp.elements[1].color = (0.24, 0.23, 0.215, 1)
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 1.0
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 250.0
    fine.inputs["Detail"].default_value = 10.0
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 60.0
    add = nt.nodes.new("ShaderNodeMath")
    nt.links.new(fine.outputs["Fac"], add.inputs[0])
    nt.links.new(vor.outputs["Distance"], add.inputs[1])
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = 0.6
    b.inputs["Distance"].default_value = 0.01
    nt.links.new(add.outputs[0], b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    g.data.materials.append(mat)
    bpy.ops.mesh.primitive_plane_add(size=4000, location=(0, 0, -0.01))  # far plains to the horizon
    far = bpy.context.object
    far.data.materials.append(mat)
    bpy.context.view_layer.objects.active = g

    # scattered rocks
    rng = random.Random(7)
    rock_mat = bpy.data.materials.new("Rock")
    nt, bsdf = node_tree(rock_mat)
    bsdf.inputs["Base Color"].default_value = (0.20, 0.195, 0.185, 1)
    bsdf.inputs["Roughness"].default_value = 0.95
    dg = bpy.context.evaluated_depsgraph_get()
    placed = 0
    while placed < (140 if rocks_near else 90):
        r = rng.uniform(1.4 if rocks_near else 2.2, 28.0)
        a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        s = 0.03 + 0.35 * rng.random() ** 4
        hit, loc, *_ = bpy.context.scene.ray_cast(dg, Vector((x, y, 10)), Vector((0, 0, -1)))
        if not hit:
            continue
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=s, location=(x, y, loc.z - 0.3 * s))
        rk = bpy.context.object
        rk.scale = (rng.uniform(0.7, 1.4), rng.uniform(0.7, 1.4), rng.uniform(0.4, 0.8))
        rk.rotation_euler = (rng.random() * 3, rng.random() * 3, rng.random() * 6)
        tex = bpy.data.textures.new(f"rock{placed}", "CLOUDS")
        tex.noise_scale = 0.35 * s / 0.2
        tex.noise_depth = 4
        d = rk.modifiers.new("d", "DISPLACE")
        d.texture = tex
        d.texture_coords = "GLOBAL"
        d.strength = s * 0.8
        rk.data.materials.append(rock_mat)
        bpy.ops.object.shade_flat()
        placed += 1


def flag(at=(0.0, 1.25, 0.0)):
    x, y, z = at
    mat = solid("Pole", (0.75, 0.75, 0.76), 0.3, metal=1.0)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.013, depth=2.1, location=(x, y, z + 1.05 - 0.15))
    bpy.context.object.data.materials.append(mat)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.008, depth=0.95, location=(x, y + 0.475, z + 1.92),
                                        rotation=(math.pi / 2, 0, 0))
    bpy.context.object.data.materials.append(mat)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=60, y_subdivisions=40, size=1.0, location=(x, y + 0.475, z + 1.62),
                                    rotation=(math.pi / 2, 0, math.pi / 2))
    f = bpy.context.object
    f.scale = (0.95, 0.6, 1.0)
    w = f.modifiers.new("wave", "WAVE")  # Apollo flags kept their deployment wrinkles
    w.height, w.width, w.speed, w.time_offset = 0.025, 0.18, 0.0, -40
    w.use_normal = True
    w.use_y = False
    m = bpy.data.materials.new("Flag")
    nt, bsdf = node_tree(m)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    stripes = nt.nodes.new("ShaderNodeMath")  # 13 stripes, red at top and bottom
    stripes.operation = "MULTIPLY"
    stripes.inputs[1].default_value = 13 / 2
    nt.links.new(sep.outputs["Y"], stripes.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath")
    fr.operation = "FRACT"
    nt.links.new(stripes.outputs[0], fr.inputs[0])
    st = nt.nodes.new("ShaderNodeMath")
    st.operation = "LESS_THAN"
    st.inputs[1].default_value = 0.5
    nt.links.new(fr.outputs[0], st.inputs[0])
    red_white = nt.nodes.new("ShaderNodeMix")
    red_white.data_type = "RGBA"
    red_white.inputs["A"].default_value = (0.85, 0.85, 0.82, 1)
    red_white.inputs["B"].default_value = (0.55, 0.04, 0.06, 1)
    nt.links.new(st.outputs[0], red_white.inputs["Factor"])
    # canton: near the pole (U small) and top 7 stripes
    cu = nt.nodes.new("ShaderNodeMath")
    cu.operation = "LESS_THAN"
    cu.inputs[1].default_value = 0.4
    nt.links.new(sep.outputs["X"], cu.inputs[0])
    cv = nt.nodes.new("ShaderNodeMath")
    cv.operation = "GREATER_THAN"
    cv.inputs[1].default_value = 6 / 13
    nt.links.new(sep.outputs["Y"], cv.inputs[0])
    canton = nt.nodes.new("ShaderNodeMath")
    canton.operation = "MULTIPLY"
    nt.links.new(cu.outputs[0], canton.inputs[0])
    nt.links.new(cv.outputs[0], canton.inputs[1])
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 22.0
    vor.inputs["Randomness"].default_value = 0.0
    nt.links.new(tc.outputs["UV"], vor.inputs["Vector"])
    star = nt.nodes.new("ShaderNodeMath")
    star.operation = "LESS_THAN"
    star.inputs[1].default_value = 0.13
    nt.links.new(vor.outputs["Distance"], star.inputs[0])
    blue = nt.nodes.new("ShaderNodeMix")
    blue.data_type = "RGBA"
    blue.inputs["A"].default_value = (0.04, 0.07, 0.25, 1)
    blue.inputs["B"].default_value = (0.85, 0.85, 0.82, 1)
    nt.links.new(star.outputs[0], blue.inputs["Factor"])
    final = nt.nodes.new("ShaderNodeMix")
    final.data_type = "RGBA"
    nt.links.new(canton.outputs[0], final.inputs["Factor"])
    nt.links.new(red_white.outputs["Result"], final.inputs["A"])
    nt.links.new(blue.outputs["Result"], final.inputs["B"])
    nt.links.new(final.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Sheen Weight"].default_value = 0.3
    f.data.materials.append(m)
    bpy.ops.object.shade_smooth()


def earth(az, el, dist=900.0):
    a, e = math.radians(az), math.radians(el)
    loc = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))) * dist
    bpy.ops.mesh.primitive_uv_sphere_add(radius=dist * math.tan(math.radians(1.0)), location=loc, segments=96, ring_count=48)
    m = bpy.data.materials.new("Earth")
    nt, bsdf = node_tree(m)
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 2.5
    n.inputs["Detail"].default_value = 10.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.01, 0.03, 0.16, 1)
    ramp.color_ramp.elements[0].position = 0.5
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (0.62, 0.64, 0.68, 1)
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    bpy.context.object.data.materials.append(m)
    bpy.ops.object.shade_smooth()


def setup_render(res, samples, scale):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            if any(d.type == dev_type for d in prefs.devices):
                break
        except TypeError:
            continue
    for d in prefs.devices:
        d.use = d.type != "CPU"
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPTIX" if prefs.compute_device_type == "OPTIX" else "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 8
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = scale
    sc.view_settings.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            continue
    sc.view_settings.exposure = -0.6
    world = bpy.data.worlds.new("Space")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    sc.world = world


def studio_world(world):
    """Black to the camera, but a soft overhead gradient in reflections (so the gold visor and glossy parts read)."""
    nt = world.node_tree
    bg = nt.nodes["Background"]
    out = nt.nodes["World Output"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    # direction Z in -1..1 (clamped to 0..1 by the ramp): dark below the horizon, a bright band above it
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (0.02, 0.02, 0.02, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.35, (0.75, 0.75, 0.78, 1)
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    env = nt.nodes.new("ShaderNodeBackground")
    env.inputs["Strength"].default_value = 0.6
    nt.links.new(ramp.outputs["Color"], env.inputs["Color"])
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(env.outputs["Background"], mix.inputs[1])
    nt.links.new(bg.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def render_shot(name, samples, scale):
    shot = SHOTS[name]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=os.path.join(ROOT, "build", f"hero_{shot['pose']}.usd"), import_skeletons=False)
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            for p in o.data.polygons:
                p.use_smooth = True
    studio = shot.get("studio", False)
    astronaut_materials(dust=not studio)
    if not studio:
        terrain(shot.get("rocks_near", False))
    if shot.get("flag"):
        flag()
    if shot.get("earth"):
        earth(*shot["earth"])
    sc = bpy.context.scene
    # (energy, azimuth, elevation, angular size): the lunar sun, or a studio key / rim / fill on black
    lights = [(5.0, -35, 32, 4.0), (9.0, 150, 22, 2.0), (0.9, 60, 8, 20.0)] if studio else [(7.0, *shot["sun"], 0.53)]
    for k, (energy, az, el, size) in enumerate(lights):
        sun = bpy.data.objects.new(f"Sun{k}", bpy.data.lights.new(f"Sun{k}", "SUN"))
        sc.collection.objects.link(sun)
        sun.data.energy = energy
        sun.data.angle = math.radians(size)
        az, el = math.radians(az), math.radians(el)
        d = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        sun.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.location = shot["cam"]
    cam.data.lens = shot["lens"]
    cam.data.sensor_width = 36
    cam.data.clip_end = 3000
    cam.rotation_euler = (Vector(shot["look"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
    if shot.get("fstop"):  # macro shots: focus on the look-at point
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = (Vector(shot["look"]) - cam.location).length
        cam.data.dof.aperture_fstop = shot["fstop"]
    setup_render(shot["res"], samples, scale)
    if studio:
        studio_world(sc.world)
    os.makedirs(os.path.join(ROOT, "renders", "hero"), exist_ok=True)
    sc.render.filepath = os.path.join(ROOT, "renders", "hero", f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print("rendered", sc.render.filepath)


# ---------------------------------------------------------------- physics grasp close-ups (build/grasp)
GRASP_DIR = os.path.join(ROOT, "build", "grasp")
GRASP_STILLS = (0.5, 1.0, 1.5, 2.5, 4.0)  # [s] open, closing, closed on the handle, lifting, holding
GRASP_CAM = dict(offset=(0.60, -0.36, -0.08), aim=(0.0, 0.0, 0.02), lens=55, sun=(-50, 28))  # relative to the handle


def _quat_xyzw(q):
    from mathutils import Quaternion
    return Quaternion((q[3], q[0], q[1], q[2]))


def grasp_props(d):
    """Sample carrier (anodized box, aluminium handle and posts) and its stand, from the recorded Newton shapes.
    Returns the carrier's parent object (posed per frame from the recorded body transform)."""
    anod = solid("CarrierAnodized", (0.13, 0.135, 0.15), 0.35, metal=0.9)
    alu = solid("Aluminium", (0.80, 0.80, 0.82), 0.22, metal=1.0)
    stand = solid("StandAluminium", (0.62, 0.62, 0.63), 0.45, metal=1.0, dust_top=0.25)
    carrier = bpy.data.objects.new("SampleCarrier", None)
    bpy.context.scene.collection.objects.link(carrier)
    for typ, xf, sc, body in zip(d["shape_type"], d["shape_xform"], d["shape_scale"], d["shape_body"]):
        loc, rq = Vector(xf[:3]), _quat_xyzw(xf[3:])
        if typ == int(d["geo_box"]):
            bpy.ops.mesh.primitive_cube_add(size=2.0)
            o = bpy.context.object
            o.scale = tuple(sc)
            bev = o.modifiers.new("bevel", "BEVEL")
            bev.width, bev.segments = min(0.004, 0.4 * float(min(sc))), 3
            bev.affect = "EDGES"
            o.data.materials.append(stand if body < 0 else (anod if min(sc) > 0.01 else alu))
        else:  # capsule along local Z: radius, half height
            r, h = float(sc[0]), float(sc[1])
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=2 * h, vertices=48)
            o = bpy.context.object
            for z in (-h, h):
                bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0, 0, z), segments=48, ring_count=24)
                cap = bpy.context.object
                cap.select_set(True)
                o.select_set(True)
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.join()
            o.data.materials.append(alu)
        bpy.ops.object.shade_smooth()
        o.rotation_mode = "QUATERNION"
        o.location, o.rotation_quaternion = loc, rq
        if body >= 0:
            o.parent = carrier
    carrier.rotation_mode = "QUATERNION"
    return carrier


def grasp_scene(d, samples, res):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=os.path.join(ROOT, "build", "hero_stand.usd"), import_skeletons=False)
    meshes = {}
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            o.matrix_world = o.matrix_world.Identity(4)  # recorded points are world (rig frame) positions
            if o.data.shape_keys:  # corrective shape keys: Blender would draw the Basis key, not the written vertices
                o.shape_key_clear()  # (the recorded points already include the correctives)
            for p in o.data.polygons:
                p.use_smooth = True
            meshes[o.name.split(".")[0]] = o
    astronaut_materials()
    terrain(False)
    carrier = grasp_props(d)
    sc = bpy.context.scene
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sc.collection.objects.link(sun)
    sun.data.energy = 7.0
    sun.data.angle = math.radians(0.53)
    az, el = map(math.radians, GRASP_CAM["sun"])
    dvec = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    sun.rotation_euler = (-dvec).to_track_quat("-Z", "Y").to_euler()
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    h = Vector(tuple(d["handle_c"]))
    cam.location = h + Vector(GRASP_CAM["offset"])
    cam.data.lens = GRASP_CAM["lens"]
    cam.data.sensor_width = 36
    cam.data.clip_end = 3000
    cam.data.dof.use_dof = True  # macro-style shallow focus on the glove
    cam.data.dof.focus_distance = (cam.location - (h + Vector(GRASP_CAM["aim"]))).length
    cam.data.dof.aperture_fstop = 5.6
    cam.rotation_euler = (h + Vector(GRASP_CAM["aim"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
    setup_render(res, samples, 100)
    return meshes, carrier


def grasp_frame(d, meshes, carrier, k):
    import numpy as np
    pts = np.load(os.path.join(GRASP_DIR, f"points_{int(d['frames'][k]):04d}.npy"))
    off = 0
    for name, n in zip(d["mesh_names"], d["mesh_counts"]):
        n = int(n)
        o = meshes.get(str(name))
        if o is not None:
            assert len(o.data.vertices) == n, f"{name}: {len(o.data.vertices)} vertices in Blender, {n} recorded"
            o.data.vertices.foreach_set("co", pts[off:off + n].ravel())
            o.data.update()
        off += n
    q = d["carrier_q"][k]
    carrier.location, carrier.rotation_quaternion = Vector(q[:3]), _quat_xyzw(q[3:])


WALK_DIR = os.path.join(ROOT, "build", "walk")
WALK_CAM = dict(offset=(2.4, -3.2, 0.6), aim=(0.0, 0.0, -0.08), lens=50, sun=(-60, 24))  # relative to the hips


def render_walk(samples, res=(960, 720)):
    """Walk-cycle GIF frames from examples/newton/utils/record_walk.py: the camera tracks the hips (height fixed), the
    recorded vertices are written into the meshes frame by frame. Output: renders/walk/<frame>.png."""
    import numpy as np
    d = dict(np.load(os.path.join(WALK_DIR, "walk.npz")))
    out = os.path.join(ROOT, "renders", "walk")
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=os.path.join(ROOT, "build", "hero_stand.usd"), import_skeletons=False)
    meshes = {}
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            o.matrix_world = o.matrix_world.Identity(4)
            if o.data.shape_keys:
                o.shape_key_clear()
            for p in o.data.polygons:
                p.use_smooth = True
            meshes[o.name.split(".")[0]] = o
    astronaut_materials()
    terrain(False)
    earth(150, 9.0)
    sc = bpy.context.scene
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sc.collection.objects.link(sun)
    sun.data.energy = 7.0
    sun.data.angle = math.radians(0.53)
    az, el = map(math.radians, WALK_CAM["sun"])
    dvec = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    sun.rotation_euler = (-dvec).to_track_quat("-Z", "Y").to_euler()
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = WALK_CAM["lens"]
    cam.data.sensor_width = 36
    cam.data.clip_end = 3000
    setup_render(res, samples, 100)
    z0 = float(np.mean(d["hips"][:, 2]))
    for k, f in enumerate(d["frames"]):
        pts = np.load(os.path.join(WALK_DIR, f"points_{int(f):04d}.npy"))
        off = 0
        for name, n in zip(d["mesh_names"], d["mesh_counts"]):
            n = int(n)
            o = meshes.get(str(name))
            if o is not None:
                o.data.vertices.foreach_set("co", pts[off:off + n].ravel())
                o.data.update()
            off += n
        h = Vector((float(d["hips"][k, 0]), float(d["hips"][k, 1]), z0))
        cam.location = h + Vector(WALK_CAM["offset"])
        cam.rotation_euler = (h + Vector(WALK_CAM["aim"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(out, f"{k:04d}.png")
        bpy.ops.render.render(write_still=True)
    print("rendered", len(d["frames"]), "walk frames")


def render_grasp(samples, stills=True, gif=True):
    import numpy as np
    d = dict(np.load(os.path.join(GRASP_DIR, "grasp.npz")))
    out = os.path.join(ROOT, "renders", "grasp")
    os.makedirs(os.path.join(out, "gif"), exist_ok=True)
    sc = bpy.context.scene
    if stills:
        meshes, carrier = grasp_scene(d, samples, (1200, 1200))
        sc = bpy.context.scene
        for t in GRASP_STILLS:
            k = int(np.argmin(np.abs(d["times"] - t)))
            grasp_frame(d, meshes, carrier, k)
            sc.render.filepath = os.path.join(out, f"still_{t:.2f}s.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", sc.render.filepath)
    if gif:
        meshes, carrier = grasp_scene(d, max(16, samples // 8), (800, 800))
        sc = bpy.context.scene
        for k in range(len(d["frames"])):
            grasp_frame(d, meshes, carrier, k)
            sc.render.filepath = os.path.join(out, "gif", f"{k:04d}.png")
            bpy.ops.render.render(write_still=True)
        print("rendered", len(d["frames"]), "GIF frames")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    samples, scale, shots, grasp, stills, gif = 384, 100, [], False, True, True
    i = 0
    while i < len(argv):
        if argv[i] == "--samples":
            samples = int(argv[i + 1]); i += 2
        elif argv[i] == "--scale":
            scale = int(argv[i + 1]); i += 2
        elif argv[i] == "--walk":
            shots.append("--walk"); i += 1
        elif argv[i] in ("--grasp", "--stills-only", "--gif-only"):
            grasp = True
            stills, gif = (argv[i] != "--gif-only") and stills, (argv[i] != "--stills-only") and gif
            i += 1
        else:
            shots.append(argv[i]); i += 1
    if "--walk" in shots:
        render_walk(samples)
    elif grasp:
        render_grasp(samples, stills, gif)
    else:
        for s in shots or list(SHOTS):
            render_shot(s, samples, scale)
