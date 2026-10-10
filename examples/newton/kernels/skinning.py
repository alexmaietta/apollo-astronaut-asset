"""Shared Newton helpers: draw the skinned, textured astronaut in a Newton viewer.

AstronautRenderer skins all 389k vertices on the GPU each frame (linear blend skinning plus the shoulder
corrective blend shapes) and logs one textured mesh per material slot, splitting vertices only at UV seams.
"""
import os

import numpy as np
import warp as wp

from rig import Rig, open_render_stage

from . import REPO as ROOT

# the committed SimReady asset (Git LFS); scripts/package_simready.py builds it, ./pipeline.sh assets copies it here
ASSET = os.path.join(ROOT, "assets", "astronaut", "astronaut.usda")
RIG = ASSET  # its skeleton, skinned meshes, correctives and looks drive the renderer


@wp.kernel
def _skin(bind: wp.array[wp.vec3], idx: wp.array2d[wp.int32], w: wp.array2d[wp.float32],
          mats: wp.array[wp.mat44], out: wp.array[wp.vec3]):
    v = wp.tid()
    p = wp.vec4(bind[v][0], bind[v][1], bind[v][2], 1.0)
    acc = wp.vec4(0.0)
    for k in range(4):
        acc += w[v, k] * (mats[idx[v, k]] @ p)
    out[v] = wp.vec3(acc[0], acc[1], acc[2])


@wp.kernel
def _gather(src: wp.array[wp.vec3], idx: wp.array[wp.int32], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = src[idx[i]]


@wp.kernel
def _skin_matrices(W: wp.array[wp.mat44], inv_bind: wp.array[wp.mat44], S: wp.array[wp.mat44]):
    j = wp.tid()
    S[j] = W[j] @ inv_bind[j]


@wp.kernel
def _hand_joints_from_bodies(body_q: wp.array[wp.transform], order: wp.array[wp.int32], joint_body: wp.array[wp.int32],
                             parent: wp.array[wp.int32], rel: wp.array[wp.mat44], W: wp.array[wp.mat44]):
    # one thread walks the hand joints parent-first: joints with a physics body take its pose (the bodies sit at
    # the OpenXR joint frames), the rest (tips, palm, metacarpals II-V) follow their parent rigidly
    for k in range(order.shape[0]):
        i = order[k]
        b = joint_body[i]
        if b >= 0:
            W[i] = wp.transform_to_matrix(body_q[b])
        else:
            W[i] = W[parent[i]] @ rel[i]


@wp.kernel
def _restore(src: wp.array[wp.vec3], dst: wp.array[wp.vec3], idx: wp.array[wp.int32]):
    i = wp.tid()
    dst[idx[i]] = src[idx[i]]


@wp.kernel
def _add_shape(dst: wp.array[wp.vec3], idx: wp.array[wp.int32], off: wp.array[wp.vec3], w: float):
    i = wp.tid()
    dst[idx[i]] = dst[idx[i]] + w * off[i]


@wp.kernel
def _min_z(points: wp.array[wp.vec3], start: int, out: wp.array[float]):
    i = wp.tid()
    wp.atomic_min(out, 0, points[start + i][2])


@wp.kernel
def _shift_z(points: wp.array[wp.vec3], minz: wp.array[float]):
    i = wp.tid()
    p = points[i]
    points[i] = wp.vec3(p[0], p[1], p[2] - minz[0])


def textured_groups(rig_path, rig, offsets):
    """One drawable per material slot with per-vertex UVs (vertices duplicated only at UV seams).
    Returns {slot: dict(gather, uvs, tris, texture)}; texture is the base-colour image path or None."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = open_render_stage(rig_path)
    groups = {}
    for m, off in zip(rig.meshes, offsets):
        prim = stage.GetPrimAtPath(rig.root.AppendPath(f"Geometry/{m['name']}"))
        slot = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].GetPrim().GetName()
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        uv_vals = np.array(st.Get(), np.float32)
        uv_idx = np.array(st.GetIndices()) if st.IsIndexed() else np.arange(len(uv_vals))
        if st.GetInterpolation() in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
            uv_idx = uv_idx[m["fvi"]]  # per-vertex UVs -> per face corner
        c, fvi = m["counts"], m["fvi"]
        starts = np.r_[0, np.cumsum(c)[:-1]]
        corners = np.concatenate([np.stack([starts[c >= k + 2], (starts + k)[c >= k + 2], (starts + k + 1)[c >= k + 2]], 1)
                                  for k in range(1, c.max() - 1)])
        key = fvi[corners.ravel()].astype(np.int64) * (len(uv_vals) + 1) + uv_idx[corners.ravel()]
        uniq, inv = np.unique(key, return_inverse=True)
        g = groups.setdefault(slot, dict(gather=[], uvs=[], tris=[], n=0))
        g["gather"].append((uniq // (len(uv_vals) + 1)) + off)
        g["uvs"].append(uv_vals[uniq % (len(uv_vals) + 1)])
        g["tris"].append(inv.reshape(-1, 3) + g["n"])
        g["n"] += len(uniq)
    out = {}
    for slot, g in groups.items():
        bc = stage.GetPrimAtPath(rig.root.AppendPath(f"Looks/{slot}/BaseColorTex"))
        tex = (bc.GetAttribute("inputs:file").Get().resolvedPath or None) if bc else None
        out[slot] = dict(gather=np.concatenate(g["gather"]).astype(np.int32), uvs=np.concatenate(g["uvs"]),
                         tris=np.concatenate(g["tris"]).astype(np.int32).ravel(), texture=tex)
    return out


def _load_image(path, max_size):
    """RGBA uint8 image, downsampled so its longest side is at most max_size."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        im = im.convert("RGBA")
        if max(im.size) > max_size:
            im = im.resize((max_size * im.size[0] // max(im.size), max_size * im.size[1] // max(im.size)), Image.LANCZOS)
        return np.ascontiguousarray(np.array(im))


class AstronautRenderer:
    def __init__(self, device, rig=None, name="/astronaut"):
        self.rig = rig or Rig(RIG)
        self.device, self.name = device, name
        rig = self.rig
        counts = [len(m["P"]) for m in rig.meshes]
        offs = np.cumsum([0] + counts[:-1])
        self.bind0 = np.concatenate([m["P"] for m in rig.meshes]).astype(np.float32)
        self.bind_pts = wp.array(self.bind0, dtype=wp.vec3, device=device)
        self.idx = wp.array(np.concatenate([m["idx"] for m in rig.meshes]).astype(np.int32), device=device)
        self.w = wp.array(np.concatenate([m["w"] for m in rig.meshes]).astype(np.float32), device=device)
        self.points = wp.zeros_like(self.bind_pts)
        self.mats = wp.zeros(len(rig.names), dtype=wp.mat44, device=device)
        sole = [m["name"] for m in rig.meshes].index("boot_soles")
        self.sole_slice = slice(int(offs[sole]), int(offs[sole] + counts[sole]))
        shapes = {}
        for m, o in zip(rig.meshes, offs):
            for nm, (ix, off) in m["shapes"].items():
                shapes.setdefault(nm, []).append((ix + o, off.astype(np.float32)))
        self.shapes = {nm: (wp.array(np.concatenate([a for a, _ in v]).astype(np.int32), device=device),
                            wp.array(np.concatenate([b for _, b in v]), dtype=wp.vec3, device=device))
                       for nm, v in shapes.items()}
        touched = np.unique(np.concatenate([a for v in shapes.values() for a, _ in v])) if shapes else np.zeros(0)
        self.shape_touched = wp.array(touched.astype(np.int32), device=device)
        self.bind0_wp = wp.array(self.bind0, dtype=wp.vec3, device=device)
        self._shapes_active = False
        self.inv_bind = wp.array(rig.inv_bind.astype(np.float32), dtype=wp.mat44, device=device)
        self.W = wp.zeros(len(rig.names), dtype=wp.mat44, device=device)
        self.minz = wp.zeros(1, dtype=float, device=device)
        self.groups = textured_groups(RIG, rig, offs)
        self._uploaded = set()
        for g in self.groups.values():
            g["image"] = _load_image(g["texture"], 2048) if g["texture"] else None
            g["gather_wp"] = wp.array(g["gather"], dtype=wp.int32, device=device)
            g["uvs_wp"] = wp.array(g["uvs"], dtype=wp.vec2, device=device)
            g["tris_wp"] = wp.array(g["tris"], dtype=wp.int32, device=device)
            g["pts_wp"] = wp.zeros(len(g["gather"]), dtype=wp.vec3, device=device)

    @property
    def vertex_count(self):
        return len(self.bind0)

    def bind_hands(self, body_index, sides=("Left", "Right")):
        """Register physics hand bodies ({rig joint name: body index}) so skin() can pose the hands on the GPU."""
        rig = self.rig
        order = [i for i, n in enumerate(rig.names) if any(n.startswith(f"{s}Hand") for s in sides)]
        joint_body = np.full(len(rig.names), -1, np.int32)
        for n, b in body_index.items():
            if n in rig.index:
                joint_body[rig.index[n]] = b
        rel = np.array([np.linalg.inv(rig.bind[max(p, 0)]) @ rig.bind[i] for i, p in enumerate(rig.parent)])
        self._hand_order = wp.array(np.array(order, np.int32), device=self.device)
        self._joint_body = wp.array(joint_body, device=self.device)
        self._parent = wp.array(np.array(rig.parent, np.int32), device=self.device)
        self._rel = wp.array(rel.astype(np.float32), dtype=wp.mat44, device=self.device)

    def skin(self, W, ground=False, body_q=None):
        """Skin on the GPU. W: (n_joints, 4, 4) joint world matrices (CPU, column vectors). If body_q (device
        transforms) is given and bind_hands() was called, the hand joints come from the simulated bodies instead.
        ground=True drops the soles onto z = 0 (on the device, no readback). Correctives use W's arm directions."""
        weights = self.rig.corrective_weights(W)
        active = any(abs(v) > 1e-4 for v in weights.values())
        if active or self._shapes_active:
            wp.launch(_restore, dim=len(self.shape_touched), inputs=[self.bind0_wp, self.bind_pts, self.shape_touched],
                      device=self.device)
            for nm, wv in weights.items():
                if abs(wv) > 1e-4 and nm in self.shapes:
                    ix, off = self.shapes[nm]
                    wp.launch(_add_shape, dim=len(ix), inputs=[self.bind_pts, ix, off, float(wv)], device=self.device)
        self._shapes_active = active
        self.W.assign(wp.array(W.astype(np.float32), dtype=wp.mat44, device=self.device))
        if body_q is not None:
            wp.launch(_hand_joints_from_bodies, dim=1, inputs=[body_q, self._hand_order, self._joint_body, self._parent,
                      self._rel], outputs=[self.W], device=self.device)
        wp.launch(_skin_matrices, dim=len(self.rig.names), inputs=[self.W, self.inv_bind], outputs=[self.mats],
                  device=self.device)
        wp.launch(_skin, dim=len(self.points), inputs=[self.bind_pts, self.idx, self.w, self.mats],
                  outputs=[self.points], device=self.device)
        if ground:
            self.minz.fill_(1.0e9)
            n = self.sole_slice.stop - self.sole_slice.start
            wp.launch(_min_z, dim=n, inputs=[self.points, self.sole_slice.start, self.minz], device=self.device)
            wp.launch(_shift_z, dim=len(self.points), inputs=[self.points, self.minz], device=self.device)
        return W

    def log(self, viewer, hidden=False):
        for slot, g in self.groups.items():
            wp.launch(_gather, dim=len(g["gather"]), inputs=[self.points, g["gather_wp"]], outputs=[g["pts_wp"]],
                      device=self.device)
            name = f"{self.name}/{slot}"
            if g["image"] is not None:
                obj = getattr(viewer, "objects", {}).get(name)
                first = name not in self._uploaded or obj is None
                # Newton's GL mesh decodes and re-uploads its texture on every update; send the (2K) image once
                # and keep it on the GPU afterwards, updating only the vertices.
                viewer.log_mesh(name, g["pts_wp"], g["tris_wp"], uvs=g["uvs_wp"], texture=g["image"] if first else None,
                                color=(1.0, 1.0, 1.0), roughness=0.8, hidden=hidden)
                obj = getattr(viewer, "objects", {}).get(name)
                if obj is not None:
                    if first:
                        obj.update_texture = lambda texture=None: None  # keep the uploaded texture
                        self._uploaded.add(name)
                    obj.material = (obj.material[0], obj.material[1], 0.0, 1.0)  # shader samples it when w > 0.5
            else:  # visor / watch glass
                gold = slot == "HelmetVisor"
                viewer.log_mesh(name, g["pts_wp"], g["tris_wp"], color=(1.0, 0.76, 0.33) if gold else (0.9, 0.92, 0.95),
                                roughness=0.08, metallic=1.0 if gold else 0.0, hidden=hidden)


def joint_world_from_bodies(rig, W, body_q, body_index, side):
    """Overwrite one hand's joint world matrices in W from simulated hand bodies.

    The physics bodies are posed at the OpenXR joint frames of the rig's bind pose, so a body's world transform
    is that joint's world matrix. Joints without a body (finger tips, palm, metacarpals II-V) follow their parent."""
    from scipy.spatial.transform import Rotation as R
    W = W.copy()
    for i, n in enumerate(rig.names):
        if not n.startswith(f"{side}Hand"):
            continue
        if n in body_index:
            q = body_q[body_index[n]]
            W[i] = np.eye(4)
            W[i, :3, :3] = R.from_quat(q[3:]).as_matrix()
            W[i, :3, 3] = q[:3]
    for i, n in enumerate(rig.names):  # parent-before-child order: dependents follow
        if n.startswith(f"{side}Hand") and n not in body_index:
            p = rig.parent[i]
            W[i] = W[p] @ np.linalg.inv(rig.bind[p]) @ rig.bind[i]
    return W
