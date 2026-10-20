"""
sim/dexel/renderer.py — GPU ray casting renderer for a tri-dexel stock.

Drop-in for VoxelRenderer (same ``render(mvp, cam_pos)`` / ``release()``,
same colour setting), drawn by the Viewport via set_voxel_renderer().

Rendering design
----------------
A fullscreen triangle; per pixel the view ray is clipped to the stock AABB
and then walks the Z-ray columns with a 2-D DDA (one column per ray
spacing h in X/Y). Inside a column the material intervals are exact in Z,
so the ray's first entry into any of them is solved analytically — no
fixed-step marching, no step-size artefacts:

• Entered through the column's top/bottom → horizontal surface hit, exact.
  Normal from the slope of the same surface in the four neighbouring
  columns (nearest end/start value, central or one-sided differences).
• Entered through a column side → wall. The wall position along X (or Y)
  is then refined from the X (or Y) ray at the hit's height, which stores
  it exactly; the normal comes from that wall's slope across neighbouring
  X/Y rays. Without a matching endpoint nearby, the axis-aligned column
  side is used as is.

OpenGL 3.3 core: sampler2DArray + texelFetch + gl_FragDepth.
"""
from __future__ import annotations

import moderngl
import numpy as np

from controller.sim.core.settings import AppSettings
from controller.sim.dexel.gpu_fields import GpuDexelGrid

_VERT = """
#version 330 core
const vec2 NDC[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(NDC[gl_VertexID], 0.0, 1.0); }
"""

_FRAG = """
#version 330 core

uniform sampler2DArray u_tex_x;   // rays along X: texel (iy, iz)
uniform sampler2DArray u_tex_y;   // rays along Y: texel (ix, iz)
uniform sampler2DArray u_tex_z;   // rays along Z: texel (ix, iy)
uniform ivec3 u_k;                // interval slots per field (x, y, z)
uniform ivec3 u_n;                // ray grid (nx, ny, nz)
uniform float u_h;                // ray spacing, mm
uniform vec3  u_origin;           // grid origin = bbox min
uniform vec3  u_bmin;
uniform vec3  u_bmax;
uniform mat4  u_mvp;
uniform mat4  u_inv_mvp;
uniform vec3  u_cam_pos;
uniform vec2  u_viewport;
uniform vec3  u_base_color;

const vec3  LIGHT_DIR = normalize(vec3(0.50, 0.70, 1.00));
const float AMBIENT   = 0.10;
const float DIFF_STR  = 0.82;
const float SPEC_POW  = 32.0;
const float SPEC_STR  = 0.12;
const int   MAX_STEPS = 8192;
const int   MAX_K     = 32;
const float BIG       = 1.0e29;

out vec4 fragColor;

// Endpoint (start if use_start, else end) of ray c closest to ref, within
// tol. ok = false if c is outside the field or nothing is close enough.
float nearest_endpoint(sampler2DArray tex, int k, ivec2 c, ivec2 size,
                       bool use_start, float ref, float tol, out bool ok) {
    ok = false;
    if (c.x < 0 || c.y < 0 || c.x >= size.x || c.y >= size.y) return ref;
    float best = ref;
    float best_d = tol;
    for (int i = 0; i < MAX_K; i++) {
        if (i >= k) break;
        vec2 se = texelFetch(tex, ivec3(c, i), 0).rg;
        if (se.x > se.y) continue;
        float v = use_start ? se.x : se.y;
        float d = abs(v - ref);
        if (d <= best_d) { best_d = d; best = v; ok = true; }
    }
    return best;
}

// Surface slope along one field-plane direction (central difference with
// one-sided fallback; 0 across discontinuities).
float slope(sampler2DArray tex, int k, ivec2 c, ivec2 off, ivec2 size,
            bool use_start, float ref) {
    float tol = 6.0 * u_h;
    bool okp, okm;
    float vp = nearest_endpoint(tex, k, c + off, size, use_start, ref, tol, okp);
    float vm = nearest_endpoint(tex, k, c - off, size, use_start, ref, tol, okm);
    if (okp && okm) return (vp - vm) / (2.0 * u_h);
    if (okp) return (vp - ref) / u_h;
    if (okm) return (ref - vm) / u_h;
    return 0.0;
}

void main() {
    vec2 ndc    = (gl_FragCoord.xy / u_viewport) * 2.0 - 1.0;
    vec4 near_h = u_inv_mvp * vec4(ndc, -1.0, 1.0);
    vec4 far_h  = u_inv_mvp * vec4(ndc,  1.0, 1.0);
    near_h /= near_h.w;
    far_h  /= far_h.w;
    vec3 ro = near_h.xyz;
    vec3 rd = normalize(far_h.xyz - near_h.xyz);

    // ── Ray vs AABB ───────────────────────────────────────────────────────
    vec3 inv = 1.0 / rd;
    vec3 t0s = (u_bmin - ro) * inv;
    vec3 t1s = (u_bmax - ro) * inv;
    vec3 tmin = min(t0s, t1s);
    vec3 tmax = max(t0s, t1s);
    float tenter = max(max(tmin.x, tmin.y), tmin.z);
    float texit  = min(min(tmax.x, tmax.y), tmax.z);
    if (texit < 0.0 || tenter > texit) discard;

    int last_axis = (tmin.x >= tmin.y && tmin.x >= tmin.z) ? 0
                  : (tmin.y >= tmin.z ? 1 : 2);
    float t = max(tenter, 0.0);
    if (tenter < 0.0) last_axis = 2;

    // ── 2-D DDA over Z-ray columns ────────────────────────────────────────
    vec3 p = ro + t * rd;
    ivec2 cell = clamp(ivec2(floor((p.xy - u_origin.xy) / u_h)), ivec2(0), u_n.xy - 1);
    ivec2 stp  = ivec2(rd.x >= 0.0 ? 1 : -1, rd.y >= 0.0 ? 1 : -1);
    vec2 tdelta = vec2(abs(rd.x) > 1e-12 ? u_h / abs(rd.x) : BIG,
                       abs(rd.y) > 1e-12 ? u_h / abs(rd.y) : BIG);
    vec2 next_b = u_origin.xy + (vec2(cell) + vec2(stp.x > 0 ? 1.0 : 0.0,
                                                    stp.y > 0 ? 1.0 : 0.0)) * u_h;
    vec2 tnext = vec2(abs(rd.x) > 1e-12 ? (next_b.x - ro.x) / rd.x : BIG,
                      abs(rd.y) > 1e-12 ? (next_b.y - ro.y) / rd.y : BIG);

    float hit_t = -1.0;
    int   face  = 2;
    vec2  hit_se = vec2(0.0);
    for (int i = 0; i < MAX_STEPS; i++) {
        float t_cell_exit = min(min(tnext.x, tnext.y), texit);
        float best = BIG;
        for (int k = 0; k < MAX_K; k++) {
            if (k >= u_k.z) break;
            vec2 se = texelFetch(u_tex_z, ivec3(cell, k), 0).rg;
            if (se.x > se.y) continue;
            float ta, tb;
            if (abs(rd.z) < 1e-12) {
                float z = ro.z + t * rd.z;
                if (z < se.x || z > se.y) continue;
                ta = -BIG; tb = BIG;
            } else {
                ta = (se.x - ro.z) / rd.z;
                tb = (se.y - ro.z) / rd.z;
                if (ta > tb) { float s = ta; ta = tb; tb = s; }
            }
            float lo = max(ta, t);
            if (lo <= min(tb, t_cell_exit) && lo < best) { best = lo; hit_se = se; }
        }
        if (best < BIG) {
            hit_t = best;
            face = (best <= t + 1e-6 * max(1.0, abs(t))) ? last_axis : 2;
            // A side hit only a sliver below the column's top (or above its
            // bottom) is a float-level step between neighbouring columns of
            // one flat surface, not a wall — shade it as that surface.
            if (face != 2) {
                float z = ro.z + hit_t * rd.z;
                float lip = rd.z < 0.0 ? hit_se.y - z : z - hit_se.x;
                if (lip < 0.1 * u_h) face = 2;
            }
            break;
        }
        if (t_cell_exit >= texit) break;
        if (tnext.x < tnext.y) {
            t = tnext.x; cell.x += stp.x; tnext.x += tdelta.x; last_axis = 0;
        } else {
            t = tnext.y; cell.y += stp.y; tnext.y += tdelta.y; last_axis = 1;
        }
        if (cell.x < 0 || cell.y < 0 || cell.x >= u_n.x || cell.y >= u_n.y) break;
    }
    if (hit_t < 0.0) discard;

    vec3 pos = ro + hit_t * rd;
    vec3 n;
    if (face == 2) {
        // Horizontal surface: top (ray going down hits an interval end) or
        // bottom (going up hits a start).
        bool top = rd.z < 0.0;
        float s = top ? 1.0 : -1.0;
        float gx = slope(u_tex_z, u_k.z, cell, ivec2(1, 0), u_n.xy, !top, pos.z);
        float gy = slope(u_tex_z, u_k.z, cell, ivec2(0, 1), u_n.xy, !top, pos.z);
        n = s * normalize(vec3(-gx, -gy, 1.0));
    } else {
        // Wall entered while stepping along X (face 0) or Y (face 1).
        int a = face;
        float sa = float(a == 0 ? stp.x : stp.y);       // step direction
        bool use_start = sa > 0.0;                      // entering at a start
        int iz = clamp(int(floor((pos.z - u_origin.z) / u_h)), 0, u_n.z - 1);
        ivec2 c = (a == 0) ? ivec2(cell.y, iz) : ivec2(cell.x, iz);
        ivec2 size = (a == 0) ? u_n.yz : u_n.xz;
        float coord = (a == 0) ? pos.x : pos.y;
        float d_a = (a == 0) ? rd.x : rd.y;
        float o_a = (a == 0) ? ro.x : ro.y;
        bool ok;
        float exact = (a == 0)
            ? nearest_endpoint(u_tex_x, u_k.x, c, size, use_start, coord, 2.5 * u_h, ok)
            : nearest_endpoint(u_tex_y, u_k.y, c, size, use_start, coord, 2.5 * u_h, ok);
        if (ok) {
            hit_t = (exact - o_a) / d_a;
            pos = ro + hit_t * rd;
            float fu = (a == 0)
                ? slope(u_tex_x, u_k.x, c, ivec2(1, 0), size, use_start, exact)
                : slope(u_tex_y, u_k.y, c, ivec2(1, 0), size, use_start, exact);
            float fz = (a == 0)
                ? slope(u_tex_x, u_k.x, c, ivec2(0, 1), size, use_start, exact)
                : slope(u_tex_y, u_k.y, c, ivec2(0, 1), size, use_start, exact);
            // Surface coord_a = f(u, z); outward normal = -sa · (1, -f_u, -f_z)
            vec3 g = (a == 0) ? vec3(1.0, -fu, -fz) : vec3(-fu, 1.0, -fz);
            n = -sa * normalize(g);
        } else {
            n = (a == 0) ? vec3(-sa, 0.0, 0.0) : vec3(0.0, -sa, 0.0);
        }
    }

    vec3  V    = normalize(u_cam_pos - pos);
    vec3  H    = normalize(LIGHT_DIR + V);
    float diff = (dot(n, LIGHT_DIR) * 0.5 + 0.5) * DIFF_STR;
    float spec = pow(max(dot(n, H), 0.0), SPEC_POW) * SPEC_STR;
    // Mild height tint: deeper surfaces slightly darker, so a pocket floor
    // stays distinguishable from the parallel stock top (both have the
    // same, exact normal — unlike the voxel renderer's blurred edges).
    float depth01 = clamp((pos.z - u_bmin.z) / max(u_bmax.z - u_bmin.z, 1e-6), 0.0, 1.0);
    vec3 color = u_base_color * (AMBIENT + diff) * (0.78 + 0.22 * depth01) + vec3(spec);

    vec4 clip = u_mvp * vec4(pos, 1.0);
    gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;
    fragColor = vec4(color, 1.0);
}
"""


class DexelRenderer:
    """Renders a GpuDexelGrid via GLSL ray casting. Construct with the GL
    context current (DatumSimWidget._create_voxel_sim())."""

    def __init__(self, ctx: moderngl.Context, grid: GpuDexelGrid) -> None:
        self._ctx  = ctx
        self._grid = grid
        self._prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self._vao  = ctx.vertex_array(self._prog, [])

        self._s = AppSettings.instance()
        self._apply_color(self._s.voxel_color_rgb())
        self._s.voxel_color_changed.connect(
            lambda _name: self._apply_color(self._s.voxel_color_rgb())
        )

    def _apply_color(self, rgb: tuple[float, float, float]) -> None:
        self._prog["u_base_color"].value = rgb

    def render(self, mvp: np.ndarray, cam_pos: np.ndarray) -> None:
        """Render the dexel stock. Must be called inside paintGL."""
        grid = self._grid
        bbox = grid.bbox
        origin = bbox.origin()
        inv_mvp = np.linalg.inv(mvp).astype("f4")

        prog = self._prog
        prog["u_mvp"].write(mvp.astype("f4").T.tobytes())
        prog["u_inv_mvp"].write(inv_mvp.T.tobytes())
        prog["u_cam_pos"].write(cam_pos.astype("f4").tobytes())
        prog["u_viewport"].value = (float(self._ctx.viewport[2]), float(self._ctx.viewport[3]))
        prog["u_origin"].write(origin.tobytes())
        prog["u_bmin"].write(origin.tobytes())
        prog["u_bmax"].write((origin + bbox.size()).astype("f4").tobytes())
        prog["u_h"].value = float(grid.voxel_size)
        prog["u_n"].value = tuple(int(n) for n in grid.shape)
        prog["u_k"].value = tuple(f.k for f in grid.dexels.fields)

        for loc, (name, tex) in enumerate(zip(("u_tex_x", "u_tex_y", "u_tex_z"), grid.textures)):
            tex.use(location=loc)
            prog[name].value = loc

        self._vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self._prog.release()
        self._vao.release()
