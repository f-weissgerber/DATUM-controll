"""
sim/dexel/collision.py — read-only tool/material collision check on dexels.

Same rule, same sampling and same CollisionHit result as
voxel/collision.check_segment() (see its docstring): G0 rapids may not touch
material with any part of tool + holder; cutting moves only with the zone up
to ``cutting_length``.

Material is read from the Z rays, which store it exactly in Z per column.

Two paths
---------
• Monotone outline (the usual case — shank narrower than the holder, holder
  not necking in above its widest point): above the checked height the
  tool + holder occupy ONE height interval per Z ray, so the carver's
  closed-form per-ray sweep (sweep.sweep_z) applies unchanged — exact for
  level moves and plunges, O(rays) per segment.
• Otherwise the outline is evaluated per stamp at the voxel check's
  discrete heights (cell centres), like voxel/collision.py.

Touching within CONTACT_TOL (e.g. the shank resting exactly on the floor the
cutting edge just left behind) is not a collision.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from controller.sim.dexel import _kernels
from controller.sim.dexel.stock import DexelStock
from controller.sim.dexel.sweep import ProfileInverse, stamp_points, sweep_z
from controller.sim.simulation.tool_definition import ToolDefinition
from controller.sim.simulation.tool_holder import HolderProfile
from controller.sim.voxel.collision import CollisionHit, _combined_radius_at_array

CONTACT_TOL = 1e-3   # mm

# (tool geometry, holder, check zone) → ProfileInverse or None (non-monotone).
_inverse_cache: dict[tuple, ProfileInverse | None] = {}


def _outline_inverse(
    tool: ToolDefinition, holder: HolderProfile | None,
    total_len: float, z_lo: float, full_len: float,
) -> ProfileInverse | None:
    holder_key = None if holder is None else (
        holder.name, holder.gauge_length, tuple(map(tuple, holder.profile)),
    )
    key = (
        tool.tool_type, tool.diameter, tool.corner_radius, tool.tip_angle,
        tool.taper_angle, total_len, holder_key, z_lo, full_len,
    )
    if key in _inverse_cache:
        return _inverse_cache[key]

    def outline(hz: np.ndarray) -> np.ndarray:
        return _combined_radius_at_array(np.asarray(hz, "f4"), tool, holder, total_len)

    zone = np.linspace(z_lo, full_len, 4096, dtype="f4")
    monotone = bool(np.all(np.diff(outline(zone)) >= -1e-6))
    inv = ProfileInverse(outline, full_len) if monotone else None
    _inverse_cache[key] = inv
    return inv


def _zone(tool: ToolDefinition, holder: HolderProfile | None, is_rapid: bool):
    """(total_len, z_check_lo, full_len) — the checked height zone."""
    total_len = tool.total_length if tool.total_length > 0.0 else (
        tool.cutting_length if tool.cutting_length > 0.0 else 999.0
    )
    holder_len = holder.gauge_length if holder is not None else 0.0
    return total_len, (0.0 if is_rapid else tool.cutting_length), total_len + holder_len


def is_closed_form(
    tool: ToolDefinition, holder: HolderProfile | None, is_rapid: bool,
) -> bool:
    """True if check_segment() takes the closed-form path for this tool —
    the path whose result is monotone in the move (a sub-move's checked
    volume lies inside the whole move's), which the pre-pass relies on to
    use a whole collinear run's check as a filter for its pieces."""
    total_len, z_lo, full_len = _zone(tool, holder, is_rapid)
    return full_len <= z_lo or _outline_inverse(tool, holder, total_len, z_lo, full_len) is not None


def check_segment(
    dexels:   DexelStock,
    start:    np.ndarray,
    end:      np.ndarray,
    tool:     ToolDefinition,
    is_rapid: bool,
    holder:   HolderProfile | None = None,
) -> CollisionHit | None:
    """Return the first collision of *tool* (+ *holder*) moving start→end,
    or None. See module docstring for the rule."""
    vs = dexels.voxel_size
    total_len, z_check_lo, full_len = _zone(tool, holder, is_rapid)
    if full_len <= z_check_lo:
        return None

    max_r = tool.radius
    if holder is not None and holder.profile:
        max_r = max(max_r, max(r for _, r in holder.profile))

    # Height pre-filter: if every material interval in the move's footprint
    # ends strictly below the lowest checked height, nothing can collide —
    # the common case for rapids at safe height (~80 % of check calls on
    # workpieces/Gcode.cnc). Exact for both paths below: they only look at
    # rays within max_r and heights >= tip + z_check_lo.
    f = dexels.z_field
    iu0, iu1 = f.index_range(min(start[0], end[0]) - max_r, max(start[0], end[0]) + max_r, 0)
    iv0, iv1 = f.index_range(min(start[1], end[1]) - max_r, max(start[1], end[1]) + max_r, 1)
    if iu0 >= iu1 or iv0 >= iv1:
        return None
    if float(f.end[iv0:iv1, iu0:iu1].max()) < min(float(start[2]), float(end[2])) + z_check_lo:
        return None

    inv = _outline_inverse(tool, holder, total_len, z_check_lo, full_len)
    if inv is not None:
        return _check_monotone(
            dexels, start, end, inv, z_check_lo, full_len, total_len, is_rapid,
        )

    pts = stamp_points(start, end, vs, full_len - z_check_lo)

    iu0, iu1 = f.index_range(pts[:, 0].min() - max_r, pts[:, 0].max() + max_r, 0)
    iv0, iv1 = f.index_range(pts[:, 1].min() - max_r, pts[:, 1].max() + max_r, 1)
    if iu0 >= iu1 or iv0 >= iv1:
        return None
    xs = f.u_centers(iu0, iu1)
    ys = f.v_centers(iv0, iv1)
    s_box = f.start[iv0:iv1, iu0:iu1]       # (ly, lx, K)
    e_box = f.end[iv0:iv1, iu0:iu1]

    # Same discrete heights as the voxel grid (cell centres).
    oz = float(dexels.bbox.origin()[2])
    nz = dexels.shape[2]

    for pt in pts:
        tx, ty, tz = float(pt[0]), float(pt[1]), float(pt[2])
        dx = xs - tx
        dy = ys - ty
        r2_xy = dx[np.newaxis, :] ** 2 + dy[:, np.newaxis] ** 2       # (ly, lx)

        iz_lo = max(0,  int(np.floor((tz + z_check_lo - oz) / vs)))
        iz_hi = min(nz, int(np.ceil ((tz + full_len   - oz) / vs)) + 1)
        if iz_lo >= iz_hi:
            continue
        zs = oz + (np.arange(iz_lo, iz_hi, dtype="f4") + 0.5) * vs
        local = zs - tz
        valid = (local >= z_check_lo) & (local <= full_len)
        zs, local = zs[valid], local[valid]
        if len(zs) == 0:
            continue

        radii = _combined_radius_at_array(local, tool, holder, total_len)
        inside = r2_xy[np.newaxis] <= (radii ** 2)[:, np.newaxis, np.newaxis]   # (n, ly, lx)
        if not inside.any():
            continue
        z4 = zs[:, np.newaxis, np.newaxis, np.newaxis]
        solid = np.any((s_box[np.newaxis] <= z4) & (z4 <= e_box[np.newaxis]), axis=-1)
        hit_mask = inside & solid
        if not hit_mask.any():
            continue

        zi, yi, xi = (int(v) for v in np.argwhere(hit_mask)[0])
        world_pt = np.array([xs[xi], ys[yi], zs[zi]], dtype="f4")
        kind: Literal["rapid", "shank", "holder"]
        if is_rapid:
            kind = "rapid"
        elif local[zi] > total_len:
            kind = "holder"
        else:
            kind = "shank"
        return CollisionHit(point=world_pt, kind=kind)

    return None


def _check_monotone(
    dexels:     DexelStock,
    start:      np.ndarray,
    end:        np.ndarray,
    inv:        ProfileInverse,
    z_check_lo: float,
    full_len:   float,
    total_len:  float,
    is_rapid:   bool,
) -> CollisionHit | None:
    """Closed-form path (see module docstring): the checked outline's
    interval per Z ray vs. that ray's material intervals."""
    f = dexels.z_field
    res = sweep_z(f, start, end, inv, z_check_lo, full_len)
    if res is None:
        return None
    rv, ru, lo, hi = res                                      # rays in row-major order
    lo_p = lo + CONTACT_TOL                                   # float32, both paths
    hi_m = hi - CONTACT_TOL
    if _kernels.HAVE_NUMBA:
        first_slot = np.empty(len(rv), dtype=np.int64)
        _kernels.overlap_rays_kernel(f.start, f.end, rv, ru, lo_p, hi_m, first_slot)
        ray_hit = first_slot >= 0
    else:
        s = f.start[rv, ru]                                   # (n, K)
        e = f.end[rv, ru]
        overlap = (s < hi_m[:, np.newaxis]) & (e > lo_p[:, np.newaxis])
        ray_hit = overlap.any(axis=1)
        first_slot = np.argmax(overlap, axis=1)
    if not ray_hit.any():
        return None

    # Report the colliding ray nearest the segment start (the voxel check
    # reports the first hit along the move as well).
    hit_idx = np.nonzero(ray_hit)[0]
    xs = f.origin[f.u_axis] + (ru[hit_idx].astype("f4") + 0.5) * f.h
    ys = f.origin[f.v_axis] + (rv[hit_idx].astype("f4") + 0.5) * f.h
    i = int(np.argmin((xs - start[0]) ** 2 + (ys - start[1]) ** 2))
    n = int(hit_idx[i])
    k = int(first_slot[n])
    z = max(float(lo[n]), float(f.start[rv[n], ru[n], k]))
    world_pt = np.array([xs[i], ys[i], z], dtype="f4")

    kind: Literal["rapid", "shank", "holder"]
    if is_rapid:
        kind = "rapid"
    else:
        # Tip height at the hit: exact for level moves; the mean for others.
        tz = 0.5 * (float(start[2]) + float(end[2]))
        # ≥: the closed form reports the lowest contact exactly, and a
        # contact at total_len is the holder's bottom face.
        kind = "holder" if z - tz >= total_len - 1e-6 else "shank"
    return CollisionHit(point=world_pt, kind=kind)
