"""
sim/dexel/sweep.py — intersection of a tool sweep with each ray set.

Per ray, not per stamp
----------------------
The tool tip moves along p(t) = start + t·(end − start), t ∈ [0, 1]. For
every ray this module first solves, in closed form, the sub-range [t0, t1]
in which the tool can reach that ray at all (a quadratic for Z rays, two
linear bounds for X/Y rays), then evaluates the tool's intersection with the
ray only at a few parameters inside it: t0, t1, the point of closest lateral
approach t*, and N_SAMPLES evenly spaced ones — except where a closed form
exists (level moves, see sweep_horizontal(); level/plunge Z rays). Work is O(rays × samples),
independent of segment length — a stamp-based sweep (voxel/carver.py) costs
O(stamps × rays), i.e. grows quadratically with segment length at fine
resolution.

The per-sample intersections are reduced to ONE interval [min lo, max hi]
per ray. A convex tool moved along a straight line sweeps a convex volume,
and a line meets a convex volume in a single interval, so that union is
correct — every ToolType profile is convex (monotone from the tip; see the
package docstring for the limit). The extremes are exact where they sit at
t0/t1/t* (straight passes, plunges, flat-bottom tools — the common cases);
elsewhere they come from the samples (removed volume within ~0.1 % for
N_SAMPLES=16 in tests/test_dexel.py's ramp cases).

Z rays need the inverse profile ("lowest height at which the tool is at
least d wide"); ProfileInverse tabulates it once per call.
"""
from __future__ import annotations

import numpy as np

from controller.sim.dexel import _kernels
from controller.sim.dexel.field import DexelField
from controller.sim.simulation.tool_definition import ToolDefinition

# Evenly spaced evaluation points inside each ray's reachable t-range, on
# top of its two ends and the closest-approach point.
N_SAMPLES = 16

# Sample fractions for 3-D moves (the numba kernels get the same array).
_FRAC = np.linspace(0.0, 1.0, N_SAMPLES, dtype="f8")

_NO_HIT_LO = np.float32(1.0e30)
_NO_HIT_HI = np.float32(-1.0e30)
_EPS = 1e-12


def cutting_length(tool: ToolDefinition) -> float:
    """Effective cutting height above the tip — same fallbacks as VoxelCarver."""
    cut_len = tool.cutting_length
    if cut_len <= 0.0:
        cut_len = tool.total_length
    if cut_len <= 0.0:
        cut_len = 999.0
    return cut_len


def stamp_points(
    start: np.ndarray, end: np.ndarray, spacing: float, z_window: float,
) -> np.ndarray:
    """(n, 3) stamp positions along start→end — VoxelCarver's sampling rule:
    adjacent stamps ≤ spacing/2 apart in XY and ≤ z_window apart in Z.
    Used by collision.py, which evaluates holder profiles per stamp."""
    seg_vec = (np.asarray(end, "f4") - np.asarray(start, "f4")).astype("f4")
    xy_len = float(np.linalg.norm(seg_vec[:2]))
    z_len = float(abs(seg_vec[2]))
    n_xy = int(np.ceil(xy_len / (spacing * 0.5))) + 1
    n_z = int(np.ceil(z_len / max(z_window, spacing))) + 1
    ts = np.linspace(0.0, 1.0, max(2, n_xy, n_z), dtype="f4")
    return np.asarray(start, "f4") + ts[:, np.newaxis] * seg_vec


class ProfileInverse:
    """zmin(d): lowest height above the tip where the tool radius reaches d
    (+inf beyond the largest radius). Tabulated on *n* heights over
    [0, height] and linearly interpolated; the table is the running maximum
    of the profile, i.e. the monotone hull (exact for all ToolTypes).

    The table only spans the height at which the profile saturates (a ball's
    radius, a drill's tip cone, …) — not the whole cutting length, which
    may be the 999 mm fallback — so its step stays far below the ray spacing.
    """

    def __init__(self, radius_at_array, height: float, n: int = 2048) -> None:
        coarse = np.linspace(0.0, height, 4 * n, dtype="f4")
        r_coarse = np.maximum.accumulate(radius_at_array(coarse).astype("f4"))
        sat = int(np.searchsorted(r_coarse, r_coarse[-1] - 1e-6, side="left"))
        top = float(coarse[min(sat + 1, len(coarse) - 1)])
        self.hz = np.linspace(0.0, max(top, 1e-3), n, dtype="f4")
        self.r = np.maximum.accumulate(radius_at_array(self.hz).astype("f4"))
        self.r_max = float(r_coarse[-1])

    def zmin(self, d: np.ndarray) -> np.ndarray:
        r, hz = self.r, self.hz
        i = np.searchsorted(r, d, side="left")
        i_c = np.clip(i, 1, len(r) - 1)
        r0, r1 = r[i_c - 1], r[i_c]
        h0, h1 = hz[i_c - 1], hz[i_c]
        span = np.where(r1 > r0, r1 - r0, np.float32(1.0))
        z = h0 + np.clip((d - r0) / span, 0.0, 1.0) * (h1 - h0)
        z = np.where(i == 0, np.float32(0.0), z)
        return np.where(i >= len(r), np.float32(np.inf), z).astype("f4")


def _sample_ts(
    t0: np.ndarray, t1: np.ndarray, t_star: np.ndarray, exact: bool,
) -> np.ndarray:
    """(n, S) evaluation parameters per ray inside [t0, t1]: the ends and the
    closest-approach point, plus N_SAMPLES evenly spaced ones unless *exact*
    (the extremes are known to lie on those three — see callers)."""
    cols = [t0[:, np.newaxis], t1[:, np.newaxis], np.clip(t_star, t0, t1)[:, np.newaxis]]
    if not exact:
        frac = np.linspace(0.0, 1.0, N_SAMPLES, dtype="f8")
        cols.append(t0[:, np.newaxis] + frac * (t1 - t0)[:, np.newaxis])
    return np.concatenate(cols, axis=-1)


def _has_material_in(
    field: DexelField, iv0: int, iv1: int, iu0: int, iu1: int, lo: float, hi: float,
) -> np.ndarray:
    """Per ray of the block: does any material interval overlap (lo, hi)?
    Rays that fail can't be cut by this segment and are skipped before the
    (much costlier) per-sample evaluation — the common case for finishing
    passes, pocket walls left and right of an already cleared span, and
    repeated passes through air. Empty slots (start > end) never overlap."""
    s = field.start[iv0:iv1, iu0:iu1]
    e = field.end[iv0:iv1, iu0:iu1]
    return ((s < hi) & (e > lo)).any(axis=-1)


RaySweep = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _compact(rv: np.ndarray, ru: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> RaySweep | None:
    """(rv, ru, lo, hi) of the rays actually hit (lo < hi): absolute ray
    indices + float32 interval — None if none. A compact list instead of
    block-shaped arrays: the carver and the collision check only ever touch
    these rays."""
    lo = lo.astype("f4")
    hi = hi.astype("f4")
    ok = lo < hi
    if not ok.any():
        return None
    return rv[ok], ru[ok], lo[ok], hi[ok]


def _compact_block(iv0: int, iu0: int, lo: np.ndarray, hi: np.ndarray) -> RaySweep | None:
    """_compact() for the numba kernels' block output (row-major, like the
    numpy paths' np.nonzero order)."""
    rv, ru = np.nonzero(lo < hi)
    if len(rv) == 0:
        return None
    return rv + iv0, ru + iu0, lo[rv, ru], hi[rv, ru]


def _linear_range(p0: float, dp: float, lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """t-range in which p0 + t·dp ∈ [lo, hi] (arrays allowed), unclipped."""
    if abs(dp) < _EPS:
        inside = (lo <= p0) & (p0 <= hi)
        return (np.where(inside, -np.inf, np.inf), np.where(inside, np.inf, -np.inf))
    ta = (lo - p0) / dp
    tb = (hi - p0) / dp
    return np.minimum(ta, tb), np.maximum(ta, tb)


def sweep_z(
    field:    DexelField,
    start:    np.ndarray,
    end:      np.ndarray,
    inv:      ProfileInverse,
    z_lo_rel: float,
    z_hi_rel: float,
) -> RaySweep | None:
    """Swept interval along every Z ray the tool reaches.

    At XY distance d from its axis the tool occupies heights
    [tip + max(zmin(d), z_lo_rel), tip + z_hi_rel]. Reachable range: the t
    with |q − p_xy(t)| ≤ r_max (quadratic in t). Returns the hit rays as
    (rv, ru, lo, hi) (see _compact()) or None.
    """
    a = np.asarray(start, "f8")
    b = np.asarray(end, "f8")
    v = b - a
    rmax = inv.r_max
    iu0, iu1 = field.index_range(min(a[0], b[0]) - rmax, max(a[0], b[0]) + rmax, 0)
    iv0, iv1 = field.index_range(min(a[1], b[1]) - rmax, max(a[1], b[1]) + rmax, 1)
    if iu0 >= iu1 or iv0 >= iv1:
        return None
    if _kernels.HAVE_NUMBA:
        out_lo = np.empty((iv1 - iv0, iu1 - iu0), dtype="f4")
        out_hi = np.empty_like(out_lo)
        exact = abs(v[2]) < _EPS or (v[0] * v[0] + v[1] * v[1]) < _EPS
        _kernels.sweep_z_kernel(
            field.start, field.end, iv0, iu0,
            field.u_centers(iu0, iu1), field.v_centers(iv0, iv1), a, b, float(rmax),
            inv.r, inv.hz, float(z_lo_rel), float(z_hi_rel),
            float(min(a[2], b[2]) + z_lo_rel), float(max(a[2], b[2]) + z_hi_rel),
            exact, _FRAC, _EPS, _NO_HIT_LO, _NO_HIT_HI, out_lo, out_hi,
        )
        return _compact_block(iv0, iu0, out_lo, out_hi)

    qx = field.u_centers(iu0, iu1).astype("f8")[np.newaxis, :] - a[0]   # (1, lu)
    qy = field.v_centers(iv0, iv1).astype("f8")[:, np.newaxis] - a[1]   # (lv, 1)

    # |q − t·v_xy|² ≤ R²  →  A t² + B t + C ≤ 0
    A = v[0] * v[0] + v[1] * v[1]
    qv = qx * v[0] + qy * v[1]
    C = qx * qx + qy * qy - rmax * rmax
    if A < _EPS:
        reach = C <= 0.0
        t0 = np.where(reach, 0.0, 1.0)
        t1 = np.where(reach, 1.0, 0.0)
        t_star = np.zeros_like(C)
    else:
        disc = qv * qv - A * C
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = np.where(disc >= 0.0, np.maximum((qv - sq) / A, 0.0), 1.0)
        t1 = np.where(disc >= 0.0, np.minimum((qv + sq) / A, 1.0), 0.0)
        t_star = qv / A
    t0, t1 = np.broadcast_arrays(t0, t1)
    reach = (t0 <= t1) & _has_material_in(
        field, iv0, iv1, iu0, iu1,
        min(a[2], b[2]) + z_lo_rel, max(a[2], b[2]) + z_hi_rel,
    )
    if not reach.any():
        return None
    # Only reachable rays are evaluated (a diagonal segment reaches a small
    # fraction of its bounding box), then scattered back into the block.
    iv, iu = np.nonzero(reach)
    t_star = np.broadcast_to(t_star, reach.shape)[iv, iu]
    # tip height + zmin(d(t)) is convex in t (zmin is convex and
    # non-decreasing for every profile, d(t) is convex), so with a level
    # move its minimum sits exactly at t*, with a plunge (d constant) at
    # an end — only a 3-D move needs the extra samples.
    exact = abs(v[2]) < _EPS or A < _EPS
    ts = _sample_ts(t0[iv, iu], t1[iv, iu], t_star, exact)             # (n, S)
    qx_r = np.broadcast_to(qx, reach.shape)[iv, iu][:, np.newaxis]
    qy_r = np.broadcast_to(qy, reach.shape)[iv, iu][:, np.newaxis]
    dx = qx_r - ts * v[0]
    dy = qy_r - ts * v[1]
    d = np.sqrt(dx * dx + dy * dy).astype("f4")
    tz = (a[2] + ts * v[2]).astype("f4")
    zrel = np.maximum(inv.zmin(d), np.float32(z_lo_rel))
    hit = zrel <= z_hi_rel
    lo = np.where(hit, tz + zrel, _NO_HIT_LO).min(axis=-1)
    hi = np.where(hit, tz + np.float32(z_hi_rel), _NO_HIT_HI).max(axis=-1)
    return _compact(iv + iv0, iu + iu0, lo, hi)


def sweep_horizontal(
    field:    DexelField,
    start:    np.ndarray,
    end:      np.ndarray,
    radius_at_array,
    r_max:    float,
    z_lo_rel: float,
    z_hi_rel: float,
) -> RaySweep | None:
    """Swept interval along every X ray (field.axis 0) or Y ray (axis 1).

    Ray (iv, iu) sits at height z = v and at the other horizontal coordinate
    u. The tool reaches it while |u − p_u(t)| ≤ r_max and
    hz = z − p_z(t) ∈ [z_lo_rel, z_hi_rel] (both linear in t); there it
    covers p_axis(t) ± sqrt(r(hz)² − du²).
    """
    ax, ua = field.axis, field.u_axis
    a = np.asarray(start, "f8")
    b = np.asarray(end, "f8")
    v = b - a
    iu0, iu1 = field.index_range(min(a[ua], b[ua]) - r_max, max(a[ua], b[ua]) + r_max, ua)
    iv0, iv1 = field.index_range(
        min(a[2], b[2]) + z_lo_rel, max(a[2], b[2]) + z_hi_rel, 2,
    )
    if iu0 >= iu1 or iv0 >= iv1:
        return None
    if _kernels.HAVE_NUMBA and abs(v[2]) < _EPS:
        zs_f4 = field.v_centers(iv0, iv1)
        r_row = radius_at_array((zs_f4.astype("f8") - a[2]).astype("f4")).astype("f8")
        has_band = abs(v[ua]) >= _EPS
        n_ax = n_ua = 0.0
        if has_band:
            l_xy = float(np.hypot(v[ax], v[ua]))
            n_ax, n_ua = float(-v[ua] / l_xy), float(v[ax] / l_xy)
        out_lo = np.empty((iv1 - iv0, iu1 - iu0), dtype="f4")
        out_hi = np.empty_like(out_lo)
        _kernels.sweep_level_kernel(
            field.start, field.end, iv0, iu0, field.u_centers(iu0, iu1), zs_f4, a, b,
            ax, ua, float(r_max), float(z_lo_rel), float(z_hi_rel),
            float(min(a[ax], b[ax]) - r_max), float(max(a[ax], b[ax]) + r_max),
            r_row, has_band, n_ax, n_ua, _EPS, _NO_HIT_LO, _NO_HIT_HI, out_lo, out_hi,
        )
        return _compact_block(iv0, iu0, out_lo, out_hi)

    us = field.u_centers(iu0, iu1).astype("f8")[np.newaxis, :]    # (1, lu)
    zs = field.v_centers(iv0, iv1).astype("f8")[:, np.newaxis]    # (lv, 1)

    tu0, tu1 = _linear_range(a[ua], v[ua], us - r_max, us + r_max)       # (1, lu)
    tz0, tz1 = _linear_range(a[2], v[2], zs - z_hi_rel, zs - z_lo_rel)   # (lv, 1)
    t0 = np.maximum(np.maximum(tu0, tz0), 0.0)
    t1 = np.minimum(np.minimum(tu1, tz1), 1.0)
    reach = (t0 <= t1) & _has_material_in(
        field, iv0, iv1, iu0, iu1, min(a[ax], b[ax]) - r_max, max(a[ax], b[ax]) + r_max,
    )
    if not reach.any():
        return None
    iv, iu = np.nonzero(reach)

    if abs(v[2]) < _EPS:
        # Level move: at a ray's height the tool is a disc of fixed radius
        # r gliding along a straight XY line, sweeping a "stadium". A line
        # meets that convex shape in one interval whose ends lie on its
        # boundary — the two end circles or the two band edges — so the
        # exact interval is the min/max over those intersection points.
        r_row = radius_at_array((zs[:, 0] - a[2]).astype("f4")).astype("f8")   # (lv,)
        r = r_row[iv]
        u = us[0, iu]
        cands_lo = []
        cands_hi = []
        for p_ax, p_ua in ((a[ax], a[ua]), (b[ax], b[ua])):
            q = r * r - (u - p_ua) ** 2
            half = np.sqrt(np.maximum(q, 0.0))
            cands_lo.append(np.where(q >= 0.0, p_ax - half, np.inf))
            cands_hi.append(np.where(q >= 0.0, p_ax + half, -np.inf))
        if abs(v[ua]) >= _EPS:
            l_xy = float(np.hypot(v[ax], v[ua]))
            n_ax, n_ua = -v[ua] / l_xy, v[ax] / l_xy            # unit normal in (ax, ua)
            for sgn in (-1.0, 1.0):
                tt = (u - a[ua] - sgn * r * n_ua) / v[ua]
                x = a[ax] + tt * v[ax] + sgn * r * n_ax
                ok = (tt >= 0.0) & (tt <= 1.0)
                cands_lo.append(np.where(ok, x, np.inf))
                cands_hi.append(np.where(ok, x, -np.inf))
        lo = np.min(cands_lo, axis=0)
        hi = np.max(cands_hi, axis=0)
    else:
        if abs(v[ua]) < _EPS:
            t_star = np.zeros(len(iv))
        else:
            t_star = (us[0, iu] - a[ua]) / v[ua]
        ts = _sample_ts(t0[iv, iu], t1[iv, iu], t_star, False)          # (n, S)
        hz = (zs[iv, 0][:, np.newaxis] - (a[2] + ts * v[2])).astype("f4")
        in_z = (hz >= z_lo_rel - 1e-4) & (hz <= z_hi_rel + 1e-4)
        r = radius_at_array(np.clip(hz, z_lo_rel, z_hi_rel).reshape(-1)).reshape(hz.shape)
        du = us[0, iu][:, np.newaxis] - (a[ua] + ts * v[ua])
        q = r.astype("f8") ** 2 - du * du
        hit = (q >= 0.0) & in_z
        half = np.sqrt(np.maximum(q, 0.0))
        t_ax = a[ax] + ts * v[ax]
        lo = np.where(hit, t_ax - half, _NO_HIT_LO).min(axis=-1)
        hi = np.where(hit, t_ax + half, _NO_HIT_HI).max(axis=-1)
    return _compact(iv + iv0, iu + iu0, lo, hi)
