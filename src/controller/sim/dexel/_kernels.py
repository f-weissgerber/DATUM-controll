"""
sim/dexel/_kernels.py — optional numba kernels for the dexel hot loops.

numba is an OPTIONAL dependency (``pip install .[fast]``). Without it — or
with the environment variable ``DATUM_DEXEL_NO_NUMBA=1`` — HAVE_NUMBA is
False and every caller keeps its numpy path unchanged.

The kernels compute bit-identical results to those numpy paths: they only
copy, compare and take min/max of the same float32 values (no arithmetic
whose rounding could differ). tests/test_dexel.py checks this against the
numpy path and the plain Python reference.

All kernels are ``nogil`` so the collision pre-pass (background thread)
and SIM carving no longer serialise on the GIL while inside them.
"""
from __future__ import annotations

import os

import numpy as np

HAVE_NUMBA = False
if not os.environ.get("DATUM_DEXEL_NO_NUMBA"):
    try:
        from numba import njit, prange

        HAVE_NUMBA = True
    except ImportError:          # optional dependency missing → numpy paths
        pass

if HAVE_NUMBA:

    @njit(cache=True, nogil=True, parallel=True)
    def subtract_rays_kernel(start, end, rv, ru, lo, hi, empty_s, empty_e, flags):
        """In-place DexelField.subtract_rays() — see its docstring for the
        algorithm (left prefix, then right suffix; overflow widens the last
        slot). flags[i] := 0 untouched, 1 changed, 2 changed + overflow."""
        k_full = start.shape[2]
        for i in prange(rv.shape[0]):
            r = rv[i]
            c = ru[i]
            a = lo[i]
            b = hi[i]
            touch = False
            for k in range(k_full):
                if start[r, c, k] < b and end[r, c, k] > a:
                    touch = True
                    break
            if not touch:
                flags[i] = 0
                continue
            out_s = np.empty(2 * k_full, dtype=start.dtype)
            out_e = np.empty(2 * k_full, dtype=start.dtype)
            n = 0
            for k in range(k_full):                 # left pieces (prefix)
                s = start[r, c, k]
                e = end[r, c, k]
                if s < a:
                    out_s[n] = s
                    out_e[n] = min(e, a)
                    n += 1
            for k in range(k_full):                 # right pieces (suffix)
                s = start[r, c, k]
                e = end[r, c, k]
                if s <= e and e > b:
                    out_s[n] = max(s, b)
                    out_e[n] = e
                    n += 1
            flag = 1
            if n > k_full:
                last = out_e[k_full - 1]
                for j in range(k_full, n):
                    if out_e[j] > last:
                        last = out_e[j]
                out_e[k_full - 1] = last
                flag = 2
            for k in range(k_full):
                if k < n:
                    start[r, c, k] = out_s[k]
                    end[r, c, k] = out_e[k]
                else:
                    start[r, c, k] = empty_s
                    end[r, c, k] = empty_e
            flags[i] = flag

    @njit(cache=True, nogil=True, parallel=True)
    def overlap_rays_kernel(start, end, rv, ru, lo_p, hi_m, first_slot):
        """Closed-form collision test per ray: first slot k with
        start < hi_m and end > lo_p (thresholds precomputed in float32 by
        the caller), or -1. Same comparisons as the numpy path."""
        k_full = start.shape[2]
        for i in prange(rv.shape[0]):
            r = rv[i]
            c = ru[i]
            first_slot[i] = -1
            for k in range(k_full):
                if start[r, c, k] < hi_m[i] and end[r, c, k] > lo_p[i]:
                    first_slot[i] = k
                    break

    # ── Sweeps (sweep.py) ─────────────────────────────────────────────────────
    # Mirrors of sweep_z() / the level branch of sweep_horizontal(), operation
    # for operation (same float64/float32 steps as numpy under NEP 50, same
    # np.maximum/np.minimum tie rules), writing per-ray lo/hi into a block;
    # the caller compacts it exactly like sweep._compact(). Ray centres, the
    # ProfileInverse table, sample fractions and row radii come in from numpy.

    @njit(cache=True, inline="always")
    def _np_max(a, b):
        return a if a >= b else b

    @njit(cache=True, inline="always")
    def _np_min(a, b):
        return a if a <= b else b

    @njit(cache=True)
    def _has_material(start, end, r, c, lo, hi):
        for k in range(start.shape[2]):
            if np.float64(start[r, c, k]) < hi and np.float64(end[r, c, k]) > lo:
                return True
        return False

    @njit(cache=True)
    def _zmin(d, r_tab, hz_tab):
        """ProfileInverse.zmin() for one float32 d."""
        n = r_tab.shape[0]
        lo_i = 0
        hi_i = n
        while lo_i < hi_i:                       # searchsorted(side="left")
            mid = (lo_i + hi_i) // 2
            if r_tab[mid] < d:
                lo_i = mid + 1
            else:
                hi_i = mid
        i = lo_i
        if i == 0:
            return np.float32(0.0)
        if i >= n:
            return np.float32(np.inf)
        r0 = r_tab[i - 1]
        r1 = r_tab[i]
        span = r1 - r0 if r1 > r0 else np.float32(1.0)
        w = (d - r0) / span
        w = _np_min(_np_max(w, np.float32(0.0)), np.float32(1.0))
        return hz_tab[i - 1] + w * (hz_tab[i] - hz_tab[i - 1])

    @njit(cache=True, nogil=True, parallel=True)
    def sweep_z_kernel(start, end, iv0, iu0, xs, ys, a, b, rmax, r_tab, hz_tab,
                       z_lo, z_hi, mat_lo, mat_hi, exact, frac, eps, no_lo, no_hi,
                       out_lo, out_hi):
        lv, lu = out_lo.shape
        v0 = b[0] - a[0]
        v1 = b[1] - a[1]
        v2 = b[2] - a[2]
        A = v0 * v0 + v1 * v1
        z_lo32 = np.float32(z_lo)
        z_hi32 = np.float32(z_hi)
        n_s = 3 if exact else 3 + frac.shape[0]
        for jv in prange(lv):
            qy = np.float64(ys[jv]) - a[1]
            for ju in range(lu):
                out_lo[jv, ju] = no_lo
                out_hi[jv, ju] = no_hi
                qx = np.float64(xs[ju]) - a[0]
                qv = qx * v0 + qy * v1
                C = qx * qx + qy * qy - rmax * rmax
                if A < eps:
                    if not (C <= 0.0):
                        continue
                    t0 = 0.0
                    t1 = 1.0
                    t_star = 0.0
                else:
                    disc = qv * qv - A * C
                    if not (disc >= 0.0):
                        continue
                    sq = np.sqrt(_np_max(disc, 0.0))
                    t0 = _np_max((qv - sq) / A, 0.0)
                    t1 = _np_min((qv + sq) / A, 1.0)
                    t_star = qv / A
                if not (t0 <= t1):
                    continue
                if not _has_material(start, end, iv0 + jv, iu0 + ju, mat_lo, mat_hi):
                    continue
                lo = no_lo
                hi = no_hi
                for s in range(n_s):
                    if s == 0:
                        t = t0
                    elif s == 1:
                        t = t1
                    elif s == 2:
                        t = _np_min(_np_max(t_star, t0), t1)
                    else:
                        t = t0 + frac[s - 3] * (t1 - t0)
                    dx = qx - t * v0
                    dy = qy - t * v1
                    d = np.float32(np.sqrt(dx * dx + dy * dy))
                    tz = np.float32(a[2] + t * v2)
                    zrel = _np_max(_zmin(d, r_tab, hz_tab), z_lo32)
                    if zrel <= z_hi32:
                        lo = _np_min(lo, tz + zrel)
                        hi = _np_max(hi, tz + z_hi32)
                out_lo[jv, ju] = lo
                out_hi[jv, ju] = hi

    @njit(cache=True, nogil=True, parallel=True)
    def sweep_level_kernel(start, end, iv0, iu0, us, zs, a, b, ax, ua, r_max,
                           z_lo, z_hi, mat_lo, mat_hi, r_row, has_band, n_ax, n_ua,
                           eps, no_lo, no_hi, out_lo, out_hi):
        lv, lu = out_lo.shape
        v_ax = b[ax] - a[ax]
        v_ua = b[ua] - a[ua]
        a_ax = a[ax]
        a_ua = a[ua]
        a2 = a[2]
        for jv in prange(lv):
            z = np.float64(zs[jv])
            # level: tz range is all-or-nothing (see sweep._linear_range)
            z_in = (z - z_hi <= a2) and (a2 <= z - z_lo)
            tz0 = -np.inf if z_in else np.inf
            tz1 = np.inf if z_in else -np.inf
            r = r_row[jv]
            for ju in range(lu):
                out_lo[jv, ju] = no_lo
                out_hi[jv, ju] = no_hi
                u = np.float64(us[ju])
                lo_u = u - r_max
                hi_u = u + r_max
                if abs(v_ua) < eps:
                    inside = (lo_u <= a_ua) and (a_ua <= hi_u)
                    tu0 = -np.inf if inside else np.inf
                    tu1 = np.inf if inside else -np.inf
                else:
                    ta = (lo_u - a_ua) / v_ua
                    tb = (hi_u - a_ua) / v_ua
                    tu0 = _np_min(ta, tb)
                    tu1 = _np_max(ta, tb)
                t0 = _np_max(_np_max(tu0, tz0), 0.0)
                t1 = _np_min(_np_min(tu1, tz1), 1.0)
                if not (t0 <= t1):
                    continue
                if not _has_material(start, end, iv0 + jv, iu0 + ju, mat_lo, mat_hi):
                    continue
                lo = np.inf
                hi = -np.inf
                for p in range(2):
                    p_ax = a[ax] if p == 0 else b[ax]
                    p_ua = a[ua] if p == 0 else b[ua]
                    du = u - p_ua
                    q = r * r - du * du
                    if q >= 0.0:
                        half = np.sqrt(_np_max(q, 0.0))
                        lo = _np_min(lo, p_ax - half)
                        hi = _np_max(hi, p_ax + half)
                if has_band:
                    for g in range(2):
                        sgn = -1.0 if g == 0 else 1.0
                        tt = ((u - a_ua) - (sgn * r) * n_ua) / v_ua
                        x = (a_ax + tt * v_ax) + (sgn * r) * n_ax
                        if tt >= 0.0 and tt <= 1.0:
                            lo = _np_min(lo, x)
                            hi = _np_max(hi, x)
                out_lo[jv, ju] = np.float32(lo)
                out_hi[jv, ju] = np.float32(hi)
