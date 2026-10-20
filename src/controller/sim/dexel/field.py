"""
sim/dexel/field.py — DexelField: one set of parallel material rays.

Layout
------
A field with ray direction ``axis`` (0 = X, 1 = Y, 2 = Z) spans the plane of
the two remaining axes ``(u_axis, v_axis)`` = (1, 2) for X, (0, 2) for Y and
(0, 1) for Z. Ray (iv, iu) runs through the cell centre

    u = origin[u_axis] + (iu + 0.5) * h,   v = origin[v_axis] + (iv + 0.5) * h

``start`` / ``end`` are float32 arrays of shape (Nv, Nu, K): up to K
material intervals per ray, sorted by start, non-overlapping. An unused slot
holds ``EMPTY_START`` / ``EMPTY_END`` (start > end, so it never contains a
point and always sorts last).

Dirty tracking
--------------
Like GpuVoxelGrid, ``subtract()`` marks which TILE×TILE ray tiles changed;
GpuDexelGrid.upload_if_dirty() consumes them. Same lock-free snapshot idiom
(see gpu_grid.py's thread-safety note): a single writer thread adds, the GL
thread snapshots with list() and removes only what it uploaded.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# Finite sentinels instead of ±inf so the values survive the trip into an
# RG32F texture unchanged on every driver.
EMPTY_START = np.float32(1.0e30)
EMPTY_END   = np.float32(-1.0e30)

# (u_axis, v_axis) for each ray axis — see module docstring.
PLANE_AXES: dict[int, tuple[int, int]] = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


class DexelField:
    """One ray set of a tri-dexel stock — see module docstring."""

    TILE: int = 32

    def __init__(
        self,
        axis:   int,
        origin: np.ndarray,
        h:      float,
        nu:     int,
        nv:     int,
        k:      int,
    ) -> None:
        self.axis = axis
        self.u_axis, self.v_axis = PLANE_AXES[axis]
        self.origin = np.asarray(origin, dtype="f4")
        self._origin_f = tuple(float(v) for v in self.origin)
        self.h = float(h)
        self.nu, self.nv, self.k = nu, nv, k

        self.start = np.full((nv, nu, k), EMPTY_START, dtype="f4")
        self.end   = np.full((nv, nu, k), EMPTY_END,   dtype="f4")

        self.overflow_count: int = 0
        self._dirty_tiles: set[tuple[int, int]] = set()

    # ── Geometry ──────────────────────────────────────────────────────────────

    def u_centers(self, i0: int = 0, i1: int | None = None) -> np.ndarray:
        i1 = self.nu if i1 is None else i1
        return self.origin[self.u_axis] + (np.arange(i0, i1, dtype="f4") + 0.5) * self.h

    def v_centers(self, i0: int = 0, i1: int | None = None) -> np.ndarray:
        i1 = self.nv if i1 is None else i1
        return self.origin[self.v_axis] + (np.arange(i0, i1, dtype="f4") + 0.5) * self.h

    def index_range(self, lo: float, hi: float, plane_axis: int) -> tuple[int, int]:
        """Ray index range [i0, i1) whose centres may lie in [lo, hi] along
        world axis *plane_axis* (u_axis or v_axis), clipped to the field."""
        # Plain Python arithmetic: called several times per carved segment,
        # where numpy's per-call overhead would dominate.
        n = self.nu if plane_axis == self.u_axis else self.nv
        o = self._origin_f[plane_axis]
        i0 = min(max(math.floor((float(lo) - o) / self.h - 0.5), 0), n)
        i1 = min(max(math.ceil((float(hi) - o) / self.h + 0.5), 0), n)
        return i0, i1

    # ── Material queries ──────────────────────────────────────────────────────

    def contains(self, iv: np.ndarray, iu: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Whether position *t* along ray (iv, iu) is inside material.
        All three arrays broadcast together."""
        s = self.start[iv, iu]          # (..., K)
        e = self.end[iv, iu]
        t = np.asarray(t, dtype="f4")[..., np.newaxis]
        return np.any((s <= t) & (t <= e), axis=-1)

    def total_length(self) -> float:
        """Sum of all interval lengths (mm) — × h² gives the material volume."""
        valid = self.start <= self.end
        return float(np.sum(np.where(valid, self.end - self.start, 0.0), dtype="f8"))

    # ── Modification ──────────────────────────────────────────────────────────

    def fill(self, start: np.ndarray, end: np.ndarray) -> None:
        """Set exactly one interval per ray (slot 0) from (Nv, Nu) arrays;
        rays with start > end become empty. Clears all other slots."""
        self.start[...] = EMPTY_START
        self.end[...]   = EMPTY_END
        valid = start <= end
        self.start[..., 0] = np.where(valid, start, EMPTY_START)
        self.end[..., 0]   = np.where(valid, end,   EMPTY_END)
        self.overflow_count = 0
        self.mark_all_dirty()

    def subtract(
        self,
        iv0: int, iv1: int,
        iu0: int, iu1: int,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> None:
        """Remove [lo, hi] from every ray of the block [iv0:iv1, iu0:iu1].

        *lo*/*hi* have shape (iv1-iv0, iu1-iu0); rays with lo >= hi are left
        untouched. Block-shaped front end of subtract_rays()."""
        cut = lo < hi
        if not cut.any():
            return
        rv, ru = np.nonzero(cut)
        self.subtract_rays(rv + iv0, ru + iu0, lo[cut], hi[cut])

    def subtract_rays(
        self, rv: np.ndarray, ru: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    ) -> None:
        """Remove [lo[i], hi[i]] from ray (rv[i], ru[i]) — the compact form
        sweep.py produces. Fully vectorised over the rays that are cut.

        Each slot (s, e) yields a left piece (s, min(e, lo)) and a right
        piece (max(s, hi), e). Slots are sorted and disjoint, so the valid
        left pieces (s < lo) form a prefix of the slots, the valid right
        pieces (e > hi) a suffix, and every left piece lies before every
        right piece: the result is "left prefix, then right suffix" — one
        gather, no sort. More than K pieces (a split of a full ray) widen
        the last kept slot to the furthest dropped end — keeping material,
        never inventing air — and count as an overflow.
        """
        # Only rays whose material actually overlaps (lo, hi) change — on a
        # real program most swept rays pass through already-cleared air.
        s_full = self.start[rv, ru]                  # (n, K)
        e_full = self.end[rv, ru]
        cut = ((s_full < hi[:, np.newaxis]) & (e_full > lo[:, np.newaxis])).any(axis=1)
        if not cut.any():
            return
        if not cut.all():
            rv, ru, lo, hi = rv[cut], ru[cut], lo[cut], hi[cut]
            s_full, e_full = s_full[cut], e_full[cut]
        lo_ = lo[:, np.newaxis]
        hi_ = hi[:, np.newaxis]

        # Used slots are a prefix, so only the first k = (most used) + 1
        # slots can change — typically 2–3 of the X/Y fields' 8.
        k_full = self.k
        k = min(k_full, int(np.count_nonzero(s_full <= e_full, axis=1).max()) + 1)
        s = s_full[:, :k]
        e = e_full[:, :k]
        used = s <= e
        n_left = np.count_nonzero(s < lo_, axis=1)               # prefix length
        n_right = np.count_nonzero((e > hi_) & used, axis=1)     # suffix length
        n_used = np.count_nonzero(used, axis=1)
        n_valid = n_left + n_right

        # Merged candidate j: left[j] for j < n_left, then the right suffix,
        # i.e. right[n_used - n_right + (j - n_left)] = index k + … into
        # the concatenation [left | right].
        cand_s = np.concatenate([s, np.maximum(s, hi_)], axis=1)
        cand_e = np.concatenate([np.minimum(e, lo_), e], axis=1)
        j = np.arange(2 * k)[np.newaxis, :]
        nl = n_left[:, np.newaxis]
        src = np.where(j < nl, j, k + (n_used - n_right)[:, np.newaxis] + (j - nl))
        src = np.minimum(src, 2 * k - 1)
        filled = j < n_valid[:, np.newaxis]
        merged_s = np.where(filled, np.take_along_axis(cand_s, src, axis=1), EMPTY_START)
        merged_e = np.where(filled, np.take_along_axis(cand_e, src, axis=1), EMPTY_END)

        new_s = merged_s[:, :k]
        new_e = merged_e[:, :k].copy()
        over = n_valid > k_full
        if over.any():
            new_e[over, k - 1] = np.maximum(new_e[over, k - 1], merged_e[over, k:].max(axis=1))
            n_over = int(over.sum())
            if self.overflow_count == 0:
                logger.warning(
                    "Dexel field axis %d: %d ray(s) need more than K=%d intervals "
                    "— keeping material there (raise K for this part).",
                    self.axis, n_over, k_full,
                )
            self.overflow_count += n_over

        self.start[rv, ru, :k] = new_s
        self.end[rv, ru, :k] = new_e
        self._mark_dirty(
            int(rv.min()), int(rv.max()) + 1, int(ru.min()), int(ru.max()) + 1,
        )

    # ── Dirty tracking ────────────────────────────────────────────────────────

    def _mark_dirty(self, iv0: int, iv1: int, iu0: int, iu1: int) -> None:
        tile = DexelField.TILE
        for tv in range(iv0 // tile, (iv1 - 1) // tile + 1):
            for tu in range(iu0 // tile, (iu1 - 1) // tile + 1):
                self._dirty_tiles.add((tv, tu))

    def mark_all_dirty(self) -> None:
        self._mark_dirty(0, self.nv, 0, self.nu)

    @property
    def is_dirty(self) -> bool:
        return bool(self._dirty_tiles)

    def take_dirty_tiles(self) -> list[tuple[int, int]]:
        """Snapshot the dirty tiles and forget them — see module docstring."""
        snapshot = list(self._dirty_tiles)
        self._dirty_tiles -= set(snapshot)
        return snapshot

    def clear_dirty(self) -> None:
        self._dirty_tiles.clear()
