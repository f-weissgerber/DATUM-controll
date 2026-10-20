"""
sim/voxel/controller.py — VoxelSimController.

Coordinates the High-Water-Mark path dispatch and CPU carving; GPU texture
upload is the caller's responsibility (``grid.upload_if_dirty()``).

High-Water-Mark algorithm
--------------------------
``_max_s`` / ``_max_idx`` track the furthest arc-length (and corresponding
path index) already carved.  Advancing past it carves new segments;
rewinding does not re-carve existing material (the tool cannot un-remove
chips).

Call ``reset()`` to rebuild the stock from scratch (e.g. after the user
presses the Reset button or changes voxel_size).

Collision detection is diagnostic, not a safety interlock, for this
controller (see check_segment()'s rule in collision.py). A hit NEVER stops
carving or advancing the High-Water-Mark — every segment is processed
regardless of its collision result. ``collision_hit`` reflects only the
LAST segment/call processed (live, not sticky): it goes back to None the
moment a clean segment follows a colliding one, which is what drives "tint
the tool red only while it's actually in the collision" in the UI.
``drain_new_hits()`` separately queues every distinct hit exactly once, for
one-shot handling (logging) regardless of how long a given collision's
visual state stays active.

Pre-pass lookup, not a live geometry check (SIM mode)
-------------------------------------------------------
``on_tick()`` (SIM-mode playback) no longer calls collision.check_segment()
itself — that per-frame, per-segment numpy work was the actual cost behind
the old live check. Instead it looks up each segment's collision status
from a ``prepass.CollisionPrepassResult`` computed ONCE, in a background
thread, for the whole program (see ``set_prepass()`` and
``DatumSimWidget._start_prepass()``). Until a result has been injected via
``set_prepass()``, every segment reads as clear — there is deliberately no
"check live until the pre-pass catches up" fallback, since a fallback would
reintroduce the exact per-tick cost this rewrite removes.

``carve_move()`` (MACHINE mode) is unaffected: it still calls
collision.check_segment() directly, live — MACHINE mode has no tessellated
path to pre-scan (see its own docstring).

Thread safety
-------------
``on_tick()`` mutates ``_max_s``/``_max_idx`` and is NOT safe to call
concurrently from two threads.  ``DatumSimWidget`` calls it either
synchronously from the Qt main thread (small per-tick backlogs) or from a
single background worker thread (large backlogs) — never both at once: a
new call is only ever dispatched once any previous background worker has
fully finished, so at any instant there is exactly one caller. The GPU
upload (``grid.upload_if_dirty()``) must be called with the GL context
current — the caller (``DatumSimWidget._tick`` / ``_drive_voxel_carve`` /
``_force_carve_catchup``) is responsible for ``makeCurrent()``.
``carve_move()`` (MACHINE mode) runs on a separate background thread from
``on_tick()`` (never concurrently with it — MACHINE and SIM mode are
mutually exclusive) but DOES run concurrently with the main/GL thread
draining ``drain_new_hits()`` — that method's list-swap is a single,
GIL-atomic rebind, the same lock-free idiom ``gpu_grid.py``'s
``_dirty_tiles`` snapshot uses, for the same reason.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from controller.sim.voxel.gpu_grid   import GpuVoxelGrid
from controller.sim.voxel.carver     import VoxelCarver
from controller.sim.voxel.collision  import check_segment, CollisionHit
from controller.sim.voxel.stock      import StockDefinition, BoundingBox
from controller.sim.simulation.tool_definition import ToolDefinition
from controller.sim.simulation.tool_holder import HolderProfile

if TYPE_CHECKING:
    from controller.sim.voxel.prepass import CollisionPrepassResult


class VoxelSimController:
    """
    Drives voxel material removal in sync with the simulation player.

    Parameters
    ----------
    grid :
        The shared GPU grid.  Owned externally; this controller only writes.
    carver :
        The carver that modifies ``grid``.
    path_points :
        (N, 3) float32 array of world-space path positions.
    path_arc_lengths :
        (N,) float32 array of cumulative arc lengths for each path point.
    tool :
        Initial tool.  Update with :meth:`update_tool` on T-commands.
    checker :
        Optional live collision check for MACHINE-mode carve_move(),
        ``checker(start, end, tool, is_rapid, holder) -> CollisionHit | None``.
        None (default) = voxel/collision.check_segment() on ``grid.material``;
        the tri-dexel backend passes dexel/collision.check_segment() bound to
        its DexelStock. *grid* and *carver* are duck-typed the same way
        (GpuDexelGrid / DexelCarver), so all High-Water-Mark and pre-pass
        logic here is shared by both backends.
    """

    def __init__(
        self,
        grid:             GpuVoxelGrid,
        carver:           VoxelCarver,
        path_points:      np.ndarray,
        path_arc_lengths: np.ndarray,
        path_feed_rates:  np.ndarray,
        tool:             ToolDefinition,
        path_line_ids:    np.ndarray | None = None,
        checker:          Callable[..., CollisionHit | None] | None = None,
    ) -> None:
        self._grid            = grid
        self._checker         = checker
        self._carver          = carver
        self._pts             = path_points       # (N, 3) float32
        self._arc             = path_arc_lengths  # (N,)   float32
        self._feeds           = path_feed_rates   # (N,)   float32 — 0.0 = G0 rapid
        self._line_ids        = path_line_ids     # (N,)   int32 | None — PathBuffer.line_ids
        self._tool            = tool
        self._holder          = None   # HolderProfile | None — see update_holder()
        self._collision_enabled = True
        self._hit: CollisionHit | None = None       # live state — see class docstring
        self._new_hits: list[CollisionHit] = []      # queue — see drain_new_hits()
        self._prepass = None   # prepass.CollisionPrepassResult | None — see set_prepass()

        # High-Water-Mark: last carved arc-length and corresponding path index
        self._max_s:   float = 0.0
        self._max_idx: int   = 0

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def grid(self) -> GpuVoxelGrid:
        return self._grid

    @property
    def max_s(self) -> float:
        """High-water-mark: arc-length (mm) up to which material has been carved."""
        return self._max_s

    @property
    def collision_hit(self) -> CollisionHit | None:
        """LIVE collision state: the result of the most recently processed
        segment/move, or None if it was clean. Never sticky — see class
        docstring. Use this to drive transient visual feedback (tool
        tinting); use drain_new_hits() to react to each distinct hit once
        (e.g. logging)."""
        return self._hit

    def set_collision_enabled(self, v: bool) -> None:
        self._collision_enabled = v

    def set_prepass(self, result: "CollisionPrepassResult | None") -> None:
        """Inject the currently-valid pre-pass collision table (or None to
        clear it, e.g. while a recompute is in flight). on_tick() reads
        segment hits from this instead of calling check_segment() itself —
        see class docstring. Does not touch collision_hit/drain_new_hits()
        state; call clear_collision() separately if a stale live tint needs
        silencing (DatumSimWidget does both together on invalidation)."""
        self._prepass = result

    def drain_new_hits(self) -> list[CollisionHit]:
        """Pop and return every CollisionHit discovered since the last call
        (each exactly once) — for one-shot handling (logging) independent
        of collision_hit's live on/off state. Safe to call from a different
        thread than the one appending (see class docstring)."""
        hits, self._new_hits = self._new_hits, []
        return hits

    def clear_collision(self) -> None:
        """Reset all live collision state — the current hit and the pending
        queue. Does NOT touch the injected prepass table (set_prepass() is
        the only way to change that) — a Reset re-solidifies the stock but
        the pre-scanned collision geometry against a fresh stock is still
        exactly as valid as before. Called by reset(); safe to call
        standalone too (e.g. to silence a stale live tint)."""
        self._hit = None
        self._new_hits = []

    # ── Collision check (MACHINE-mode carve_move() only — see class
    # docstring: SIM-mode on_tick() reads the prepass table instead) ──────────

    def _check_collision(
        self, start: np.ndarray, end: np.ndarray, tool: ToolDefinition, is_rapid: bool,
    ) -> CollisionHit | None:
        if not self._collision_enabled:
            return None
        if self._checker is not None:
            return self._checker(start, end, tool, is_rapid, self._holder)
        return check_segment(
            self._grid.material, self._grid.bbox, self._grid.voxel_size,
            start, end, tool, is_rapid, self._holder,
        )

    # ── Playback interface ────────────────────────────────────────────────────

    def on_tick(self, current_s: float) -> bool:
        """
        Called every render tick with the player's current arc-length.

        Finds path segments between ``_max_s`` and ``current_s``, checks
        each for a collision (non-blocking — see class docstring) and
        carves it, and returns True if at least one segment was carved (so
        the caller knows to call ``grid.upload_if_dirty()``).

        Rewinding (current_s < _max_s) is a no-op — material is not
        restored.  Call :meth:`reset` to rebuild from scratch.
        """
        if current_s <= self._max_s or len(self._pts) < 2:
            return False

        # Find the path index range [_max_idx, new_idx)
        new_idx = int(np.searchsorted(self._arc, current_s, side="right"))
        new_idx = min(new_idx, len(self._pts) - 1)

        if new_idx <= self._max_idx:
            return False

        carved_any = False
        last_hit: CollisionHit | None = None
        for i in range(self._max_idx, new_idx):
            # Skip G0 rapid moves — feed_rate == 0.0 marks rapid segments.
            # path_buffer.py convention: feed_rates[k] is the feed that
            # ARRIVED at point k (the move from k-1 → k).  So the segment
            # from i → i+1 is rapid when feeds[i+1] == 0.0.
            is_rapid = self._feeds[i + 1] == 0.0

            # O(1) table lookup, not a live geometry check — see class
            # docstring. line_number is already stamped correctly by
            # prepass.run_prepass(); nothing to re-derive here.
            hit = None
            if self._collision_enabled and self._prepass is not None:
                hit = self._prepass.hits_by_segment.get(i)
            if hit is not None:
                self._new_hits.append(hit)
            last_hit = hit

            if not is_rapid:
                self._carver.carve_segment(
                    self._pts[i],
                    self._pts[i + 1],
                    self._tool,
                )
                carved_any = True

        self._hit      = last_hit
        self._max_s    = current_s
        self._max_idx  = new_idx
        return carved_any

    def carve_move(
        self,
        start:       np.ndarray,
        end:         np.ndarray,
        tool:        ToolDefinition,
        feed_rate:   float = 1.0,
        line_number: int = -1,
    ) -> bool:
        """
        Carve one real-machine move (position-based, not arc-length based).

        Used in MACHINE mode where the controller feeds live position updates
        instead of a pre-computed toolpath.  Does not touch ``_max_s`` or
        ``_max_idx`` (those are SIM-mode high-water-mark state).

        Parameters
        ----------
        start, end :
            World-space segment endpoints in mm.
        tool :
            Tool geometry to use for the carved volume.
        feed_rate :
            Feed rate of the move (mm/min).  Pass 0.0 or negative to mark a
            G0 rapid move — still collision-checked (rapids must not touch
            material at all — see collision.check_segment), just not carved.
        line_number :
            Best-known current G-code line, for the reported hit's
            line_number — MACHINE mode has no tessellated path to look this
            up from, so the caller (DatumSimWidget, which already tracks it
            for _check_tool_change()) passes its own last-known value.

        Returns
        -------
        bool
            True if the grid was modified (i.e. ``grid.is_dirty`` is set).
            A collision (see collision_hit/drain_new_hits() after calling)
            never suppresses carving on its own — only feed_rate <= 0
            (a rapid) does, same as before.
        """
        is_rapid = feed_rate <= 0.0

        hit = self._check_collision(start, end, tool, is_rapid)
        if hit is not None:
            hit.line_number = line_number
            self._new_hits.append(hit)
        self._hit = hit

        if is_rapid:
            return False
        self._carver.carve_segment(start, end, tool)
        return self._grid.is_dirty

    def update_tool(self, tool: ToolDefinition) -> None:
        """Called when the simulation passes a T-command."""
        self._tool = tool

    def update_holder(self, holder: "HolderProfile | None") -> None:
        """Called alongside update_tool() when the active tool's assigned
        holder changes (including to None, for a tool with none assigned).
        Factored into collision checks (see collision.check_segment) —
        purely bookkeeping here, no geometry itself."""
        self._holder = holder

    def reset(self) -> None:
        """
        Rebuild the stock from scratch.

        Resets the material grid to fully solid, clears the High-Water-Mark
        and the live collision state, and marks the grid dirty so the next
        upload pushes the full reset. The injected prepass table (see
        set_prepass()) is left untouched — it's still valid against the
        freshly-solid stock.
        """
        self._grid.reset()
        self.clear_collision()
        self._max_s   = 0.0
        self._max_idx = 0
