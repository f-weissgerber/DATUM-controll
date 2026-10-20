"""
sim/voxel/prepass.py — full-program collision pre-pass.

Replaces the old per-tick live collision check
(VoxelSimController._check_collision(), removed) with a single, complete,
one-off scan of the WHOLE loaded program that runs in a background thread
(driven by DatumSimWidget, see its docstring on _start_prepass()). The
result is a CollisionPrepassResult — a plain lookup table keyed by path
segment index / G-code line — that on_tick() then reads in O(1) instead of
calling collision.check_segment() itself on every frame.

This is deliberately the same algorithm DatumSimWidget's previous
presim_check_collisions() worker used (fresh, disposable, fully-solid
scratch stock; replay every segment; carve cutting moves into the scratch
copy so later segments see earlier removal, exactly like real playback) —
the only difference is it no longer stops at the first hit: every segment
is checked and every hit recorded, because the result is meant to serve the
*entire* subsequent playback, not just answer "is there a problem at all"
once before a MACHINE start.

Kept independent of Qt/threading concerns on purpose — run_prepass() is a
plain function that only touches numpy/collision.py, so it works the same
whether called from a background thread (the normal case) or synchronously
(e.g. in a test). Cancellation is cooperative: pass a threading.Event and
it is polled periodically; the caller is responsible for not acting on a
None result (returned on cancellation).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from controller.sim.gcode.path_buffer import PathBuffer
from controller.sim.gcode.compiler import ToolChange
from controller.sim.simulation.tool_definition import ToolDefinition
from controller.sim.simulation.tool_holder import HolderProfile
from controller.sim.voxel.carver import VoxelCarver
from controller.sim.voxel.collision import check_segment, CollisionHit
from controller.sim.voxel.gpu_grid import solid_material
from controller.sim.voxel.stock import StockDefinition

# How many segments to process between cooperative-cancellation checks.
# Small enough that an abort (reload / parameter change) is noticed quickly,
# large enough that Event.is_set() polling overhead is negligible.
_ABORT_CHECK_STRIDE = 200


@dataclass
class CollisionPrepassResult:
    """Complete collision table for one fully pre-scanned program.

    hits_by_segment is the primary lookup VoxelSimController.on_tick() uses
    (segment index i -> the CollisionHit found for path segment i -> i+1).
    hit_by_line / hit_lines are line-number-keyed views of the same data,
    for callers that think in G-code lines rather than path segments (the
    Start-button pre-check dialog, a future "highlight affected lines" UI).
    """
    hits_by_segment: dict[int, CollisionHit] = field(default_factory=dict)
    hit_by_line:      dict[int, CollisionHit] = field(default_factory=dict)
    hit_lines:         frozenset[int]         = frozenset()

    @property
    def first_hit(self) -> CollisionHit | None:
        """The hit with the lowest line number, or None if the program is
        clean — what a "does this program have a problem at all" caller
        (the Start-button pre-check) wants."""
        if not self.hits_by_segment:
            return None
        return min(self.hits_by_segment.values(), key=lambda h: h.line_number)


class _CpuMaterialSink:
    """Minimal duck-typed carve target — see presim_check_collisions()'s
    former implementation of the same helper in main_widget.py, which this
    replaces. Implements just what VoxelCarver.carve_segment() calls on its
    grid (.voxel_size/.bbox/.shape/.carve()), writing into a disposable
    numpy array. No GPU, no GpuVoxelGrid, no GL context needed."""

    def __init__(self, mat: np.ndarray, bbox, voxel_size: float) -> None:
        self._mat = mat
        nz, ny, nx = mat.shape
        self.shape = (nx, ny, nz)
        self.bbox = bbox
        self.voxel_size = voxel_size

    def carve(self, ix0, ix1, iy0, iy1, iz0, iz1, mask) -> None:
        self._mat[iz0:iz1, iy0:iy1, ix0:ix1][mask] = 0


def run_prepass(
    path:         PathBuffer,
    tool_changes: list[ToolChange],
    initial_tool: ToolDefinition,
    stock:        StockDefinition,
    get_tool_by_pocket,
    get_holder,
    abort:        threading.Event | None = None,
    backend:      str = "voxel",
) -> CollisionPrepassResult | None:
    """Scan the whole *path* for collisions against a fresh scratch stock.

    Parameters mirror what DatumSimWidget already has on hand when a
    program is loaded — *get_tool_by_pocket*/*get_holder* are callables
    (ToolDatabase lookups) rather than a bound ToolDatabase instance so
    this stays trivially testable with fakes. get_tool_by_pocket, NOT a
    tool_number lookup: each ToolChange.pocket_number is a G-code
    T-address, which selects a magazine pocket directly (see
    gcode/compiler.py's ToolChange docstring) — the caller
    (DatumSimWidget._start_prepass()) passes tool_database.get_tool_by_pocket.

    *backend* selects the scratch stock the scan replays against:
    "voxel" (default) or "dexel" (tri-dexel, see sim/dexel) — so the table
    always matches the geometry the live simulation actually carves.

    Returns None if *abort* is set before the scan completes (the caller
    started a newer pre-pass and this one's result no longer applies — see
    DatumSimWidget._start_prepass()'s generation-counter race guard).
    """
    if backend == "dexel":
        from controller.sim.dexel.carver import DexelCarver
        from controller.sim.dexel.collision import check_segment as dexel_check
        from controller.sim.dexel.stock import solid_dexels

        dexels = solid_dexels(stock)
        carver = DexelCarver(dexels)

        def check(start, end, tool, is_rapid, holder):
            return dexel_check(dexels, start, end, tool, is_rapid, holder)
    else:
        material = solid_material(stock)
        bbox, voxel_size = stock.bbox, stock.voxel_size
        carver = VoxelCarver(_CpuMaterialSink(material, bbox, voxel_size))

        def check(start, end, tool, is_rapid, holder):
            return check_segment(
                material, bbox, voxel_size, start, end, tool, is_rapid, holder,
            )

    pts, feeds, line_ids = path.points, path.feed_rates, path.line_ids
    # Vectorized once, outside the loop — a 1:1 restatement of the per-
    # iteration `feeds[i + 1] == 0.0` this replaces, no semantic change
    # (feeds has exactly one more element than there are segments, so
    # feeds[1:] lines up with segment index i via is_rapid_arr[i]).
    # NOTE: this alone does not make the loop fast — the actual cost here
    # (profiled at ~3.8s for a 1022-line program) is inside check_segment()/
    # carve_segment()'s own geometric sampling (np.clip/np.linspace calls),
    # which cannot be safely batched across segments without changing
    # collision-detection semantics (carve_segment() mutates the shared
    # material array; segment i+1's check legitimately depends on segment
    # i having already carved — see this module's own docstring). That is
    # tracked as a separate, larger follow-up task, not attempted here.
    is_rapid_arr = feeds[1:] == 0.0

    current_tool = initial_tool
    current_holder = get_holder(current_tool.holder_preset) if current_tool else None
    tc_idx = 0

    hits_by_segment: dict[int, CollisionHit] = {}
    hit_by_line: dict[int, CollisionHit] = {}

    n_segments = len(pts) - 1
    for i in range(n_segments):
        if abort is not None and i % _ABORT_CHECK_STRIDE == 0 and abort.is_set():
            return None

        while (
            tc_idx < len(tool_changes)
            and tool_changes[tc_idx].line_index <= line_ids[i]
        ):
            current_tool = get_tool_by_pocket(tool_changes[tc_idx].pocket_number) or current_tool
            current_holder = get_holder(current_tool.holder_preset) if current_tool else None
            tc_idx += 1
        if current_tool is None:
            continue

        is_rapid = bool(is_rapid_arr[i])
        hit = check(pts[i], pts[i + 1], current_tool, is_rapid, current_holder)
        if hit is not None:
            hit.line_number = int(line_ids[i + 1])
            hits_by_segment[i] = hit
            # First hit found for a given line wins — later segments on the
            # same line (an arc tessellated into many samples, say) don't
            # overwrite it; the line is already known to be affected.
            hit_by_line.setdefault(hit.line_number, hit)
        if not is_rapid:
            carver.carve_segment(pts[i], pts[i + 1], current_tool)

    if abort is not None and abort.is_set():
        return None

    return CollisionPrepassResult(
        hits_by_segment=hits_by_segment,
        hit_by_line=hit_by_line,
        hit_lines=frozenset(hit_by_line.keys()),
    )
