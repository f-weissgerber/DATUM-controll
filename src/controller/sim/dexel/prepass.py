"""
sim/dexel/prepass.py — the collision pre-pass for the tri-dexel backend.

Same contract and same result as voxel/prepass.run_prepass() (which
delegates here for backend="dexel"): replay the whole program against a
fresh scratch stock, check every segment BEFORE carving it, record every
hit per segment and per G-code line.

Speed-up over the plain per-segment loop, result unchanged
------------------------------------------------------------
Collinear runs (see grouping.py — the ~10 pieces PathBuffer cuts each G1
line into) are checked as one move first. That check is a conservative
filter (grouping.py's docstring) when check_segment() takes its closed-form
path (collision.is_closed_form()): a clear run is carved as one move; a run
that hits — or any run whose tool needs the sampled path — is replayed piece
by piece exactly like the plain loop, so hits_by_segment is identical.
"""
from __future__ import annotations

import threading

from controller.sim.dexel.carver import DexelCarver
from controller.sim.dexel.collision import check_segment, is_closed_form
from controller.sim.dexel.grouping import run_ends
from controller.sim.dexel.stock import solid_dexels

_ABORT_CHECK_STRIDE = 200


def run_dexel_prepass(
    path, tool_changes, initial_tool, stock, get_tool_by_pocket, get_holder,
    abort: threading.Event | None = None,
    merge_runs: bool = True,
):
    """See module docstring; returns (hits_by_segment, hit_by_line) or
    None if *abort* was set."""
    import numpy as np

    dexels = solid_dexels(stock)
    carver = DexelCarver(dexels)
    pts, feeds, line_ids = path.points, path.feed_rates, path.line_ids
    n_seg = len(pts) - 1
    is_rapid_arr = feeds[1:] == 0.0

    # Tool + holder per segment (same tool-change rule as the voxel loop),
    # resolved up front so runs can be kept within one tool.
    tools: list = []
    holders: list = []
    tool_idx = np.zeros(n_seg, dtype=np.int64)
    current_tool = initial_tool
    current_holder = get_holder(current_tool.holder_preset) if current_tool else None
    tc_idx = 0
    for i in range(n_seg):
        changed = False
        while tc_idx < len(tool_changes) and tool_changes[tc_idx].line_index <= line_ids[i]:
            current_tool = get_tool_by_pocket(tool_changes[tc_idx].pocket_number) or current_tool
            current_holder = get_holder(current_tool.holder_preset) if current_tool else None
            tc_idx += 1
            changed = True
        if changed or not tools:
            tools.append(current_tool)
            holders.append(current_holder)
        tool_idx[i] = len(tools) - 1

    if merge_runs:
        ends = run_ends(pts, feeds, line_ids, seg_key=tool_idx)
    else:
        ends = np.arange(1, n_seg + 1)

    hits_by_segment: dict = {}
    hit_by_line: dict = {}

    def piecewise(i0: int, i1: int) -> None:
        tool, holder = tools[tool_idx[i0]], holders[tool_idx[i0]]
        for i in range(i0, i1):
            is_rapid = bool(is_rapid_arr[i])
            hit = check_segment(dexels, pts[i], pts[i + 1], tool, is_rapid, holder)
            if hit is not None:
                hit.line_number = int(line_ids[i + 1])
                hits_by_segment[i] = hit
                hit_by_line.setdefault(hit.line_number, hit)
            if not is_rapid:
                carver.carve_segment(pts[i], pts[i + 1], tool)

    i = 0
    next_abort_check = 0
    while i < n_seg:
        if abort is not None and i >= next_abort_check:
            if abort.is_set():
                return None
            next_abort_check = i + _ABORT_CHECK_STRIDE
        j = int(ends[i])
        tool, holder = tools[tool_idx[i]], holders[tool_idx[i]]
        if tool is None:
            i = j
            continue
        is_rapid = bool(is_rapid_arr[i])
        if j - i == 1 or not is_closed_form(tool, holder, is_rapid):
            piecewise(i, j)
        elif check_segment(dexels, pts[i], pts[j], tool, is_rapid, holder) is None:
            if not is_rapid:
                carver.carve_segment(pts[i], pts[j], tool)
        else:
            piecewise(i, j)
        i = j

    if abort is not None and abort.is_set():
        return None
    return hits_by_segment, hit_by_line
