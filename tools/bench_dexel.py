"""
tools/bench_dexel.py — benchmark the voxel vs. tri-dexel simulation backends.

Runs on a real program (default: workpieces/Gcode.cnc) and reports, per
resolution and backend:

  prepass   — full-program collision pre-pass (voxel/prepass.run_prepass)
  carve     — carving every cutting move into a fresh stock (all ray sets
              for dexel), i.e. what SIM playback does over the whole program
  stock MB  — CPU memory of the solid stock
  hits      — number of colliding G-code lines (+ equality with --ref)

Usage (from the repo root):

  PYTHONPATH=src python tools/bench_dexel.py --res 0.5 0.1 --backends voxel dexel
  PYTHONPATH=src python tools/bench_dexel.py --res 0.1 --backends dexel --profile
  PYTHONPATH=src python tools/bench_dexel.py --segments        # length statistics
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np


def _load(path: str):
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.instance() or QCoreApplication([])   # AppSettings/ToolDatabase
    from controller.sim.gcode.compiler import GCodeCompiler

    return GCodeCompiler().load_file(path)


def _tools():
    from controller.persistence.tool_db import ToolDatabase
    from controller.sim.simulation.tool_database import get_tool_by_pocket

    return get_tool_by_pocket, ToolDatabase.instance().get_holder


def segment_stats(prog) -> None:
    pts, feeds, lines = prog.path.points, prog.path.feed_rates, prog.path.line_ids
    seg = np.diff(pts, axis=0)
    length = np.linalg.norm(seg, axis=1)
    cutting = feeds[1:] > 0.0
    print(f"segments: {len(seg)} (cutting {int(cutting.sum())}, rapid {int((~cutting).sum())})")
    for name, m in (("cutting", cutting), ("rapid", ~cutting)):
        pct = np.percentile(length[m], [10, 50, 90])
        print(f"  {name:7s} length mean {length[m].mean():.3f}  P10/P50/P90 "
              f"{pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f} mm")
    print(f"  G-code lines with motion: {len(np.unique(lines[1:]))}, "
          f"segments per line: {len(seg) / len(np.unique(lines[1:])):.1f}")


def merged_path(path, tol: float):
    """Analysis helper: drop interior points of collinear runs that stay on
    one G-code line with one feed (tolerance *tol* mm). Hit lines may differ
    slightly from the real pre-pass — only used to see how time scales."""
    from types import SimpleNamespace

    pts, feeds, lines = path.points, path.feed_rates, path.line_ids
    keep = [0]
    start = 0
    for j in range(1, len(pts) - 1):
        same = lines[j] == lines[j + 1] and feeds[j] == feeds[j + 1]
        if same:
            a, b = pts[start], pts[j + 1]
            d = b - a
            n = float(np.dot(d, d))
            inner = pts[start + 1:j + 1]
            if n > 0.0:
                tt = np.clip((inner - a) @ d / n, 0.0, 1.0)
                dev = np.linalg.norm(inner - (a + tt[:, None] * d), axis=1).max()
                if dev <= tol:
                    continue
        keep.append(j)
        start = j
    keep.append(len(pts) - 1)
    keep = np.array(keep)
    return SimpleNamespace(
        points=pts[keep], feed_rates=feeds[keep], line_ids=lines[keep],
        arc_lengths=None,
    )


def _stock(prog, h: float):
    from controller.sim.voxel.stock import StockDefinition

    st = StockDefinition(voxel_size=h)
    st.build_bbox(prog.path)
    return st


def bench_prepass(prog, h: float, backend: str, profile: bool, runs: bool = True):
    from controller.sim.voxel.prepass import CollisionPrepassResult, run_prepass

    get_tool, get_holder = _tools()
    st = _stock(prog, h)
    init = get_tool(prog.tool_changes[0].pocket_number)
    prof = cProfile.Profile() if profile else None
    t = time.perf_counter()
    if prof:
        prof.enable()
    if backend == "dexel":
        from controller.sim.dexel.prepass import run_dexel_prepass

        by_seg, by_line = run_dexel_prepass(
            prog.path, prog.tool_changes, init, st, get_tool, get_holder, merge_runs=runs,
        )
        res = CollisionPrepassResult(by_seg, by_line, frozenset(by_line))
    else:
        res = run_prepass(
            prog.path, prog.tool_changes, init, st,
            get_tool_by_pocket=get_tool, get_holder=get_holder, backend=backend,
        )
    if prof:
        prof.disable()
    dt = time.perf_counter() - t
    return dt, res, prof


def bench_carve(prog, h: float, backend: str, runs: bool = True) -> float:
    """Carve every cutting move (tool changes honoured) into a fresh stock."""
    from controller.sim.voxel.carver import VoxelCarver
    from controller.sim.voxel.gpu_grid import solid_material
    from controller.sim.voxel.prepass import _CpuMaterialSink

    get_tool, _ = _tools()
    st = _stock(prog, h)
    if backend == "dexel":
        from controller.sim.dexel.carver import DexelCarver
        from controller.sim.dexel.stock import solid_dexels

        carver = DexelCarver(solid_dexels(st))
    else:
        carver = VoxelCarver(_CpuMaterialSink(solid_material(st), st.bbox, h))

    pts, feeds, lines = prog.path.points, prog.path.feed_rates, prog.path.line_ids
    tcs = prog.tool_changes
    n_seg = len(pts) - 1
    # Tool per segment (same rule as the pre-pass), then — for dexel —
    # collinear runs carved as one move, exactly as SIM playback does.
    seg_tools, tool, tc = [], get_tool(tcs[0].pocket_number), 0
    for i in range(n_seg):
        while tc < len(tcs) and tcs[tc].line_index <= lines[i]:
            tool = get_tool(tcs[tc].pocket_number) or tool
            tc += 1
        seg_tools.append(tool)
    ends = np.arange(1, n_seg + 1)
    if backend == "dexel" and runs:
        from controller.sim.dexel.grouping import run_ends

        key = np.array([id(x) for x in seg_tools])
        ends = run_ends(pts, feeds, lines, seg_key=key)
    t = time.perf_counter()
    i = 0
    while i < n_seg:
        j = int(ends[i])
        if feeds[i + 1] > 0.0:
            carver.carve_segment(pts[i], pts[j], seg_tools[i])
        i = j
    return time.perf_counter() - t


def stock_mb(prog, h: float, backend: str) -> float:
    st = _stock(prog, h)
    if backend == "dexel":
        from controller.sim.dexel.stock import solid_dexels

        return sum(f.start.nbytes + f.end.nbytes for f in solid_dexels(st).fields) / 1e6
    from controller.sim.voxel.gpu_grid import solid_material

    return solid_material(st).nbytes / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--program", default="workpieces/Gcode.cnc")
    ap.add_argument("--res", type=float, nargs="+", default=[0.5, 0.1, 0.05])
    ap.add_argument("--backends", nargs="+", default=["voxel", "dexel"])
    ap.add_argument("--no-carve", action="store_true", help="skip the carve-only run")
    ap.add_argument("--profile", action="store_true", help="cProfile the pre-pass")
    ap.add_argument("--segments", action="store_true", help="print segment statistics only")
    ap.add_argument("--no-runs", action="store_true",
                    help="dexel: disable collinear-run processing (grouping.py)")
    ap.add_argument("--merge", action="store_true",
                    help="analysis: merge collinear segments (0.5 x res) before running")
    ap.add_argument("--ref", help="JSON file with reference hit lines (written if missing)")
    args = ap.parse_args()

    prog = _load(args.program)
    if args.segments:
        segment_stats(prog)
        return 0
    n_seg = len(prog.path.points) - 1

    ref = None
    if args.ref and Path(args.ref).exists():
        ref = set(json.loads(Path(args.ref).read_text()))

    print(f"{'res':>5} {'backend':7} {'prepass s':>9} {'carve s':>8} "
          f"{'stock MB':>9} {'hits':>5} ref")
    ok = True
    full_path = prog.path
    for h in args.res:
        if args.merge:
            prog.path = merged_path(full_path, 0.5 * h)
            print(f"  merged: {len(full_path.points) - 1} -> {len(prog.path.points) - 1} segments")
        for b in args.backends:
            dt, res, prof = bench_prepass(prog, h, b, args.profile, not args.no_runs)
            hits = sorted(res.hit_lines)
            if args.ref and ref is None:
                Path(args.ref).write_text(json.dumps(hits))
                ref = set(hits)
            same = "-" if ref is None else ("==" if set(hits) == ref else "DIFF")
            ok &= same != "DIFF"
            carve = float("nan") if args.no_carve else bench_carve(prog, h, b, not args.no_runs)
            print(f"{h:5.2f} {b:7} {dt:9.1f} {carve:8.1f} {stock_mb(prog, h, b):9.0f} "
                  f"{len(hits):5d} {same}", flush=True)
            if prof:
                st = pstats.Stats(prof)
                st.sort_stats("tottime").print_stats(15)
                calls = {k[2]: v[1] for k, v in st.stats.items()}
                for fn in ("carve_segment", "check_segment", "subtract", "sweep_z",
                           "sweep_horizontal"):
                    if fn in calls:
                        print(f"  ncalls/segment {fn:16s} {calls[fn] / n_seg:.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
