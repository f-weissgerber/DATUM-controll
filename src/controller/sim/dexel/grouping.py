"""
sim/dexel/grouping.py — collinear runs of tessellated path segments.

PathBuffer splits every G1 line into ≤ 1 mm pieces (``max_step_mm``), so
~10 consecutive segments describe one straight move. For the dexel backend
such a run can be processed as ONE segment:

• Carving: the sweep of a straight move equals the union of its pieces'
  sweeps, and sweep.py computes it in closed form for level moves and
  plunges — so only those runs are merged (3-D moves are sampled; merging
  them would spread the same samples over a longer move).
• Collision (pre-pass): a run's check is a conservative FILTER — each
  piece's checked volume lies inside the run's, and material only shrinks
  while the run is carved, so "run clear" ⇒ "every piece clear". Only a
  run that hits is re-checked piece by piece (see voxel/prepass.py).

Runs never cross a G-code line, a feed change (incl. rapid ↔ cut) or a
tool change, so line numbers and per-segment results stay exact.
"""
from __future__ import annotations

import numpy as np

_LEVEL_EPS = 1e-6   # mm — |dz| (level) or |dxy| (plunge) treated as zero

# Default collinearity tolerance: float32 noise of PathBuffer's linear
# tessellation only. Anything coarser (e.g. half the ray spacing) also
# merges tessellated ARCS into chords — measurably different geometry
# (up to ~100 mm³ less removed on workpieces/Gcode.cnc at 0.5 mm) — so
# the result would no longer match the per-segment carve.
COLLINEAR_TOL = 1e-4


def run_ends(
    points:   np.ndarray,          # (N, 3)
    feeds:    np.ndarray,          # (N,)  — feeds[k]: move ARRIVING at point k
    line_ids: np.ndarray,          # (N,)
    tol:      float = COLLINEAR_TOL,   # max. deviation of interior points, mm
    seg_key:  np.ndarray | None = None,   # (N-1,) e.g. tool index per segment
) -> np.ndarray:
    """For every segment i (points[i] → points[i+1]) the exclusive end index
    of the run starting at i: segments i .. run_end[i]-1 form one straight,
    mergeable move (run_end[i] == i + 1 means "not merged"). Only values at
    run starts are used by callers; inside a run they describe the
    remaining tail, so processing may start anywhere."""
    pts = np.asarray(points, dtype="f8")
    n_seg = len(pts) - 1
    ends = np.arange(1, n_seg + 1)
    if n_seg < 2:
        return ends

    d = np.diff(pts, axis=0)
    level = np.abs(d[:, 2]) <= _LEVEL_EPS
    plunge = np.hypot(d[:, 0], d[:, 1]) <= _LEVEL_EPS
    kind = np.where(level, 1, np.where(plunge, 2, 0))          # 0 = 3-D: never merged
    seg_line = np.asarray(line_ids)[1:]
    seg_feed = np.asarray(feeds)[1:]
    key = np.zeros(n_seg, dtype=np.int64) if seg_key is None else np.asarray(seg_key)

    # Pairwise "j can follow j-1 in a run" without the geometric test.
    joinable = np.zeros(n_seg, dtype=bool)
    joinable[1:] = (
        (kind[1:] != 0) & (kind[1:] == kind[:-1])
        & (seg_line[1:] == seg_line[:-1])
        & (seg_feed[1:] == seg_feed[:-1])
        & (key[1:] == key[:-1])
    )

    # Walk backwards so each segment's end is its run's end.
    i = n_seg - 1
    while i >= 0:
        j = i
        while j > 0 and joinable[j] and _collinear(pts, j - 1, i + 1, tol):
            j -= 1
        ends[j:i + 1] = i + 1
        i = j - 1
    return ends


def _collinear(pts: np.ndarray, a: int, b: int, tol: float) -> bool:
    """Do all points strictly between a and b lie within *tol* of a→b?"""
    p, q = pts[a], pts[b]
    v = q - p
    n = float(v @ v)
    if n == 0.0:
        return False
    inner = pts[a + 1:b]
    t = np.clip((inner - p) @ v / n, 0.0, 1.0)
    dev = inner - (p + t[:, np.newaxis] * v)
    return float(np.max(np.einsum("ij,ij->i", dev, dev))) <= tol * tol
