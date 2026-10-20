"""Tests for the tri-dexel simulation backend (controller.sim.dexel).

CPU parts run everywhere (numpy only). The GPU test needs a standalone
OpenGL 3.3 context — headless, e.g. ``xvfb-run -a pytest`` — and is skipped
when none can be created.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from controller.sim.dexel.carver import DexelCarver
from controller.sim.dexel.collision import check_segment as dexel_check
from controller.sim.dexel.field import EMPTY_START, DexelField
from controller.sim.dexel.stock import DexelStock
from controller.sim.simulation.tool_definition import ToolDefinition, ToolType
from controller.sim.voxel.carver import VoxelCarver
from controller.sim.voxel.collision import check_segment as voxel_check
from controller.sim.voxel.gpu_grid import solid_material
from controller.sim.voxel.prepass import _CpuMaterialSink, run_prepass
from controller.sim.voxel.stock import BoundingBox, StockDefinition, StockShape

H = 0.5


def _stock(shape: StockShape = StockShape.BOUNDING_BOX, h: float = H) -> StockDefinition:
    st = StockDefinition(shape=shape, voxel_size=h, round_radius_mm=20.0)
    st.bbox = BoundingBox(-20, 20, -20, 20, -10, 0)
    return st


def _tool(tool_type: ToolType = ToolType.ENDMILL, d: float = 6.0) -> ToolDefinition:
    return ToolDefinition(1, 1, d, tool_type=tool_type, cutting_length=15.0, total_length=40.0)


def _seg(a, b):
    return np.array(a, "f4"), np.array(b, "f4")


# ── DexelField.subtract ───────────────────────────────────────────────────────

def _one_ray(k: int = 2) -> DexelField:
    f = DexelField(2, np.zeros(3, "f4"), 1.0, nu=1, nv=1, k=k)
    f.fill(np.array([[0.0]], "f4"), np.array([[10.0]], "f4"))
    return f


def _intervals(f: DexelField) -> list[tuple[float, float]]:
    s, e = f.start[0, 0], f.end[0, 0]
    return [(float(a), float(b)) for a, b in zip(s, e) if a <= b]


@pytest.mark.parametrize("lo, hi, expected", [
    (3.0, 5.0,   [(0.0, 3.0), (5.0, 10.0)]),   # split
    (-1.0, 4.0,  [(4.0, 10.0)]),               # cut from below
    (6.0, 12.0,  [(0.0, 6.0)]),                # cut from above
    (-1.0, 11.0, []),                          # fully removed
    (11.0, 12.0, [(0.0, 10.0)]),               # miss
])
def test_subtract_cases(lo, hi, expected):
    f = _one_ray()
    f.subtract(0, 1, 0, 1, np.array([[lo]], "f4"), np.array([[hi]], "f4"))
    assert _intervals(f) == expected
    assert f.overflow_count == 0


def test_subtract_overflow_keeps_material():
    f = _one_ray(k=2)
    f.subtract(0, 1, 0, 1, np.array([[2.0]], "f4"), np.array([[3.0]], "f4"))
    f.subtract(0, 1, 0, 1, np.array([[6.0]], "f4"), np.array([[7.0]], "f4"))
    # Three pieces needed, two slots: the last slot is widened over the gap.
    assert _intervals(f) == [(0.0, 2.0), (3.0, 10.0)]
    assert f.overflow_count == 1


def test_lo_ge_hi_leaves_ray_untouched():
    f = _one_ray()
    f.subtract(0, 1, 0, 1, np.array([[5.0]], "f4"), np.array([[5.0]], "f4"))
    assert _intervals(f) == [(0.0, 10.0)]
    assert not f.start[0, 0, 1] < EMPTY_START


def test_subtract_random_against_reference():
    """Random cut sequences on one ray vs. a plain Python interval list."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        f = _one_ray(k=8)
        ref = [(0.0, 10.0)]
        for _ in range(rng.integers(1, 6)):
            lo, hi = sorted(rng.uniform(-1.0, 11.0, 2).astype("f4").tolist())
            f.subtract(0, 1, 0, 1, np.array([[lo]], "f4"), np.array([[hi]], "f4"))
            nxt = []
            for s, e in ref:
                if min(e, lo) > s:
                    nxt.append((s, min(e, lo)))
                if e > max(s, hi):
                    nxt.append((max(s, hi), e))
            ref = nxt
        assert f.overflow_count == 0
        assert _intervals(f) == pytest.approx(ref)


# ── Initial stock ─────────────────────────────────────────────────────────────

def test_solid_box_volume_exact_on_all_axes():
    dx = DexelStock(_stock())
    for axis in range(3):
        assert dx.volume_mm3(axis) == pytest.approx(40 * 40 * 10, rel=1e-6)


def test_solid_round_matches_voxel_occupancy():
    st = _stock(StockShape.ROUND)
    dx = DexelStock(st)
    np.testing.assert_array_equal(dx.occupancy(), solid_material(st) == 255)
    # X/Y rays carry exact chords: volume within 1 % of π r² h.
    for axis in (0, 1):
        assert dx.volume_mm3(axis) == pytest.approx(np.pi * 20**2 * 10, rel=0.01)


# ── Carving ───────────────────────────────────────────────────────────────────

# Off-grid coordinates: on-grid ones put exact surfaces precisely on cell
# centres (e.g. a 90° cone), where the two backends legitimately break the
# tie differently.
_SEGMENTS = [
    ((-15.13, -5.07, -3.11), (15.13, -5.07, -3.11)),   # level pass
    ((15.13, -5.07, -3.11), (10.21, 9.93, -6.03)),     # 3-D ramp
    ((0.07, 0.11, 2.0), (0.07, 0.11, -7.97)),          # plunge
]


@pytest.mark.parametrize("tool_type", [
    ToolType.ENDMILL, ToolType.BALL_ENDMILL, ToolType.BULL_ENDMILL, ToolType.CHAMFER,
])
def test_carve_matches_voxel_carver(tool_type):
    tool = _tool(tool_type)
    tool.corner_radius = 1.0
    tool.tip_angle = 90.0
    st = _stock()
    mat = solid_material(st)
    vc = VoxelCarver(_CpuMaterialSink(mat, st.bbox, H))
    dx = DexelStock(st)
    dc = DexelCarver(dx)
    for a, b in _SEGMENTS:
        vc.carve_segment(*_seg(a, b), tool)
        dc.carve_segment(*_seg(a, b), tool)

    vox = mat == 255
    diff = np.count_nonzero(dx.occupancy() != vox)
    assert diff <= 0.002 * vox.size, diff          # only boundary cells differ
    assert all(f.overflow_count == 0 for f in dx.fields)


@pytest.mark.parametrize("tool_type", [
    ToolType.ENDMILL, ToolType.BALL_ENDMILL, ToolType.BULL_ENDMILL, ToolType.CHAMFER,
])
def test_three_ray_sets_agree(tool_type):
    """All three ray sets describe the same solid: their volumes agree at a
    fine spacing. Each is quantised crosswise (a slot floor between two
    X-ray rows counts as a full row), hence the 2 % margin."""
    tool = _tool(tool_type)
    tool.corner_radius = 1.0
    tool.tip_angle = 90.0
    dx = DexelStock(_stock(h=0.2))
    dc = DexelCarver(dx)
    for a, b in _SEGMENTS:
        dc.carve_segment(*_seg(a, b), tool)
    removed = [40 * 40 * 10 - dx.volume_mm3(axis) for axis in range(3)]
    assert max(removed) - min(removed) <= 0.02 * max(removed), removed


@pytest.mark.parametrize("tool_type", [ToolType.ENDMILL, ToolType.BALL_ENDMILL])
def test_level_pass_volume_analytic(tool_type):
    """Level slot, tip at z, length L, radius R, stock top at 0:
    endmill → (2RL + πR²)·|z|; ball (R ≤ |z|) → half-cylinder + sphere
    below the ball centre, full slot above it.

    The Z rays measure the depth exactly (1 %); X/Y rays see it quantised
    to their row spacing in Z, so they get h/|z| relative tolerance.
    """
    r, z, length, h = 3.0, -4.13, 20.26, 0.25
    dx = DexelStock(_stock(h=h))
    DexelCarver(dx).carve_segment(*_seg((-10.13, 0.07, z), (10.13, 0.07, z)), _tool(tool_type))
    slot = 2 * r * length + np.pi * r * r
    if tool_type == ToolType.ENDMILL:
        expected = slot * abs(z)
    else:
        expected = np.pi * r * r / 2 * length + 2 / 3 * np.pi * r**3 + slot * (abs(z) - r)
    for axis in range(3):
        removed = 40 * 40 * 10 - dx.volume_mm3(axis)
        rel = 0.01 if axis == 2 else h / abs(z)
        assert removed == pytest.approx(expected, rel=rel), axis


def test_level_pass_floor_is_exact():
    """A level move leaves the Z rays under it ending exactly at the tip
    height — not rounded to the ray spacing like a voxel layer."""
    dx = DexelStock(_stock())
    DexelCarver(dx).carve_segment(*_seg((-10, 0, -3.17), (10, 0, -3.17)), _tool())
    f = dx.z_field
    iy = int((0.0 + 20) / H)
    ix = int((0.0 + 20) / H)
    assert float(f.end[iy, ix, 0]) == pytest.approx(-3.17, abs=1e-5)
    assert float(f.start[iy, ix, 0]) == pytest.approx(-10.0, abs=1e-5)


def test_ball_floor_profile_is_exact():
    """Under a level ball-end pass, a Z ray at lateral offset d ends at
    tip + R − sqrt(R² − d²)."""
    r = 3.0
    dx = DexelStock(_stock())
    DexelCarver(dx).carve_segment(*_seg((-10, 0, -4), (10, 0, -4)), _tool(ToolType.BALL_ENDMILL))
    f = dx.z_field
    ix = int(20 / H)
    for iy in range(int((20 - 2.5) / H), int((20 + 2.5) / H)):
        d = abs(float(f.v_centers(iy, iy + 1)[0]))
        expected = -4 + r - np.sqrt(r * r - d * d)
        assert float(f.end[iy, ix, 0]) == pytest.approx(expected, abs=0.01)


# ── Collision ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a, b, is_rapid", [
    ((-15, 0, -2), (15, 0, -2), True),     # rapid through material
    ((-15, 0, 5), (15, 0, 5), True),       # rapid in air
    ((-15, 0, -2), (15, 0, -2), False),    # normal cut
    ((-15, 0, -16), (15, 0, -16), False),  # shank below cutting length
])
def test_collision_matches_voxel(a, b, is_rapid):
    st = _stock()
    mat = solid_material(st)
    dx = DexelStock(st)
    tool = _tool()
    hv = voxel_check(mat, st.bbox, H, *_seg(a, b), tool, is_rapid)
    hd = dexel_check(dx, *_seg(a, b), tool, is_rapid)
    assert (hv is None) == (hd is None)
    if hv is not None:
        assert hd.kind == hv.kind
        # Any point of the contact is a valid report (the two backends pick
        # different ones): it must be within the tool's reach of the move
        # and inside the stock.
        p = hd.point
        a_, b_ = _seg(a, b)
        assert min(a_[0], b_[0]) - 3 - H <= p[0] <= max(a_[0], b_[0]) + 3 + H
        assert abs(p[1] - a_[1]) <= 3 + H
        assert st.bbox.z_min - 1e-4 <= p[2] <= st.bbox.z_max + 1e-4


@pytest.mark.parametrize("profile", [
    [(0.0, 10.0), (30.0, 10.0)],                              # monotone → closed form
    [(0.0, 10.0), (5.0, 10.0), (5.01, 4.0), (30.0, 4.0)],     # necks in → sampled path
])
def test_holder_collision_matches_voxel(profile):
    """A cut deeper than the tool's total length drives the holder into
    the stock; both backends must report it as a holder hit."""
    from controller.sim.simulation.tool_holder import HolderProfile

    holder = HolderProfile("test", "ER_COLLET", 30.0, profile)
    tool = _tool()
    tool.cutting_length = tool.total_length = 8.0     # no shank zone: holder only
    st = _stock()
    a, b = _seg((-15.13, 0.07, -9.0), (15.13, 0.07, -9.0))
    hv = voxel_check(solid_material(st), st.bbox, H, a, b, tool, False, holder)
    hd = dexel_check(DexelStock(st), a, b, tool, False, holder)
    assert hv is not None and hd is not None
    assert hd.kind == hv.kind == "holder"


def test_prepass_backends_agree():
    pts = np.array([
        (-15, 0, 5), (-15, 0, -2), (15, 0, -2),   # plunge + cut
        (15, 0, 5), (-15, 0, 5),                  # retract + rapid back
        (-15, 10, -1), (15, 10, -1),              # rapid THROUGH uncut material
    ], "f4")
    feeds = np.array([0, 100, 100, 0, 0, 0, 0], "f4")
    path = SimpleNamespace(points=pts, feed_rates=feeds, line_ids=np.arange(len(pts), dtype="i4"))
    st = _stock()
    tool = _tool()
    results = {
        backend: run_prepass(
            path, [], tool, st,
            get_tool_by_pocket=lambda _p: tool, get_holder=lambda _n: None, backend=backend,
        )
        for backend in ("voxel", "dexel")
    }
    assert results["voxel"].hit_lines == results["dexel"].hit_lines
    assert results["dexel"].hit_lines        # the last rapid must collide


# ── GPU ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def gl_ctx():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # no display / no GL driver
        pytest.skip(f"no OpenGL 3.3 context: {exc}")
    yield ctx
    ctx.release()


@pytest.fixture
def core_app(request):
    """DexelRenderer reads AppSettings (a QObject), so a Qt application
    must exist. Prefer pytest-qt's shared QApplication (other test modules
    need a QApplication, and only one Qt application may ever exist per
    process); fall back to a QCoreApplication, which needs no GUI platform."""
    try:
        return request.getfixturevalue("qapp")
    except pytest.FixtureLookupError:
        from PySide6.QtCore import QCoreApplication
        return QCoreApplication.instance() or QCoreApplication([])


def test_gpu_upload_and_render(gl_ctx, core_app):
    import moderngl

    from controller.sim.dexel.gpu_fields import GpuDexelGrid
    from controller.sim.dexel.renderer import DexelRenderer

    st = _stock()
    grid = GpuDexelGrid(gl_ctx, st)
    DexelCarver(grid.dexels).carve_segment(*_seg((-10, 0, -3), (10, 0, -3)), _tool())
    assert grid.upload_if_dirty()
    assert not grid.is_dirty

    # Texture contents == CPU arrays, layer by layer.
    for f, tex in zip(grid.dexels.fields, grid.textures):
        data = np.frombuffer(tex.read(), "f4").reshape(f.k, f.nv, f.nu, 2)
        np.testing.assert_array_equal(data[..., 0], f.start.transpose(2, 0, 1))
        np.testing.assert_array_equal(data[..., 1], f.end.transpose(2, 0, 1))

    w = h = 64
    fbo = gl_ctx.framebuffer(gl_ctx.renderbuffer((w, h)), gl_ctx.depth_renderbuffer((w, h)))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
    gl_ctx.enable(moderngl.DEPTH_TEST)
    # Orthographic top view: x,y ∈ [-25, 25] → NDC, z ∈ [-50, 50].
    mvp = np.diag([1 / 25, 1 / 25, -1 / 50, 1]).astype("f4")
    renderer = DexelRenderer(gl_ctx, grid)
    renderer.render(mvp, np.array([0, 0, 100], "f4"))
    img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(h, w, 3)
    lit = img.sum(axis=-1) > 0
    # Stock covers x,y ∈ [-20, 20] of [-25, 25] → central 80 % lit, border not.
    assert lit[h // 2, w // 2] and lit[8, 8]
    assert not lit[1, 1]
    # The slot floor is deeper → darker (height tint) than the stock top.
    assert img[h // 2, w // 2].sum() < img[8, 8].sum()
    renderer.release()
    grid.release()


# ── Collinear runs (grouping.py / dexel prepass) ──────────────────────────────

def _tessellated(moves):
    """Path like PathBuffer's: each (target, feed) move split into ≤ 1 mm
    pieces, one G-code line per move."""
    pts, feeds, lines = [np.array(moves[0][0], "f4")], [0.0], [0]
    for line, (target, feed) in enumerate(moves[1:], start=1):
        a, b = pts[-1], np.array(target, "f4")
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / 1.0)))
        for k in range(1, n + 1):
            pts.append(a + (b - a) * (k / n))
            feeds.append(feed)
            lines.append(line)
    return SimpleNamespace(
        points=np.array(pts, "f4"), feed_rates=np.array(feeds, "f4"),
        line_ids=np.array(lines, "i4"),
    )


def test_run_ends_respect_line_feed_and_kind():
    from controller.sim.dexel.grouping import run_ends

    path = _tessellated([
        ((-15, 0, 5), 0),
        ((-15, 0, -2), 100),     # plunge → one run
        ((15, 0, -2), 100),      # level  → one run (new line)
        ((15, 10, -2), 200),     # level, other feed → own run
        ((5, 10, -6), 200),      # 3-D ramp → never merged
    ])
    ends = run_ends(path.points, path.feed_rates, path.line_ids, 0.05)
    lines = path.line_ids[1:]
    starts = [0]
    while starts[-1] < len(ends):
        starts.append(int(ends[starts[-1]]))
    runs = [(s, e) for s, e in zip(starts[:-1], starts[1:])]
    for s, e in runs:
        assert len(set(lines[s:e])) == 1          # never crosses a G-code line
    run_lines = [int(lines[s]) for s, _ in runs]
    assert run_lines[:3] == [1, 2, 3]             # plunge, level, level: one run each
    ramp = [e - s for s, e in runs if lines[s] == 4]
    assert ramp and all(n == 1 for n in ramp)     # ramp pieces stay separate


@pytest.mark.parametrize("tool_kw", [
    {},                                                    # endmill, shank zone above 15 mm
    {"cutting_length": 2.0, "total_length": 30.0},         # short flutes → shank hits on ramps
])
def test_dexel_prepass_runs_identical(tool_kw):
    """Run-wise processing must give exactly the per-segment results."""
    from controller.sim.dexel.prepass import run_dexel_prepass

    tool = _tool()
    for k, v in tool_kw.items():
        setattr(tool, k, v)
    path = _tessellated([
        ((-15, -5, 5), 0),
        ((-15, -5, -1), 100), ((15, -5, -1), 100),    # level cut
        ((15, -5, -4), 100), ((-15, -5, -4), 100),    # deeper level cut (shank on short tool)
        ((-15, 5, -1), 100), ((15, 5, -7), 100),      # 3-D ramp
        ((15, 5, 5), 0), ((-15, 8, 5), 0),
        ((-15, 8, -1), 0), ((15, 8, -1), 0),          # rapid through material
    ])
    st = _stock()
    args = (path, [], tool, st, lambda _p: tool, lambda _n: None)
    seg_a, line_a = run_dexel_prepass(*args, merge_runs=False)
    seg_b, line_b = run_dexel_prepass(*args, merge_runs=True)
    assert set(seg_a) == set(seg_b) and set(line_a) == set(line_b)
    assert seg_a                                       # the scenario does collide
    for i in seg_a:
        assert seg_a[i].kind == seg_b[i].kind
        np.testing.assert_allclose(seg_a[i].point, seg_b[i].point, atol=1e-4)
