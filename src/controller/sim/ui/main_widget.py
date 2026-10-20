"""
sim/ui/main_widget.py — DatumSimWidget: 3D G-code viewer + sim player.

Composes:
  • Viewport       — ModernGL OpenGL widget (G-code path + tool)
  • SettingsPanel  — right-edge slide-out with functional sim settings
  • ControlHub     — bottom-centre playback bar + info labels

All three are positioned as overlays on top of the viewport via
_layout_overlays(), which is called from resizeEvent and open/close panel
actions so the geometry always stays correct.

Voxel material-removal simulation
----------------------------------
When the C++ extension is gone, the voxel sim is implemented entirely in
Python + numpy (carver.py / gpu_grid.py) with a GLSL raymarching renderer
(renderer.py).  All objects (GpuVoxelGrid, VoxelCarver, VoxelSimController,
VoxelRenderer) require an active GL context and are therefore created lazily
after the viewport's initializeGL fires.

AppSettings.sim_backend == "dexel" swaps in the tri-dexel model (sim/dexel:
GpuDexelGrid, DexelCarver, DexelRenderer, dexel collision check) behind the
same VoxelSimController and pre-pass; everything else here is shared.

Public interface (mirrors SimPlaceholder so machine_page.py can swap freely):
    set_file(path)              — load and compile a G-code file
    set_mode("SIM"|"MACHINE")  — sim player vs. live machine follow
    set_state(state_str)        — string state label (passed through)
    set_position(x, y, z)      — push live machine position (MACHINE mode)
    set_line(line)              — highlight a G-code line
    sim_play/pause/reset/seek/set_speed — simulation playback controls
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QWidget

from controller.sim.ui.viewport        import Viewport, ToolMode, PathMode
from controller.sim.ui.overlay.settings_panel import SettingsPanel
from controller.sim.ui.overlay.control_hub    import ControlHub
from controller.sim.gcode.compiler            import GCodeCompiler
from controller.sim.simulation.player         import SimulationPlayer
from controller.sim.simulation.tool_database  import get_tool, get_tool_by_pocket
from controller.sim.simulation.tool_definition import ToolDefinition
from controller.sim.simulation.tool_holder    import HolderProfile
from controller.persistence.tool_db           import ToolDatabase, ToolDatabaseSignals
from controller.persistence.workpiece_db      import WorkpieceDatabase
from controller.sim.core.settings             import AppSettings

# Voxel sim — pure Python/numpy + GLSL raymarching (no C++ required)
try:
    from controller.sim.voxel.stock      import StockDefinition, StockShape
    from controller.sim.voxel.gpu_grid   import GpuVoxelGrid, solid_material
    from controller.sim.voxel.carver     import VoxelCarver
    from controller.sim.voxel.controller import VoxelSimController
    from controller.sim.voxel.renderer   import VoxelRenderer
    from controller.sim.voxel.collision  import CollisionHit
    from controller.sim.voxel.prepass    import CollisionPrepassResult, run_prepass
    # Tri-dexel backend (AppSettings.sim_backend == "dexel") — same deps.
    from controller.sim.dexel.stock      import solid_dexels
    from controller.sim.dexel.gpu_fields import GpuDexelGrid
    from controller.sim.dexel.carver     import DexelCarver
    from controller.sim.dexel.collision  import check_segment as dexel_check_segment
    from controller.sim.dexel.renderer   import DexelRenderer
    from controller.sim.dexel.grouping   import run_ends as dexel_run_ends
    _VOXEL_AVAILABLE = True
except ImportError:
    _VOXEL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Backlog (mm of arc-length still needing to be carved) below which _tick()
# carves synchronously, in-tick, on the main/GL thread instead of handing it
# to the background worker. Derived from the worst plausible single-tick
# advance: max speed-slider multiplier (20x, see control_hub.py _SPEED_MAX)
# x a generous cutting-feed ceiling (5000 mm/min) / 60 x the 32ms tick
# interval ~= 53mm; the highest cutting feed actually used in
# workpieces/Gcode.cnc (1828.8 mm/min) gives ~= 19.5mm at 20x, so 40mm leaves
# real margin either way. This is only a soft performance/UX threshold —
# exceeding it never produces a wrong result, it just routes to the (already
# correct) chunked background path instead of carving inline. It relies on
# PathBuffer capping tessellated-point density to ~1-2 points/mm (see
# path_buffer.py's max_step_mm) — without that bound this threshold would not
# actually bound per-tick carve cost.
_SYNC_CARVE_MAX_MM = 40.0


class DatumSimWidget(QWidget):
    """Top-level 3D simulation widget — drop-in replacement for SimPlaceholder."""

    # Emitted whenever the estimated part/cycle time is (re)computed — a
    # float in seconds, or None if no program is loaded. The sim widget owns
    # the computation (it has the PathBuffer + current overrides); the
    # display lives elsewhere (ProgramInfoCard on MachinePage), pushed-state
    # style like set_position()/set_line().
    part_time_changed = Signal(object)

    # Emitted whenever the active tool changes (T-command crossed, program
    # (re)loaded, or reset) — a ToolDefinition. Lets MachinePage's
    # ToolInfoCard reflect the tool called out in the running program,
    # including during a real MACHINE-mode run.
    tool_changed = Signal(object)

    # Emitted the first time a new VoxelSimController.collision_hit
    # (collision.CollisionHit) is observed — never re-emitted for the same
    # hit on subsequent ticks. See _handle_collision_state().
    collision_detected = Signal(object)

    # Emitted once a background-compiled program from set_file() is fully
    # applied to this widget (path uploaded to GL, tool/time state
    # refreshed, voxel/prepass scheduled) — see _on_compile_done(). The
    # async analogue of "set_file() returned" under the old synchronous
    # design; MachinePage waits for this before treating the file as
    # actually loaded (see its _file_ready flag) rather than trusting
    # set_file()'s own (now immediate, fire-and-forget) return.
    file_ready = Signal(str)      # gcode path

    # Emitted if the background compile in set_file() raised — today's
    # synchronous compiler.load_file() call let exceptions propagate
    # straight to whatever called set_file(); backgrounding it means the
    # worker thread must catch instead, so this exists to give that
    # failure somewhere to go.
    load_failed = Signal(object)  # Exception

    # Internal plumbing for the collision pre-pass's background->main
    # thread handoff (generation, CollisionPrepassResult|None) — see
    # _start_prepass()'s docstring for why this (a Qt signal) is used
    # instead of QTimer.singleShot() from the worker thread.
    _prepass_done = Signal(int, object)

    # Internal plumbing for the voxel-grid CPU build's background->main
    # thread handoff (generation, StockDefinition, material ndarray) — see
    # _schedule_voxel_sim()'s docstring. Same pattern/reasoning as
    # _prepass_done above.
    _voxel_material_ready = Signal(int, object, object)

    # Internal plumbing for the G-code compile's background->main thread
    # handoff (generation, GCodeProgram|None, Workpiece|None,
    # Exception|None) — see set_file()'s docstring. Same pattern as
    # _prepass_done/_voxel_material_ready above.
    _compile_done = Signal(int, object, object, object)

    _TOOL_MAP = {
        "Endmill":    ToolMode.CYLINDER,
        "Point":      ToolMode.POINT,
        "None":       ToolMode.NONE,
    }
    _PATH_MAP = {
        "Complete":    PathMode.FULL,
        "Progressive": PathMode.PROGRESSIVE,
        "None":        PathMode.NONE,
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.viewport    = Viewport(self)
        self.settings    = SettingsPanel(self)
        self.control_hub = ControlHub(self)
        self.compiler    = GCodeCompiler()

        # ── Internal state ────────────────────────────────────────────────────
        self._player:               SimulationPlayer | None = None
        self._state:                str                     = "IDLE"
        self._mode:                 str                     = "SIM"
        self._clean_lines:          list[str]               = []
        self._raw_lines:            list[str]               = []
        self._tool_changes:         list                    = []
        self._last_tool_change_idx: int                     = -1
        self._path_mode:            PathMode                = PathMode.FULL
        self._tool_mode:            ToolMode                = ToolMode.POINT
        self._current_tool:         ToolDefinition | None   = None
        # The (tool, holder) pair last actually pushed to the viewport/voxel
        # controller via _apply_tool() — lets it skip the expensive mesh
        # rebuild entirely when nothing changed (see _apply_tool()'s own
        # docstring: this used to run unconditionally on every set_mode()
        # call, i.e. every Start/Stop/Feed-Hold/Reset click, at ~186-420ms
        # each, even though the tool essentially never actually changes on
        # those). ToolDefinition/HolderProfile are plain @dataclass, so
        # value equality (not identity) correctly recognizes "the same
        # tool" even across separate ToolDatabase lookups returning fresh
        # instances with equal field values.
        self._last_applied_holder: HolderProfile | None = None
        self._last_program                                  = None
        self._current_workpiece                             = None   # domain.models.Workpiece | None
        self._current_line:         int                     = -1
        # Whether the viewport/control_hub are CURRENTLY showing the
        # collision-active state — toggled to match ctrl.collision_hit's
        # live on/off value each tick, so viewport.set_collision()/
        # control_hub.set_collision() are only called on actual transitions
        # (cheap to call _handle_collision_state() unconditionally).
        self._collision_active = False

        # Feed/rapid overrides (1.0 = 100%), pushed in from MACHINE mode via
        # set_feed_override()/set_rapid_override() — drive the estimated
        # part-time display; SIM mode simply leaves them at 1.0.
        self._feed_override:  float = 1.0
        self._rapid_override: float = 1.0

        # Voxel objects (all None until _create_voxel_sim() runs after initializeGL)
        self._voxel_ctrl:     VoxelSimController | None = None
        self._voxel_renderer: VoxelRenderer | None      = None
        self._pending_voxel_setup: bool = False

        # Voxel-grid CPU build (solid_material() over the whole stock
        # volume — a numpy allocation that can run into tens of
        # milliseconds or more for a large/fine-grained stock) runs in a
        # background thread — see _schedule_voxel_sim(). generation/abort
        # guard against a stale worker's result landing after a newer
        # load/settings-change/teardown superseded it, same pattern as
        # _prepass_generation/_prepass_abort above. The pending_voxel_*
        # pair carries a completed worker's result over to
        # _create_voxel_sim(), which only ever does the GL-dependent half
        # (GpuVoxelGrid's texture upload + Carver/Controller/Renderer) —
        # never solid_material() itself anymore.
        self._voxel_build_generation: int             = 0
        self._voxel_build_abort:      threading.Event = threading.Event()
        self._pending_voxel_stock:    "StockDefinition | None" = None
        self._pending_voxel_material: "np.ndarray | None"      = None

        # G-code compile (GCodeCompiler.load_file() — tokenize/parse/plan/
        # tessellate the whole file, profiled at ~195ms for a modest
        # 1000-line program) runs in a background thread — see set_file().
        # Same generation/abort-Event guard pattern as the two pairs above;
        # _pending_load_path records which path THIS generation is for,
        # used only for a same-path sanity check in _on_compile_done().
        self._compile_generation: int             = 0
        self._compile_abort:      threading.Event = threading.Event()
        self._pending_load_path:  str | None      = None

        # Carving runs in a background thread so the main-thread controller UI
        # is never blocked by long numpy operations (deep plunges etc.).
        self._carve_thread: threading.Thread | None = None
        self._carve_done   = threading.Event()   # set by worker when grid is dirty
        self._carve_abort  = threading.Event()   # set to cancel in-flight carving

        # MACHINE-mode carving: track consecutive positions for segment carving.
        # Written and read exclusively on the main thread — no lock required.
        self._last_machine_pos:       np.ndarray | None                    = None
        self._pending_machine_carve:  tuple[np.ndarray, np.ndarray] | None = None

        # Collision pre-pass — see _start_prepass()'s docstring. _prepass_result
        # is the last COMPLETED table (or None, before the first one lands);
        # _prepass_generation guards against a stale background worker's
        # result overwriting a newer trigger's (reload / parameter change
        # arriving while an older pre-pass is still running).
        self._prepass_result:     "CollisionPrepassResult | None"      = None
        self._prepass_generation: int                                  = 0
        self._prepass_abort:      threading.Event                      = threading.Event()
        self._pending_prepass_cb: Callable[["CollisionPrepassResult | None"], None] | None = None
        self._prepass_done.connect(self._on_prepass_done)
        self._voxel_material_ready.connect(self._on_voxel_material_ready)
        self._compile_done.connect(self._on_compile_done)

        # ── Signal connections ────────────────────────────────────────────────
        # Tool/path display mode are read straight from AppSettings rather
        # than forwarded through a specific settings-widget instance: the
        # same setting can now be edited from either the sim widget's own
        # overlay panel or the app-wide SettingsPage (see sim_panel.py's
        # build_sections()), so AppSettings is the single source of truth
        # both listen to and write through.
        s = AppSettings.instance()
        s.tool_mode_changed.connect(
            lambda name: self.set_tool_mode(self._TOOL_MAP.get(name, ToolMode.CYLINDER)))
        s.path_mode_changed.connect(
            lambda name: self.set_path_mode(self._PATH_MAP.get(name, PathMode.PROGRESSIVE)))

        s.voxel_size_changed.connect(self._on_voxel_size_changed)
        s.sim_backend_changed.connect(self._on_sim_backend_changed)
        s.voxel_enabled_changed.connect(self._on_voxel_enabled_changed)
        # Rebuild the sim whenever stock geometry settings change
        s.stock_shape_changed.connect(self._on_stock_settings_changed)
        s.stock_z_offset_changed.connect(self._on_stock_settings_changed)
        s.stock_height_changed.connect(self._on_stock_settings_changed)
        s.stock_round_radius_changed.connect(self._on_stock_settings_changed)
        s.stock_width_changed.connect(self._on_stock_settings_changed)
        s.stock_depth_changed.connect(self._on_stock_settings_changed)
        s.stock_x_offset_changed.connect(self._on_stock_settings_changed)
        s.stock_y_offset_changed.connect(self._on_stock_settings_changed)
        s.collision_detection_enabled_changed.connect(self._on_collision_enabled_changed)
        # NOTE: start_safe_z_mm deliberately does NOT invalidate the
        # pre-pass — per its own docstring it only takes effect on the
        # next (re)load, since the value is baked into the already-
        # tessellated path at load time; recomputing against the same,
        # unchanged path would just waste a background pass.

        # Re-apply the active tool whenever ITS row changes in the tool DB
        # (edited on ToolPage, or reassigned to a different magazine
        # pocket) — without this, _current_tool stays whatever was looked
        # up at the last tool-change/load/reset, so edits made on ToolPage
        # while a program is loaded never reach the 3D view/tool info
        # pill until the next T-command or reload happens to re-trigger
        # _apply_tool(). get_tool()/_apply_tool() themselves already read
        # live from the DB — only this missing subscription was stale.
        ToolDatabaseSignals.instance().tool_changed.connect(self._on_tool_db_changed)

        # ── Render tick (~30 fps) ─────────────────────────────────────────────
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(32)
        self._render_timer.timeout.connect(self._tick)
        self._render_timer.start()

        self._connect_control_hub()
        # Re-centre the hub whenever its height changes (info row toggled)
        self.control_hub.layout_changed.connect(self._layout_overlays)
        self._layout_overlays()

    # ── File loading ──────────────────────────────────────────────────────────

    def set_file(self, path: str) -> None:
        """Kick off compiling *path* in the background and return
        immediately — does NOT block until the file is actually loaded.

        GCodeCompiler.load_file() (tokenize/parse/plan/tessellate the
        whole file) profiled at ~195ms for a modest 1000-line program —
        real work, but pure Python/numpy with no GL dependency, so unlike
        the GPU-bound steps further down it has no reason to run on the
        GUI thread. Runs in a background thread (same generation-counter/
        threading.Event abort-guard pattern as _start_prepass()/
        _schedule_voxel_sim() above); the result — or an exception, since
        a background call can no longer let one simply propagate to the
        caller the way this method used to — is delivered back via
        _compile_done, picked up by _on_compile_done() below, which does
        everything that DOES need the GUI thread (the GL path upload,
        tool/time state, and scheduling the voxel/prepass work that reads
        _last_program) and finally emits file_ready/load_failed so
        MachinePage knows the file is actually usable now, not just
        requested — see that page's _file_ready flag.
        """
        self._teardown_voxel_sim()
        # A new program invalidates any in-flight/completed pre-pass for
        # the OLD one outright — bump the generation and drop the stale
        # result now so nothing between here and the fresh _start_prepass()
        # call (from _on_compile_done()) could read it as still valid.
        self._prepass_generation += 1
        self._prepass_abort.set()
        self._prepass_result = None

        self._compile_generation += 1
        self._compile_abort.set()
        gen = self._compile_generation
        my_abort = threading.Event()
        self._compile_abort = my_abort
        self._pending_load_path = path

        s = AppSettings.instance()
        # The program's first move is implicitly a rapid FROM this position
        # (see GCodeCompiler.load_file()'s docstring) — work origin + a safe
        # Z clearance, not the work origin itself, so that rapid doesn't
        # read as a false-positive plunge into the stock right at tick 1.
        start_position = np.array([0.0, 0.0, s.start_safe_z_mm], dtype='f4')

        def _worker() -> None:
            try:
                program = self.compiler.load_file(path, start_position=start_position)
                # Resolve/create this file's Workpiece record — see
                # persistence.workpiece_db.WorkpieceDatabase and
                # domain.models.Workpiece's collision_detection_enabled
                # docstring. On a first-ever-seen path this does a real
                # sha256 file hash + sqlite write — safe off the GUI
                # thread since WorkpieceDatabase's connection is opened
                # with check_same_thread=False (persistence/workpiece_db.py).
                workpiece = WorkpieceDatabase.instance().get_or_create_by_path(path)
                error = None
            except Exception as exc:  # noqa: BLE001 — reported via load_failed, not swallowed
                program, workpiece, error = None, None, exc
            if not my_abort.is_set():
                self._compile_done.emit(gen, program, workpiece, error)

        threading.Thread(target=_worker, daemon=True, name="gcode-compile").start()

    def _on_compile_done(self, gen: int, program, workpiece, error) -> None:
        """Slot for _compile_done — GUI thread. Drops a stale generation's
        result exactly like _on_prepass_done()/_on_voxel_material_ready()
        (a newer set_file() call, or a teardown, superseded it)."""
        if gen != self._compile_generation:
            return
        if error is not None:
            self.load_failed.emit(error)
            return

        self._clean_lines          = program.clean_lines
        self._raw_lines            = program.raw_lines   # for collision log line text
        self._player               = SimulationPlayer(program.path)
        self._tool_changes         = program.tool_changes
        self._last_tool_change_idx = -1
        self._last_program         = program   # kept for voxel re-init
        self._current_workpiece    = workpiece

        # ---- Everything below is GL-bound or reads _last_program/
        # _current_workpiece — stays on the GUI thread, same calls as
        # before this method existed, just running from here instead of
        # inline in set_file(). ----
        self.viewport.set_path(program.path)

        s = AppSettings.instance()
        self.set_tool_mode(self._TOOL_MAP.get(s.tool_mode, ToolMode.CYLINDER))
        self.set_path_mode(self._PATH_MAP.get(s.path_mode, PathMode.PROGRESSIVE))

        self._apply_initial_tool()
        self._recompute_part_time()

        if _VOXEL_AVAILABLE and s.voxel_enabled:
            self._schedule_voxel_sim()

        # Trigger a) — background, whole-program collision pre-pass. Runs
        # independently of voxel-sim/GL readiness (it works against a
        # disposable numpy scratch stock, see prepass.run_prepass()); the
        # result is applied to _voxel_ctrl (now or once it's created) via
        # _on_prepass_done()/_create_voxel_sim().
        self._start_prepass()

        self.file_ready.emit(self._pending_load_path)

    # ── Collision pre-pass lifecycle ──────────────────────────────────────────

    def _start_prepass(
        self,
        on_done: "Callable[[CollisionPrepassResult | None], None] | None" = None,
    ) -> None:
        """(Re)start the background, whole-program collision pre-pass.

        Always aborts whatever pre-pass is currently in flight and starts a
        fresh one — see class-level note on the three triggers (load,
        Start-click, parameter change) in the plan this implements. Cheap
        to over-call: an abort of an already-finished worker is a no-op,
        and a worker that's only just starting notices the abort at its
        next _ABORT_CHECK_STRIDE checkpoint (prepass.py) and exits quickly.

        Race safety: each call captures its own generation number
        (self._prepass_generation, incremented here) and binds a FRESH
        threading.Event to the new worker — never the previous one — so an
        older worker's belated completion can never be mistaken for this
        call's abort signal. _on_prepass_done() (the delivery slot) drops
        any result whose generation no longer matches
        self._prepass_generation, which is what actually makes a stale
        worker's result harmless even in the (normally impossible, since
        each worker gets its own Event) case both are somehow still
        running.

        Skips the scan entirely (calls on_done(None) synchronously) if
        there's no program loaded or collision detection is effectively
        disabled — same short-circuit presim_check_collisions() always had.
        """
        self._prepass_generation += 1
        gen = self._prepass_generation
        self._prepass_abort.set()

        if (
            not _VOXEL_AVAILABLE
            or self._last_program is None
            or len(self._last_program.path.points) < 2
            or not self._effective_collision_enabled()
        ):
            self._prepass_result = None
            if self._voxel_ctrl is not None:
                self._voxel_ctrl.set_prepass(None)
            if on_done:
                on_done(None)
            return

        s = AppSettings.instance()
        program = self._last_program
        path = program.path
        try:
            shape = StockShape(s.stock_shape)
        except ValueError:
            shape = StockShape.BOUNDING_BOX
        stock = StockDefinition(
            shape=shape, voxel_size=s.voxel_size, z_offset_mm=s.stock_z_offset_mm,
            height_mm=s.stock_height_mm, round_radius_mm=s.stock_round_radius_mm,
            width_mm=s.stock_width_mm, depth_mm=s.stock_depth_mm,
            x_offset_mm=s.stock_x_offset_mm, y_offset_mm=s.stock_y_offset_mm,
        )
        stock.build_bbox(path)

        tool_changes = program.tool_changes
        initial_tool = (
            get_tool_by_pocket(tool_changes[0].pocket_number) if tool_changes
            else get_tool_by_pocket(1)
        )
        db = ToolDatabase.instance()
        backend = s.sim_backend

        my_abort = threading.Event()
        self._prepass_abort = my_abort
        self._pending_prepass_cb = on_done

        def _worker() -> None:
            result = run_prepass(
                path, tool_changes, initial_tool, stock,
                get_tool_by_pocket=get_tool_by_pocket, get_holder=db.get_holder, abort=my_abort,
                backend=backend,
            )
            if not my_abort.is_set():
                self._prepass_done.emit(gen, result)

        threading.Thread(
            target=_worker, daemon=True, name="collision-prepass",
        ).start()

    def _on_prepass_done(self, gen: int, result: "CollisionPrepassResult | None") -> None:
        """Slot for _prepass_done — runs on the GUI thread (Qt queues the
        cross-thread emit). Drops the result outright if a newer trigger
        has already superseded it (see _start_prepass()'s docstring)."""
        if gen != self._prepass_generation:
            return
        self._prepass_result = result
        if self._voxel_ctrl is not None:
            self._voxel_ctrl.set_prepass(result)
        cb, self._pending_prepass_cb = self._pending_prepass_cb, None
        if cb is not None:
            cb(result)

    def _invalidate_prepass(self, *_args) -> None:
        """A parameter that affects collision geometry changed (voxel size,
        stock shape/dimensions, collision-detection re-enabled) — drop the
        now-stale table immediately (so on_tick() reads no hits rather than
        wrong ones in the meantime) and kick off a fresh background
        recompute."""
        self._prepass_result = None
        if self._voxel_ctrl is not None:
            self._voxel_ctrl.set_prepass(None)
        self._collision_active = False
        self.viewport.set_collision(False)
        self.control_hub.clear_collision()
        self._start_prepass()

    # ── Voxel sim lifecycle ───────────────────────────────────────────────────

    def _schedule_voxel_sim(self) -> None:
        """(Re)start the voxel grid's CPU-side build in the background,
        then hand the GL-dependent finish (_create_voxel_sim()) to
        _on_voxel_material_ready() once it completes.

        Building the material array (solid_material() — a numpy
        allocation across the WHOLE stock volume, potentially tens of
        millions of voxels for a large part at a fine voxel_size) used to
        run synchronously right here, on whatever thread called
        set_file() — i.e. the GUI thread, blocking the entire UI for the
        duration on every single file load ("Ausführen" included). It now
        only ever runs in a background thread; this method itself does
        nothing heavier than build_bbox() (a single vectorized min/max
        pass over the path, cheap even for long programs) before handing
        off.

        generation/abort follow the exact same pattern as
        _start_prepass()'s: a bumped generation number plus a fresh
        Event bound to this call only, so a stale worker's belated result
        (a superseding load, settings change, or teardown arrived while
        it was still running) is dropped by _on_voxel_material_ready()
        rather than silently overwriting a newer state.
        """
        self._voxel_build_generation += 1
        self._voxel_build_abort.set()

        if (
            not _VOXEL_AVAILABLE
            or self._last_program is None
            or not AppSettings.instance().voxel_enabled
        ):
            return

        gen = self._voxel_build_generation
        my_abort = threading.Event()
        self._voxel_build_abort = my_abort

        s = AppSettings.instance()
        try:
            shape = StockShape(s.stock_shape)
        except ValueError:
            shape = StockShape.BOUNDING_BOX

        stock = StockDefinition(
            shape           = shape,
            voxel_size      = s.voxel_size,
            z_offset_mm     = s.stock_z_offset_mm,
            height_mm       = s.stock_height_mm,
            round_radius_mm = s.stock_round_radius_mm,
            width_mm        = s.stock_width_mm,
            depth_mm        = s.stock_depth_mm,
            x_offset_mm     = s.stock_x_offset_mm,
            y_offset_mm     = s.stock_y_offset_mm,
        )
        stock.build_bbox(self._last_program.path)
        # The "material" handed over is the backend's CPU stock: a uint8
        # voxel array or a DexelStock — _create_voxel_sim() dispatches on it.
        build = solid_dexels if s.sim_backend == "dexel" else solid_material

        def _worker() -> None:
            material = build(stock)
            if not my_abort.is_set():
                self._voxel_material_ready.emit(gen, stock, material)

        threading.Thread(
            target=_worker, daemon=True, name="voxel-material-build",
        ).start()

    def _on_voxel_material_ready(self, gen: int, stock, material) -> None:
        """Slot for _voxel_material_ready — runs on the GUI thread (Qt
        queues the cross-thread emit). Drops the result outright if a
        newer trigger already superseded it (see _schedule_voxel_sim()'s
        docstring), same guard _on_prepass_done() uses for the prepass."""
        if gen != self._voxel_build_generation:
            return
        self._pending_voxel_stock = stock
        self._pending_voxel_material = material
        if hasattr(self.viewport, "ctx"):
            self._create_voxel_sim()
        else:
            self._pending_voxel_setup = True
            self.viewport.installEventFilter(self)

    def _create_voxel_sim(self) -> None:
        """
        Instantiate GpuVoxelGrid, VoxelCarver, VoxelSimController, VoxelRenderer
        from the already-computed self._pending_voxel_stock/_material (see
        _schedule_voxel_sim()/_on_voxel_material_ready() — the CPU-heavy
        solid_material() build has already happened in a background
        thread by the time this runs). Only GL-dependent work is left
        here: the Texture3D upload and the carver/controller/renderer
        construction.

        Must be called with a ready GL context (viewport.ctx) — either
        because _on_voxel_material_ready() found one immediately, or
        because the eventFilter() below is firing this after the
        viewport's first paint.
        """
        if not _VOXEL_AVAILABLE or self._last_program is None:
            return
        stock = self._pending_voxel_stock
        material = self._pending_voxel_material
        if stock is None or material is None:
            return   # nothing pending — stale eventFilter fire, or already consumed
        self._pending_voxel_stock = None
        self._pending_voxel_material = None

        self.viewport.makeCurrent()
        try:
            checker = None
            run_ends = None
            if isinstance(material, np.ndarray):
                grid   = GpuVoxelGrid(self.viewport.ctx, stock, material=material)
                carver = VoxelCarver(grid)
            else:
                grid   = GpuDexelGrid(self.viewport.ctx, stock, dexels=material)
                carver = DexelCarver(grid.dexels)
                dexels = grid.dexels

                def checker(start, end, tool, is_rapid, holder):
                    return dexel_check_segment(dexels, start, end, tool, is_rapid, holder)

                path = self._last_program.path
                run_ends = dexel_run_ends(path.points, path.feed_rates, path.line_ids)

            active_tool = self._current_tool or get_tool_by_pocket(1)
            self._voxel_ctrl = VoxelSimController(
                grid             = grid,
                carver           = carver,
                path_points      = self._last_program.path.points,
                path_arc_lengths = self._last_program.path.arc_lengths,
                path_feed_rates  = self._last_program.path.feed_rates,
                tool             = active_tool,
                path_line_ids    = self._last_program.path.line_ids,
                checker          = checker,
                run_ends         = run_ends,
            )
            self._voxel_ctrl.set_collision_enabled(self._effective_collision_enabled())
            if active_tool is not None:
                self._voxel_ctrl.update_holder(
                    ToolDatabase.instance().get_holder(active_tool.holder_preset)
                )
            # Apply whatever pre-pass table is currently valid (may still
            # be None — a fresh one is already in flight from set_file(),
            # see _start_prepass(); this just avoids a gap where a
            # rebuild-triggering settings change loses an otherwise still-
            # valid, already-completed result).
            self._voxel_ctrl.set_prepass(self._prepass_result)
            self._collision_active = False

            renderer_cls = VoxelRenderer if isinstance(grid, GpuVoxelGrid) else DexelRenderer
            self._voxel_renderer = renderer_cls(self.viewport.ctx, grid)
        finally:
            self.viewport.doneCurrent()

        self.viewport.set_voxel_renderer(self._voxel_renderer)

    def _teardown_voxel_sim(self) -> None:
        """Stop the controller and remove the renderer from the viewport."""
        # Invalidate any in-flight/completed-but-not-yet-consumed voxel
        # material build outright — bump the generation and drop whatever
        # is pending now, so a background worker that's still running (or
        # one that already finished but hasn't been picked up via
        # _create_voxel_sim() yet, e.g. still waiting on the eventFilter's
        # first paint) can never attach a grid to a program this teardown
        # is meant to get rid of. Needed here specifically (not just at
        # _schedule_voxel_sim()'s own next call) because some callers
        # (_on_voxel_enabled_changed(False)) tear down WITHOUT scheduling
        # a new build afterward.
        self._voxel_build_generation += 1
        self._voxel_build_abort.set()
        self._pending_voxel_stock = None
        self._pending_voxel_material = None
        self._pending_voxel_setup = False

        # Signal the carving thread to abort and wait briefly for it to exit.
        # This prevents the thread from writing to the (about-to-be-freed) grid.
        self._carve_abort.set()
        if self._carve_thread is not None and self._carve_thread.is_alive():
            self._carve_thread.join(timeout=0.3)
        self._carve_abort.clear()
        self._carve_done.clear()
        self._carve_thread = None

        grid = self._voxel_ctrl.grid if self._voxel_ctrl is not None else None
        self._voxel_ctrl = None          # carving stops immediately

        if self._voxel_renderer is not None or grid is not None:
            # Release GPU shader objects and the material texture(s) — the
            # latter can be hundreds of MB at a fine resolution.
            self.viewport.makeCurrent()
            try:
                if self._voxel_renderer is not None:
                    self._voxel_renderer.release()
                if grid is not None:
                    grid.release()
            finally:
                self.viewport.doneCurrent()
            self._voxel_renderer = None

        self.viewport.set_voxel_renderer(None)   # viewport stops rendering it

    def eventFilter(self, obj, event) -> bool:
        """One-shot: create voxel sim after initializeGL fires (first Paint)."""
        from PySide6.QtCore import QEvent
        if (
            obj is self.viewport
            and event.type() == QEvent.Type.Paint
            and self._pending_voxel_setup
        ):
            self._pending_voxel_setup = False
            self.viewport.removeEventFilter(self)
            self._create_voxel_sim()
        return super().eventFilter(obj, event)

    # ── AppSettings callbacks ─────────────────────────────────────────────────

    def _on_voxel_size_changed(self, voxel_size: float) -> None:
        """Voxel size changed → rebuild grid from scratch (no live resize)."""
        if not _VOXEL_AVAILABLE:
            return
        self._teardown_voxel_sim()
        if self._last_program is not None and AppSettings.instance().voxel_enabled:
            self._schedule_voxel_sim()
        # The pre-pass sampled against the old voxel resolution — recompute
        # against the new one (see _invalidate_prepass()).
        self._invalidate_prepass()

    def _on_sim_backend_changed(self, _backend: str) -> None:
        """Voxel ↔ tri-dexel switched → rebuild exactly like a resolution
        change: new stock model, and a pre-pass against that model."""
        self._on_voxel_size_changed(AppSettings.instance().voxel_size)

    def _on_voxel_enabled_changed(self, enabled: bool) -> None:
        """Voxel enable/disable checkbox toggled."""
        if not _VOXEL_AVAILABLE:
            return
        if enabled:
            if self._last_program is not None:
                self._schedule_voxel_sim()
        else:
            self._teardown_voxel_sim()

    def _on_collision_enabled_changed(self, _enabled: bool) -> None:
        # Re-resolve rather than trust the raw global value directly — a
        # per-workpiece override (if one is set) must keep taking
        # precedence over a global-setting change.
        if self._voxel_ctrl is not None:
            self._voxel_ctrl.set_collision_enabled(self._effective_collision_enabled())
        # Toggling detection on/off doesn't change what the pre-pass would
        # find (on_tick() itself gates the lookup on _collision_enabled),
        # EXCEPT that _start_prepass() skips the scan entirely while
        # detection is off (see its docstring) — so turning it back on
        # needs a real (re)compute, not just a flag flip.
        self._invalidate_prepass()

    def _effective_collision_enabled(self) -> bool:
        """AppSettings.collision_detection_enabled, unless the current
        workpiece has an explicit override (see domain.models.Workpiece's
        collision_detection_enabled docstring) — used both for the live
        VoxelSimController and presim_check_collisions()."""
        wp = self._current_workpiece
        if wp is not None and wp.collision_detection_enabled is not None:
            return wp.collision_detection_enabled
        return AppSettings.instance().collision_detection_enabled

    def _on_stock_settings_changed(self, *_args) -> None:
        """Any stock geometry setting changed → tear down and rebuild the sim."""
        if not _VOXEL_AVAILABLE:
            return
        self._teardown_voxel_sim()
        if self._last_program is not None and AppSettings.instance().voxel_enabled:
            self._schedule_voxel_sim()
        # New stock shape/dimensions → the pre-pass's collision geometry is
        # stale (it scanned against the OLD stock) — recompute.
        self._invalidate_prepass()

    # ── Tool management ───────────────────────────────────────────────────────

    def _apply_tool(self, tool: ToolDefinition | None) -> None:
        """Push *tool* (and its resolved holder) to the viewport/voxel
        controller/info pill — but ONLY if it's actually different from
        what's already applied.

        This guard matters a lot more than it looks: _apply_initial_tool()
        (below) is called unconditionally from set_mode(), which itself
        runs on EVERY Start/Stop/Feed-Hold/Reset click (via
        MachinePage._sync_ui_state()) — not just on an actual tool change.
        Without this check, every one of those clicks rebuilt the full 3D
        tool mesh from scratch via viewport.set_tool_definition()/
        set_tool_holder() (see viewport.py's _rebuild_tool_mesh(), called
        TWICE per apply before this fix) even though the tool never
        changes on a Stop/Feed-Hold click at all — measured at ~93ms
        (plain endmill) to ~210ms per rebuild, i.e. ~186-420ms wasted on
        the GUI thread per click. ToolDefinition/HolderProfile are plain
        @dataclass, so value equality (not identity) correctly recognizes
        "the same tool" even across separate ToolDatabase lookups that
        return fresh instances with equal field values.
        """
        if tool is None:
            return
        holder = ToolDatabase.instance().get_holder(tool.holder_preset)
        if tool == self._current_tool and holder == self._last_applied_holder:
            return
        self._current_tool = tool
        self._last_applied_holder = holder
        # One rebuild, not two — see viewport.set_tool() docstring.
        self.viewport.set_tool(tool, holder)
        # Push to the info pill directly rather than waiting for the next
        # SIM-mode _tick() to refresh it — _tick()'s tool-label refresh only
        # ran while self._mode == "SIM", so MACHINE-mode tool changes (via
        # set_line() -> _check_tool_change()) and sim_reset() never reached
        # the display until (if ever) SIM mode ticked again.
        self.control_hub.set_tool(tool.tool_number)
        self.tool_changed.emit(tool)
        if self._voxel_ctrl is not None:
            self._voxel_ctrl.update_tool(tool)
            self._voxel_ctrl.update_holder(holder)

    def _on_tool_db_changed(self, tool_number: int) -> None:
        """ToolDatabaseSignals.tool_changed — only the CURRENTLY ACTIVE
        tool matters here (its geometry, name, holder, or magazine-pocket
        assignment may have just changed on ToolPage); any other tool's
        edit has nothing to refresh until it's actually selected via a
        T-command, which already reads live (see _check_tool_change()).
        Deliberately get_tool(tool_number), NOT get_tool_by_pocket(): this
        re-fetches the SAME already-selected ToolDefinition by its own
        identity to pick up whatever changed on it (its pocket included)
        — it is not re-resolving a T-address."""
        if self._current_tool is None or self._current_tool.tool_number != tool_number:
            return
        self._apply_tool(get_tool(tool_number))

    def _check_tool_change(self, current_line: int) -> None:
        for tc in self._tool_changes:
            if tc.line_index <= current_line and tc.line_index > self._last_tool_change_idx:
                self._last_tool_change_idx = tc.line_index
                self._apply_tool(get_tool_by_pocket(tc.pocket_number))

    def _apply_initial_tool(self) -> None:
        """Apply the program's first tool and reset tool-change tracking.

        Shared by set_file() (new program loaded) and sim_reset() (program
        restarted from the beginning) so the tool info field always shows
        the correct start-of-program tool rather than whatever tool was
        last active before the reload/reset.
        """
        if self._last_program is None:
            return
        first_tool = (
            get_tool_by_pocket(self._last_program.tool_changes[0].pocket_number)
            if self._last_program.tool_changes else get_tool_by_pocket(1)
        )
        self._apply_tool(first_tool)

    # ── Feed/rapid override + part-time estimate ──────────────────────────────

    def set_feed_override(self, value: float) -> None:
        """Push the machine's current feed override (1.0 = 100%) in from
        outside — mirrors the set_position()/set_line() "pushed state"
        style already used for MACHINE mode. Recomputes the displayed
        part-time estimate."""
        self._feed_override = value
        self._recompute_part_time()

    def set_rapid_override(self, value: float) -> None:
        """Push the machine's current rapid override (1.0 = 100%) in from
        outside. Recomputes the displayed part-time estimate."""
        self._rapid_override = value
        self._recompute_part_time()

    def _recompute_part_time(self) -> None:
        if self._last_program is None:
            self.part_time_changed.emit(None)
            return
        seconds = self._last_program.path.estimated_time_s(
            self._feed_override, self._rapid_override,
        )
        self.part_time_changed.emit(seconds)

    # ── Machine interface (SimPlaceholder-compatible API) ─────────────────────

    def set_state(self, state: str) -> None:
        self._state = state

    def set_position(self, x: float, y: float, z: float) -> None:
        pos = np.array([x, y, z], dtype="f4")
        if self._mode == "MACHINE":
            self.viewport.set_tool_position(pos)
            if self._voxel_ctrl is not None:
                if self._last_machine_pos is not None:
                    # Overwrite any pending carve — only the latest endpoint
                    # matters; the worker always uses the newest position.
                    self._pending_machine_carve = (
                        self._last_machine_pos.copy(),
                        pos.copy(),
                    )
                self._last_machine_pos = pos.copy()

    def set_line(self, line: int) -> None:
        self._current_line = line
        self.viewport.set_active_line(line)
        # Keep the tool info field correct during a real machine run too —
        # previously only the SIM-mode _tick() loop applied tool changes,
        # so the info field never moved past the program's first tool while
        # MACHINE mode pushed lines in from the controller.
        self._check_tool_change(line)

    # ── Mode ──────────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        assert mode in ("SIM", "MACHINE"), f"Unknown mode: {mode!r}"
        self._mode = mode
        self.control_hub.set_mode(mode)
        if self._player:
            self._player.reset()
        # Reset machine-mode position tracking on every transition so we never
        # accidentally carve a phantom segment from the last known position.
        self._last_machine_pos      = None
        self._pending_machine_carve = None
        # Reset tool-change tracking so a MACHINE run always re-evaluates T
        # changes from the start of the program, independent of how far SIM
        # playback had already progressed before switching modes.
        self._last_tool_change_idx = -1
        self._apply_initial_tool()

    def set_path_mode(self, mode: PathMode) -> None:
        self._path_mode = mode
        self.viewport.set_path_mode(mode)

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool_mode = mode
        self.viewport.set_tool_mode(mode)

    # ── Playback controls ─────────────────────────────────────────────────────

    def sim_play(self) -> None:
        if self._player:
            self._player.play()

    def sim_pause(self) -> None:
        if self._player:
            self._player.pause()
        self._force_carve_catchup()

    def sim_reset(self) -> None:
        # Stop any in-flight carving thread and WAIT for it to actually
        # exit before touching the grid. A timed-out join here used to drop
        # the Python reference while the OS thread kept running in the
        # background — free to keep calling carve_segment()/grid.carve()
        # against the SAME grid right after _voxel_ctrl.reset() re-solidified
        # it, silently re-carving material back out and making Reset look
        # like it hadn't cleared the stock. The worker loop (_carve_worker)
        # already checks _carve_abort at the top of every 5mm chunk, so a
        # plain join() here only ever waits out however long the chunk that
        # was already in flight takes to finish — milliseconds in practice.
        self._carve_abort.set()
        if self._carve_thread is not None and self._carve_thread.is_alive():
            self._carve_thread.join()
        self._carve_abort.clear()
        self._carve_done.clear()
        self._carve_thread = None

        if self._player:
            self._player.reset()
            self._last_tool_change_idx = -1
            self._apply_initial_tool()
            self.control_hub.reset_play_state()
        if self._voxel_ctrl is not None:
            # VoxelSimController.reset() -> GpuVoxelGrid.reset() does its own
            # immediate, full self._tex.write(...) (not the incremental
            # upload_if_dirty() path) and then clears the dirty flag itself
            # — so the GL context must already be current for THIS call, not
            # after it. It used to be called before makeCurrent(), which
            # made that write execute with no context bound (a no-op on
            # most drivers rather than a crash); reset() then leaving
            # `_dirty = False` meant the upload_if_dirty() call that
            # followed had nothing left to do either, so the CPU-side grid
            # went back to solid but the actually-rendered GPU texture
            # never did — Reset looked like it hadn't cleared the stock.
            self.viewport.makeCurrent()
            try:
                self._voxel_ctrl.reset()   # also clears collision_hit — see its docstring
            finally:
                self.viewport.doneCurrent()
        self._collision_active = False
        self.viewport.set_collision(False)
        self.control_hub.clear_collision()

    def sim_seek(self, fraction: float) -> None:
        if self._player:
            self._player.seek(fraction)

    def seek_to_point(self, point: np.ndarray) -> None:
        """Seek playback to the path position closest to *point* (world
        mm) — used to jump the sim view to a collision location for
        inspection after the user cancels a pre-flight-blocked Start."""
        if self._player is None:
            return
        path = self._player._path
        if path.total_length < 1e-9:
            return
        s, _line = path.find_nearest(np.asarray(point, dtype="f4"))
        self._player.seek(s / path.total_length)

    def sim_set_speed(self, speed: float) -> None:
        if self._player:
            self._player.speed_scale = speed

    def presim_to_s(
        self,
        target_s: float,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """Fast-forward the voxel grid to *target_s* in a background thread.

        Intended for "start program from line X" — pre-carves all material
        that would have been removed before that line, so the stock state is
        correct when the program starts.

        Usage::

            s = path.arc[np.searchsorted(path.line_ids, start_line)]
            widget.presim_to_s(s, on_done=lambda: start_machine())

        The *on_done* callback is invoked on the main thread via
        ``QTimer.singleShot(0, ...)`` when the background thread completes.
        """
        from PySide6.QtCore import QTimer

        if self._voxel_ctrl is None:
            if on_done:
                on_done()
            return

        # Abort any running carve first
        self._carve_abort.set()
        if self._carve_thread is not None and self._carve_thread.is_alive():
            self._carve_thread.join(timeout=0.5)
        self._carve_abort.clear()
        self._carve_done.clear()

        ctrl = self._voxel_ctrl

        def _worker() -> None:
            # Single call — fastest path, no chunking needed for a one-shot
            # fast-forward.  The UI shows a "preparing" state during this.
            ctrl.on_tick(target_s)
            self._carve_done.set()
            if on_done and not self._carve_abort.is_set():
                QTimer.singleShot(0, on_done)

        self._carve_thread = threading.Thread(
            target=_worker, daemon=True, name="presim"
        )
        self._carve_thread.start()

    @property
    def tool_changes(self) -> list:
        """The currently loaded program's T-command list (compiler.py's
        ToolChange, one per distinct tool change) — [] if nothing is
        loaded. Public so MachinePage can run a pre-start tool-
        availability check (see gcode/compiler.py's validate_tools()) —
        everything else that reads tool changes does so internally via
        the private _tool_changes this mirrors."""
        return self._tool_changes

    def presim_check_collisions(
        self,
        on_done: Callable[["CollisionHit | None"], None],
    ) -> None:
        """Collision pre-flight check ("Vorab-Check") for the Start button —
        name/signature intentionally unchanged (MachinePage._on_start_clicked
        calls this exactly as before) even though the underlying mechanism
        is now the shared collision pre-pass (see _start_prepass()) rather
        than a dedicated one-shot scan: it answers with the first hit (by
        line number) from a COMPLETE table that then also serves the
        subsequent SIM playback, instead of throwing away a scan that
        stopped at the first hit.

        Answers immediately, synchronously, from the already-completed
        table if one is valid (the common case — trigger a) already ran
        this at load time); only actually (re)computes if none is
        available yet, e.g. a parameter changed since the last completed
        pre-pass and the recompute hasn't landed yet. Calls on_done(None)
        outright (no scan) if there's no program loaded or collision
        detection is effectively disabled — same short-circuit as before.
        """
        if self._prepass_result is not None:
            on_done(self._prepass_result.first_hit)
            return
        self._start_prepass(
            on_done=lambda result: on_done(result.first_hit if result is not None else None)
        )

    # ── ControlHub wiring ─────────────────────────────────────────────────────

    def _connect_control_hub(self) -> None:
        self.control_hub.play_clicked.connect(self.sim_play)
        self.control_hub.pause_clicked.connect(self.sim_pause)
        self.control_hub.stop_clicked.connect(self.sim_reset)
        self.control_hub.speed_changed.connect(self.sim_set_speed)
        self.control_hub.skip_forward_clicked.connect(
            lambda: self.sim_seek(min(self._player.progress() + 0.05, 1.0))
            if self._player else None
        )
        self.control_hub.skip_backward_clicked.connect(
            lambda: self.sim_seek(max(self._player.progress() - 0.05, 0.0))
            if self._player else None
        )

    # ── Render tick ───────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Called ~30 fps.  Advances the player and drives voxel carving."""
        if self._mode == "SIM":
            if self._player is None:
                self.viewport.update()
                return

            pos, line, s = self._player.tick()

            if self._player.is_finished or self._player.is_at_start:
                self._on_sim_finished()

            self._check_tool_change(line)

            self.viewport.set_tool_position(pos)
            self.viewport.set_active_line(line)
            self.viewport.set_progress(s)

            if self._clean_lines and 0 <= line < len(self._clean_lines):
                self.control_hub.set_gcode(f"({line}) {self._clean_lines[line]}")

            feed = self._player._path.feed_at(s)
            self.control_hub.set_feedrate(feed)
            self.control_hub.set_datum(getattr(self.viewport, "_active_wcs", 1))
            # (tool pill is now pushed directly from _apply_tool(), not
            # refreshed here — see its docstring)

            # Voxel carving: small backlogs are carved synchronously right
            # here so the tool marker and the carved material are always for
            # the same `s`; only large backlogs (seeks/skips) go through the
            # background worker so the UI thread never stalls. See
            # _drive_voxel_carve()'s docstring.
            if self._voxel_ctrl is not None:
                ctrl = self._voxel_ctrl   # capture; teardown may null self._voxel_ctrl
                self._drive_voxel_carve(ctrl, s)

        elif self._mode == "MACHINE" and self._voxel_ctrl is not None:
            # MACHINE mode: carve from real machine positions delivered via
            # set_position().  Uses the same _carve_done / _carve_abort events
            # so _teardown_voxel_sim() correctly aborts both worker types.
            if self._carve_done.is_set():
                self._carve_done.clear()
                self.viewport.makeCurrent()
                self._voxel_ctrl.grid.upload_if_dirty()
                self.viewport.doneCurrent()

            if (
                self._pending_machine_carve is not None
                and (self._carve_thread is None or not self._carve_thread.is_alive())
            ):
                start, end = self._pending_machine_carve
                self._pending_machine_carve = None
                tool = self._current_tool
                if tool is not None:
                    ctrl = self._voxel_ctrl
                    self._carve_thread = threading.Thread(
                        target=self._machine_carve_worker,
                        args=(ctrl, start, end, tool),
                        daemon=True,
                        name="machine-carver",
                    )
                    self._carve_thread.start()

        self._handle_collision_state()
        self.viewport.update()

    def _handle_collision_state(self) -> None:
        """Non-blocking collision feedback, called unconditionally every
        tick — NEVER pauses or stops the simulation (see collision.py /
        VoxelSimController's docstrings: a hit is diagnostic here, not a
        safety interlock).

        Two independent things happen:
        - Every NEW hit (ctrl.drain_new_hits()) is logged exactly once,
          with its G-code line number and source text, and still emits the
          public collision_detected signal — MachinePage's live-machine
          handler reacts to that only while an actual MACHINE-mode program
          is RUNNING, so this firing repeatedly during plain SIM playback
          is harmless (see its docstring; unchanged by this rewrite).
        - The viewport tool tint / control_hub pill are synced to whether
          the tool is IN a collision RIGHT NOW (ctrl.collision_hit, live —
          clears itself the moment a clean segment follows), so the red
          highlight lasts exactly as long as the collision itself.
        """
        ctrl = self._voxel_ctrl
        if ctrl is None:
            return

        for hit in ctrl.drain_new_hits():
            self._log_collision(hit)
            self.collision_detected.emit(hit)

        active = ctrl.collision_hit is not None
        if active == self._collision_active:
            return
        self._collision_active = active
        if active:
            hit = ctrl.collision_hit
            self.viewport.set_collision(True, hit.point)
            self.control_hub.set_collision(hit)
        else:
            self.viewport.set_collision(False)
            self.control_hub.clear_collision()

    def _log_collision(self, hit: "CollisionHit") -> None:
        line_text = ""
        if 0 <= hit.line_number < len(self._raw_lines):
            line_text = self._raw_lines[hit.line_number].strip()
        logger.warning(
            "Kollision erkannt: Zeile %d (%s): %s",
            hit.line_number, hit.kind, line_text or "?",
        )

    def _drive_voxel_carve(self, ctrl: "VoxelSimController", s: float) -> None:
        """Advance voxel carving to *s*, keeping the displayed tool position
        and the carved material provably in sync for normal playback.

        Small backlogs (`s - ctrl.max_s`) are carved synchronously, in this
        call, on the calling (Qt main/GL) thread — so the grid upload below
        always reflects the exact same `s` the tool marker was just set to
        this frame. Large backlogs (a seek/skip jump, or any unexpectedly
        heavy stretch of path) fall back to the existing chunked background
        worker so the UI thread is never blocked for long. The two paths are
        mutually exclusive by construction: a new synchronous or background
        pass is only ever started once any previous background worker has
        fully finished (`is_alive()` is False only after its target callable
        has returned), so on_tick()/carve_segment() are never called from two
        threads at once against the same controller/grid.
        """
        # Flush anything a still/previously-running background worker produced.
        if self._carve_done.is_set():
            self._carve_done.clear()
            self.viewport.makeCurrent()
            ctrl.grid.upload_if_dirty()
            self.viewport.doneCurrent()

        if self._carve_thread is not None and self._carve_thread.is_alive():
            return  # background catch-up still running; re-check next tick

        backlog = s - ctrl.max_s
        if backlog <= 0.0:
            return  # rewinding, or already caught up

        if backlog <= _SYNC_CARVE_MAX_MM:
            if ctrl.on_tick(s):
                self.viewport.makeCurrent()
                ctrl.grid.upload_if_dirty()
                self.viewport.doneCurrent()
        else:
            self._carve_thread = threading.Thread(
                target=self._carve_worker,
                args=(ctrl, s),
                daemon=True,
                name="voxel-carver",
            )
            self._carve_thread.start()

    def _force_carve_catchup(self) -> None:
        """Guarantee the voxel state is fully caught up to the player's
        current position right now — call whenever the tool visibly stops
        (paused, or playback naturally finished/reversed to start).

        If a background worker is still chewing through a large backlog at
        that exact moment, wait for it (bounded) rather than leaving it to
        trickle across future frames, then run one final synchronous
        on_tick() pass for whatever (if anything) remains.
        """
        if self._voxel_ctrl is None or self._player is None:
            return
        ctrl = self._voxel_ctrl

        if self._carve_thread is not None and self._carve_thread.is_alive():
            # Bounded: background chunks are 5mm each (_carve_worker's
            # CHUNK_MM), so this returns well under the timeout in all but
            # truly pathological cases.
            self._carve_thread.join(timeout=2.0)

        if self._carve_done.is_set():
            self._carve_done.clear()
            self.viewport.makeCurrent()
            ctrl.grid.upload_if_dirty()
            self.viewport.doneCurrent()

        # Only safe to carve synchronously if no thread is still alive (the
        # join above could in principle time out on a pathological backlog).
        if self._carve_thread is None or not self._carve_thread.is_alive():
            if ctrl.on_tick(self._player.current_s()):
                self.viewport.makeCurrent()
                ctrl.grid.upload_if_dirty()
                self.viewport.doneCurrent()

    def _carve_worker(self, ctrl: "VoxelSimController", s_target: float) -> None:
        """Background thread: advance HWM carving to *s_target* in 5 mm chunks.

        Chunking lets the main thread upload and display partial results while
        the worker is still running, eliminating the visual lag between tool
        position and carved material at high playback speeds (5×/10×).
        """
        CHUNK_MM = 5.0
        s = ctrl.max_s
        while s < s_target and not self._carve_abort.is_set():
            s_next = min(s + CHUNK_MM, s_target)
            carved = ctrl.on_tick(s_next)
            if carved and not self._carve_abort.is_set():
                self._carve_done.set()
            s = s_next

    def _machine_carve_worker(
        self,
        ctrl:  "VoxelSimController",
        start: "np.ndarray",
        end:   "np.ndarray",
        tool:  "ToolDefinition",
    ) -> None:
        """Background thread: carve one real-machine move segment.

        Called in MACHINE mode with consecutive position pairs delivered by
        the machine controller.  No arc-length concept — uses carve_move()
        directly.
        """
        if self._carve_abort.is_set():
            return
        # feed_rate=1.0 means "it is a cutting move" — G0 rapid filtering will
        # be added when MachineController exposes the current feed rate.
        # Same limitation applies to collision checking here: every MACHINE-
        # mode move is checked with is_rapid=False (shank/holder-only), so a
        # real rapid that plows into material isn't caught by this path —
        # only by the pre-flight whole-program check before Start, which
        # tessellates from the loaded file and knows real G0/G1 moves apart.
        carved = ctrl.carve_move(
            start, end, tool, feed_rate=1.0, line_number=self._current_line,
        )
        if carved and not self._carve_abort.is_set():
            self._carve_done.set()

    def _on_sim_finished(self) -> None:
        self.control_hub.reset_play_state()
        self._force_carve_catchup()

    # ── Overlay layout ────────────────────────────────────────────────────────

    def _layout_overlays(self) -> None:
        W, H = self.width(), self.height()
        self.viewport.setGeometry(0, 0, W, H)

        sw = self.settings.width()
        self.settings.setGeometry(W - sw, 0, sw, H)

        cw, ch = self.control_hub.width(), self.control_hub.height()
        self.control_hub.setGeometry((W - cw) // 2, H - ch - 16, cw, ch)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._layout_overlays()
