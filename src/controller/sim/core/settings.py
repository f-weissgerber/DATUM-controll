"""sim/core/settings.py — Per-user simulation settings, persisted via QSettings."""

from PySide6.QtCore import QSettings, QObject, Signal


class AppSettings(QObject):
    """
    Singleton. Settings are written to disk immediately on change.
    Signals fire when a value changes so the UI can react.

    Usage:
        from controller.sim.core.settings import AppSettings
        s = AppSettings.instance()
        s.zoom_speed        # read
        s.zoom_speed = 2.0  # write + auto-save
    """

    # Signals ──────────────────────────────────────────────────────────────────
    bg_color_changed        = Signal(str)
    bg_color_2_changed = Signal(str)

    zoom_speed_changed      = Signal(float)
    rotate_speed_changed    = Signal(float)
    pan_speed_changed       = Signal(float)

    invert_zoom_changed     = Signal(bool)
    invert_rotate_x_changed = Signal(bool)
    invert_rotate_y_changed = Signal(bool)
    invert_pan_x_changed    = Signal(bool)
    invert_pan_y_changed    = Signal(bool)
    show_grid_changed = Signal(bool)

    show_axes_changed         = Signal(bool)
    show_datum_symbol_changed = Signal(bool)

    show_gcode_line_changed = Signal(bool)
    show_datum_changed = Signal(bool)
    show_tool_changed = Signal(bool)
    show_feedrate_changed = Signal(bool)

    tool_mode_changed = Signal(str)
    path_mode_changed = Signal(str)

    voxel_size_changed    = Signal(float)
    voxel_enabled_changed = Signal(bool)
    voxel_color_changed   = Signal(str)
    sim_backend_changed   = Signal(str)

    tool_cutting_color_changed = Signal(str)

    show_tool_holder_changed = Signal(bool)
    collision_detection_enabled_changed = Signal(bool)
    collision_check_before_start_enabled_changed = Signal(bool)

    tool_db_path_changed             = Signal(str)
    workpiece_db_path_changed        = Signal(str)
    linuxcnc_tool_table_path_changed = Signal(str)
    workpieces_root_path_changed     = Signal(str)
    workpieces_explorer_root_path_changed = Signal(str)

    # Stock-shape signals (any change requires a sim rebuild)
    stock_shape_changed         = Signal(str)
    stock_z_offset_changed      = Signal(float)
    stock_height_changed        = Signal(float)
    stock_round_radius_changed  = Signal(float)
    stock_width_changed         = Signal(float)
    stock_depth_changed         = Signal(float)
    stock_x_offset_changed      = Signal(float)
    stock_y_offset_changed      = Signal(float)

    start_safe_z_mm_changed = Signal(float)
    tool_pocket_count_changed = Signal(int)
    has_rotary_axes_changed = Signal(bool)

    jog_feed_velocity_mm_s_changed  = Signal(float)
    jog_rapid_velocity_mm_s_changed = Signal(float)

    # ── Voxel colour presets ───────────────────────────────────────────────────
    VOXEL_COLORS: dict[str, tuple[float, float, float]] = {
        "Orange-Gelb":  (0.95, 0.68, 0.12),
        "Koralle":      (0.92, 0.38, 0.25),
        "Hellblau":     (0.30, 0.62, 0.92),
        "Mintgrün":     (0.22, 0.78, 0.55),
        "Violett":      (0.58, 0.35, 0.85),
        "Aluminium":    (0.72, 0.65, 0.56),
        "Stahl":        (0.55, 0.57, 0.60),
        "Holz":         (0.76, 0.60, 0.35),
    }

    # ── Tool cutting-edge colour presets ───────────────────────────────────────
    TOOL_COLORS: dict[str, tuple[float, float, float]] = {
        "Gold (Standard)": (1.00, 0.84, 0.00),
        "Silber":          (0.75, 0.75, 0.78),
        "Kupfer":          (0.85, 0.55, 0.35),
        "Blau":            (0.25, 0.55, 0.95),
        "Rot":             (0.90, 0.25, 0.25),
        "Grün":            (0.35, 0.80, 0.35),
    }

    # Singleton ────────────────────────────────────────────────────────────────
    _instance: "AppSettings | None" = None

    @classmethod
    def instance(cls) -> "AppSettings":
        if cls._instance is None:
            cls._instance = AppSettings()
        return cls._instance

    def __init__(self):
        super().__init__()
        # Keys stay "DatumSim"/"DatumSim" so user prefs survive upgrades
        self._qs = QSettings("DatumSim", "DatumSim")

    # Named background gradient presets: name -> (inner_hex, outer_hex).
    # Selected as a whole combo via apply_bg_theme() — no free colour picker.
    BG_COLORS: dict[str, tuple[str, str]] = {
        "Dark (Standard)": ("#1c1c1c", "#161b22"),
        "Black":           ("#000000", "#000000"),
        "Dark Grey":       ("#2d2d2d", "#232323"),
        "Grey":            ("#4a4a4a", "#3a3a3a"),
        "Dark Blue":       ("#0d1b2a", "#0a141f"),
        "DATUM":           ("#232a35", "#1a1f28"),
    }

    @property
    def bg_color(self) -> str:
        return self._qs.value("camera/bg_color", "#1c1c1c", type=str)

    @bg_color.setter
    def bg_color(self, v: str):
        self._qs.setValue("camera/bg_color", v)
        self.bg_color_changed.emit(v)

    @property
    def bg_color_2(self) -> str:
        return self._qs.value("camera/bg_color_2", "#161b22", type=str)

    @bg_color_2.setter
    def bg_color_2(self, v: str):
        self._qs.setValue("camera/bg_color_2", v)
        self.bg_color_2_changed.emit(v)

    def apply_bg_theme(self, name: str) -> None:
        """Set bg_color/bg_color_2 together from a named BG_COLORS preset.

        The UI only ever offers these fixed combinations, never a free
        colour picker — selecting one writes both existing hex properties,
        so nothing downstream (Viewport's bg_color_changed/bg_color_2_changed
        listeners) needs to change.
        """
        inner, outer = self.BG_COLORS.get(
            name, self.BG_COLORS["Dark (Standard)"]
        )
        self.bg_color = inner
        self.bg_color_2 = outer

    def bg_theme_name(self) -> str:
        """Best-effort reverse lookup: preset name matching the current
        bg_color/bg_color_2 pair, or the default preset's name if the
        stored colours don't match any known preset (e.g. pre-upgrade
        QSettings state)."""
        current = (self.bg_color, self.bg_color_2)
        for name, pair in self.BG_COLORS.items():
            if pair == current:
                return name
        return "Dark (Standard)"

    @property
    def zoom_speed(self) -> float:
        return self._qs.value("camera/zoom_speed", 1.0, type=float)

    @zoom_speed.setter
    def zoom_speed(self, v: float):
        self._qs.setValue("camera/zoom_speed", v)
        self.zoom_speed_changed.emit(v)

    @property
    def rotate_speed(self) -> float:
        return self._qs.value("camera/rotate_speed", 1.0, type=float)

    @rotate_speed.setter
    def rotate_speed(self, v: float):
        self._qs.setValue("camera/rotate_speed", v)
        self.rotate_speed_changed.emit(v)

    @property
    def pan_speed(self) -> float:
        return self._qs.value("camera/pan_speed", 1.0, type=float)

    @pan_speed.setter
    def pan_speed(self, v: float):
        self._qs.setValue("camera/pan_speed", v)
        self.pan_speed_changed.emit(v)

    @property
    def invert_zoom(self) -> bool:
        return self._qs.value("camera/invert_zoom", False, type=bool)

    @invert_zoom.setter
    def invert_zoom(self, v: bool):
        self._qs.setValue("camera/invert_zoom", v)
        self.invert_zoom_changed.emit(v)

    @property
    def invert_rotate_x(self) -> bool:
        return self._qs.value("camera/invert_rotate_x", False, type=bool)

    @invert_rotate_x.setter
    def invert_rotate_x(self, v: bool):
        self._qs.setValue("camera/invert_rotate_x", v)
        self.invert_rotate_x_changed.emit(v)

    @property
    def invert_rotate_y(self) -> bool:
        return self._qs.value("camera/invert_rotate_y", False, type=bool)

    @invert_rotate_y.setter
    def invert_rotate_y(self, v: bool):
        self._qs.setValue("camera/invert_rotate_y", v)
        self.invert_rotate_y_changed.emit(v)

    @property
    def invert_pan_x(self) -> bool:
        return self._qs.value("camera/invert_pan_x", False, type=bool)

    @invert_pan_x.setter
    def invert_pan_x(self, v: bool):
        self._qs.setValue("camera/invert_pan_x", v)
        self.invert_pan_x_changed.emit(v)

    @property
    def invert_pan_y(self) -> bool:
        return self._qs.value("camera/invert_pan_y", False, type=bool)

    @invert_pan_y.setter
    def invert_pan_y(self, v: bool):
        self._qs.setValue("camera/invert_pan_y", v)
        self.invert_pan_y_changed.emit(v)

    @property
    def show_grid(self) -> bool:
        return self._qs.value("camera/show_grid", True, type=bool)

    @show_grid.setter
    def show_grid(self, v: bool):
        self._qs.setValue("camera/show_grid", v)
        self.show_grid_changed.emit(v)

    @property
    def show_axes(self) -> bool:
        """Whether to render the XYZ axis lines. Default True."""
        return self._qs.value("camera/show_axes", True, type=bool)

    @show_axes.setter
    def show_axes(self, v: bool) -> None:
        self._qs.setValue("camera/show_axes", v)
        self.show_axes_changed.emit(v)

    @property
    def show_datum_symbol(self) -> bool:
        """Whether to render the datum origin symbol (quarter-circle). Default True."""
        return self._qs.value("camera/show_datum_symbol", True, type=bool)

    @show_datum_symbol.setter
    def show_datum_symbol(self, v: bool) -> None:
        self._qs.setValue("camera/show_datum_symbol", v)
        self.show_datum_symbol_changed.emit(v)

    @property
    def tool_mode(self) -> str:
        return self._qs.value("sim/tool_mode", "ENDMILL", type=str)

    @tool_mode.setter
    def tool_mode(self, v: str):
        self._qs.setValue("sim/tool_mode", v)
        self.tool_mode_changed.emit(v)

    @property
    def path_mode(self) -> str:
        return self._qs.value("sim/path_mode", "Full", type=str)

    @path_mode.setter
    def path_mode(self, v: str):
        self._qs.setValue("sim/path_mode", v)
        self.path_mode_changed.emit(v)

    @property
    def show_gcode_line(self) -> bool:
        return self._qs.value("controlhub/show_gcode_line", True, type=bool)

    @show_gcode_line.setter
    def show_gcode_line(self, v: bool):
        self._qs.setValue("controlhub/show_gcode_line", v)
        self.show_gcode_line_changed.emit(v)

    @property
    def show_datum(self) -> bool:
        return self._qs.value("controlhub/show_datum", True, type=bool)

    @show_datum.setter
    def show_datum(self, v: bool):
        self._qs.setValue("controlhub/show_datum", v)
        self.show_datum_changed.emit(v)

    @property
    def show_tool(self) -> bool:
        return self._qs.value("controlhub/show_tool", True, type=bool)

    @show_tool.setter
    def show_tool(self, v: bool):
        self._qs.setValue("controlhub/show_tool", v)
        self.show_tool_changed.emit(v)

    @property
    def show_feedrate(self) -> bool:
        return self._qs.value("controlhub/show_feedrate", True, type=bool)

    @show_feedrate.setter
    def show_feedrate(self, v: bool):
        self._qs.setValue("controlhub/show_feedrate", v)
        self.show_feedrate_changed.emit(v)

    @property
    def voxel_size(self) -> float:
        """Edge length of one voxel in mm. Default 0.5 mm."""
        return self._qs.value("sim/voxel_size", 0.5, type=float)

    @voxel_size.setter
    def voxel_size(self, v: float) -> None:
        v = float(max(0.05, v))   # prevent accidentally tiny grids
        self._qs.setValue("sim/voxel_size", v)
        self.voxel_size_changed.emit(v)

    SIM_BACKENDS: tuple[str, ...] = ("voxel", "dexel")

    @property
    def sim_backend(self) -> str:
        """Material-removal model: "voxel" (dense grid, sim/voxel) or
        "dexel" (tri-dexel rays, sim/dexel). Default "voxel"."""
        v = self._qs.value("sim/backend", "voxel", type=str)
        return v if v in self.SIM_BACKENDS else "voxel"

    @sim_backend.setter
    def sim_backend(self, v: str) -> None:
        if v not in self.SIM_BACKENDS:
            v = "voxel"
        self._qs.setValue("sim/backend", v)
        self.sim_backend_changed.emit(v)

    @property
    def voxel_enabled(self) -> bool:
        """Whether the voxel material-removal simulation should run. Default True."""
        return self._qs.value("sim/voxel_enabled", True, type=bool)

    @voxel_enabled.setter
    def voxel_enabled(self, v: bool) -> None:
        self._qs.setValue("sim/voxel_enabled", bool(v))
        self.voxel_enabled_changed.emit(bool(v))

    @property
    def voxel_color(self) -> str:
        """Name key from VOXEL_COLORS. Default: 'Orange-Gelb'."""
        return self._qs.value("sim/voxel_color", "Orange-Gelb", type=str)

    @voxel_color.setter
    def voxel_color(self, name: str) -> None:
        self._qs.setValue("sim/voxel_color", name)
        self.voxel_color_changed.emit(name)

    def voxel_color_rgb(self) -> tuple[float, float, float]:
        """Current voxel colour as (r, g, b) floats 0–1."""
        return self.VOXEL_COLORS.get(self.voxel_color, (0.95, 0.68, 0.12))

    @property
    def tool_cutting_color(self) -> str:
        """Name key from TOOL_COLORS. Default: 'Gold (Standard)'."""
        return self._qs.value("sim/tool_cutting_color", "Gold (Standard)", type=str)

    @tool_cutting_color.setter
    def tool_cutting_color(self, name: str) -> None:
        self._qs.setValue("sim/tool_cutting_color", name)
        self.tool_cutting_color_changed.emit(name)

    def tool_cutting_color_rgb(self) -> tuple[float, float, float]:
        """Current tool cutting-edge colour as (r, g, b) floats 0–1."""
        return self.TOOL_COLORS.get(self.tool_cutting_color, (1.0, 0.84, 0.0))

    @property
    def show_tool_holder(self) -> bool:
        """Whether the active tool's holder ("Werkzeugaufnahme") geometry is
        rendered above the tool shank. Default False — most tools have no
        holder_preset assigned yet, so this stays off until one is."""
        return self._qs.value("sim/show_tool_holder", False, type=bool)

    @show_tool_holder.setter
    def show_tool_holder(self, v: bool) -> None:
        self._qs.setValue("sim/show_tool_holder", bool(v))
        self.show_tool_holder_changed.emit(bool(v))

    @property
    def collision_detection_enabled(self) -> bool:
        """Whether the voxel engine checks each move for a tool/material
        collision (rapids touching material; shank/holder touching material
        on a cutting move). Default True — on unless explicitly disabled."""
        return self._qs.value("sim/collision_detection_enabled", True, type=bool)

    @collision_detection_enabled.setter
    def collision_detection_enabled(self, v: bool) -> None:
        self._qs.setValue("sim/collision_detection_enabled", bool(v))
        self.collision_detection_enabled_changed.emit(bool(v))

    @property
    def collision_check_before_start_enabled(self) -> bool:
        """Whether MachinePage runs the whole-program collision pre-check
        (DatumSimWidget.presim_check_collisions()) before Start actually
        launches the program. Independent of collision_detection_enabled
        above — that one only gates the voxel engine's own live/simulation
        checking, not whether Start pre-flights the program at all. Default
        False — the pre-start check is opt-in, unlike the always-on
        in-simulation collision feedback."""
        return self._qs.value("sim/collision_check_before_start_enabled", False, type=bool)

    @collision_check_before_start_enabled.setter
    def collision_check_before_start_enabled(self, v: bool) -> None:
        self._qs.setValue("sim/collision_check_before_start_enabled", bool(v))
        self.collision_check_before_start_enabled_changed.emit(bool(v))

    # ── Persistence paths ──────────────────────────────────────────────────────

    @property
    def tool_db_path(self) -> str:
        """Path to the tool database sqlite file. Empty string (default)
        means "use the default location" (persistence/paths.py's
        <repo_root>/data/db/tools.db) — resolved by ToolDatabase itself,
        not here, so this stays a pure string setting with no Qt-app-
        instance dependency at import time."""
        return self._qs.value("persistence/tool_db_path", "", type=str)

    @tool_db_path.setter
    def tool_db_path(self, v: str) -> None:
        self._qs.setValue("persistence/tool_db_path", v)
        self.tool_db_path_changed.emit(v)

    @property
    def workpiece_db_path(self) -> str:
        """Path to the workpiece database sqlite file. Empty string
        (default) means "use the default location" — see tool_db_path,
        same pattern; resolved by WorkpieceDatabase itself."""
        return self._qs.value("persistence/workpiece_db_path", "", type=str)

    @workpiece_db_path.setter
    def workpiece_db_path(self, v: str) -> None:
        self._qs.setValue("persistence/workpiece_db_path", v)
        self.workpiece_db_path_changed.emit(v)

    @property
    def linuxcnc_tool_table_path(self) -> str:
        """Path a real LinuxCNC-format tool.tbl is auto-exported to on every
        tool database change. Empty string (default) disables the export —
        no path is configured, so there's nothing to keep in sync yet."""
        return self._qs.value("persistence/linuxcnc_tool_table_path", "", type=str)

    @linuxcnc_tool_table_path.setter
    def linuxcnc_tool_table_path(self, v: str) -> None:
        self._qs.setValue("persistence/linuxcnc_tool_table_path", v)
        self.linuxcnc_tool_table_path_changed.emit(v)

    @property
    def workpieces_root_path(self) -> str:
        """Root folder the Workpieces page syncs against — every direct
        subfolder becomes one Workpiece (see persistence/workpiece_sync.py).

        Defaults to persistence.paths.DEFAULT_WORKPIECES_ROOT
        (<repo_root>/workpieces, where the bundled example G-code already
        lives) rather than an empty no-op path, so sync works out of the
        box; set an explicit value (an absolute path, or override this
        default) to point it elsewhere."""
        configured = self._qs.value("persistence/workpieces_root_path", "", type=str)
        if configured:
            return configured

        from controller.persistence.paths import DEFAULT_WORKPIECES_ROOT

        return str(DEFAULT_WORKPIECES_ROOT)

    @workpieces_root_path.setter
    def workpieces_root_path(self, v: str) -> None:
        self._qs.setValue("persistence/workpieces_root_path", v)
        self.workpieces_root_path_changed.emit(v)

    @property
    def workpieces_explorer_root_path(self) -> str:
        """Folder the "Programm laden" file picker (WorkpieceDetailPage,
        see ui/pages/workpiece_detail_page.py) opens in by default — e.g.
        a USB stick's mount point where new G-code files typically come
        from. Empty string (default) lets Qt's file dialog pick its own
        default (usually the last-used directory)."""
        return self._qs.value("persistence/workpieces_explorer_root_path", "", type=str)

    @workpieces_explorer_root_path.setter
    def workpieces_explorer_root_path(self, v: str) -> None:
        self._qs.setValue("persistence/workpieces_explorer_root_path", v)
        self.workpieces_explorer_root_path_changed.emit(v)

    # ── Stock shape settings ───────────────────────────────────────────────────

    @property
    def stock_shape(self) -> str:
        """Stock shape: 'bounding_box' or 'round'. Default: 'bounding_box'."""
        return self._qs.value("sim/stock_shape", "bounding_box", type=str)

    @stock_shape.setter
    def stock_shape(self, v: str) -> None:
        self._qs.setValue("sim/stock_shape", v)
        self.stock_shape_changed.emit(v)

    @property
    def stock_z_offset_mm(self) -> float:
        """Distance from Z=0 to the stock top surface (mm). Default 0.0."""
        return self._qs.value("sim/stock_z_offset_mm", 0.0, type=float)

    @stock_z_offset_mm.setter
    def stock_z_offset_mm(self, v: float) -> None:
        self._qs.setValue("sim/stock_z_offset_mm", float(v))
        self.stock_z_offset_changed.emit(float(v))

    @property
    def stock_height_mm(self) -> float:
        """Stock height in mm. 0.0 = auto (derived from cutting path)."""
        return self._qs.value("sim/stock_height_mm", 0.0, type=float)

    @stock_height_mm.setter
    def stock_height_mm(self, v: float) -> None:
        self._qs.setValue("sim/stock_height_mm", float(v))
        self.stock_height_changed.emit(float(v))

    @property
    def stock_round_radius_mm(self) -> float:
        """Cylinder radius for ROUND stock (mm). Default 50.0."""
        return self._qs.value("sim/stock_round_radius_mm", 50.0, type=float)

    @stock_round_radius_mm.setter
    def stock_round_radius_mm(self, v: float) -> None:
        self._qs.setValue("sim/stock_round_radius_mm", float(v))
        self.stock_round_radius_changed.emit(float(v))

    @property
    def stock_width_mm(self) -> float:
        """BOUNDING_BOX stock X size in mm. 0.0 = auto (derived from the
        G-code cutting path's extent + margin, the historical behavior)."""
        return self._qs.value("sim/stock_width_mm", 0.0, type=float)

    @stock_width_mm.setter
    def stock_width_mm(self, v: float) -> None:
        self._qs.setValue("sim/stock_width_mm", float(v))
        self.stock_width_changed.emit(float(v))

    @property
    def stock_depth_mm(self) -> float:
        """BOUNDING_BOX stock Y size in mm. 0.0 = auto (see stock_width_mm)."""
        return self._qs.value("sim/stock_depth_mm", 0.0, type=float)

    @stock_depth_mm.setter
    def stock_depth_mm(self, v: float) -> None:
        self._qs.setValue("sim/stock_depth_mm", float(v))
        self.stock_depth_changed.emit(float(v))

    @property
    def stock_x_offset_mm(self) -> float:
        """Distance from the work origin to the stock's X corner (mm).
        Only meaningful when stock_width_mm is set (not auto) — 0.0 puts
        the origin exactly at the stock's lower-X edge."""
        return self._qs.value("sim/stock_x_offset_mm", 0.0, type=float)

    @stock_x_offset_mm.setter
    def stock_x_offset_mm(self, v: float) -> None:
        self._qs.setValue("sim/stock_x_offset_mm", float(v))
        self.stock_x_offset_changed.emit(float(v))

    @property
    def stock_y_offset_mm(self) -> float:
        """Distance from the work origin to the stock's Y corner (mm). See
        stock_x_offset_mm."""
        return self._qs.value("sim/stock_y_offset_mm", 0.0, type=float)

    @stock_y_offset_mm.setter
    def stock_y_offset_mm(self, v: float) -> None:
        self._qs.setValue("sim/stock_y_offset_mm", float(v))
        self.stock_y_offset_changed.emit(float(v))

    # ── Simulation start position ───────────────────────────────────────────────

    @property
    def start_safe_z_mm(self) -> float:
        """Assumed tool Z position (above work zero) BEFORE the G-code
        program's first command, mm. Default 100.0 — the simulation's first
        move is otherwise implicitly a rapid FROM the work origin itself,
        which is usually inside/at the stock and reads as a false-positive
        collision. Takes effect on the next file (re)load, not live."""
        return self._qs.value("sim/start_safe_z_mm", 100.0, type=float)

    @start_safe_z_mm.setter
    def start_safe_z_mm(self, v: float) -> None:
        self._qs.setValue("sim/start_safe_z_mm", float(v))
        self.start_safe_z_mm_changed.emit(float(v))

    # ── Manual jog (ManualPage / JogControlPanel) ───────────────────────────
    # No backend Rapid-jog mode exists (jog_continuous()/jog_increment() just
    # take a velocity, sign = direction — see MachineController's own
    # docstrings) — Feed/Rapid on the jog pad is purely these two UI-level
    # velocity presets, swapped by the Rapid toggle button.

    @property
    def jog_feed_velocity_mm_s(self) -> float:
        """Jog velocity (mm/s) used while the JogControlPanel's Rapid toggle
        is OFF. Default 10.0 mm/s."""
        return self._qs.value("manual/jog_feed_velocity_mm_s", 10.0, type=float)

    @jog_feed_velocity_mm_s.setter
    def jog_feed_velocity_mm_s(self, v: float) -> None:
        self._qs.setValue("manual/jog_feed_velocity_mm_s", float(v))
        self.jog_feed_velocity_mm_s_changed.emit(float(v))

    @property
    def jog_rapid_velocity_mm_s(self) -> float:
        """Jog velocity (mm/s) used while the JogControlPanel's Rapid toggle
        is ON. Default 50.0 mm/s."""
        return self._qs.value("manual/jog_rapid_velocity_mm_s", 50.0, type=float)

    @jog_rapid_velocity_mm_s.setter
    def jog_rapid_velocity_mm_s(self, v: float) -> None:
        self._qs.setValue("manual/jog_rapid_velocity_mm_s", float(v))
        self.jog_rapid_velocity_mm_s_changed.emit(float(v))

    # ── Tool magazine ────────────────────────────────────────────────────────

    @property
    def tool_pocket_count(self) -> int:
        """Number of physical tool-magazine pockets shown by ToolPage's
        pinned magazine bar (P1..Pn). Default 10."""
        return self._qs.value("tools/pocket_count", 10, type=int)

    @tool_pocket_count.setter
    def tool_pocket_count(self, v: int) -> None:
        v = max(1, int(v))
        self._qs.setValue("tools/pocket_count", v)
        self.tool_pocket_count_changed.emit(v)

    # ── Machine axis configuration ──────────────────────────────────────────

    @property
    def has_rotary_axes(self) -> bool:
        """Whether this machine has configured rotary axes (A/B/C) in
        addition to X/Y/Z. Default False — no UI to change this yet, but
        ManualPage's jog grid (ui/pages/manual_page.py) checks it to decide
        whether to show A/B/C jog controls at all."""
        return self._qs.value("machine/has_rotary_axes", False, type=bool)

    @has_rotary_axes.setter
    def has_rotary_axes(self, v: bool) -> None:
        self._qs.setValue("machine/has_rotary_axes", bool(v))
        self.has_rotary_axes_changed.emit(bool(v))
