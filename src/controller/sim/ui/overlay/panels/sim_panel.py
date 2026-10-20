"""
sim/ui/overlay/panels/sim_panel.py — Sim-widget settings section widgets.

Each class here is a self-contained settings section bound directly to the
AppSettings singleton — not just writing to it on user interaction, but also
listening to its *_changed signals to keep its own displayed state in sync.
That two-way binding is what makes it safe to instantiate the SAME section
class more than once (build_sections() is called once for the sim widget's
own slide-out overlay panel, and again for the "official" app-wide
SettingsPage) — whichever instance the user interacts with, every other
instance updates itself from the same underlying setting, no widget-to-
widget wiring needed.

Sections:
  Darstellung   Tool/path display mode, info-bar toggles, viewport toggles
  Optik         All color settings: background theme, tool cutting-edge
                color, stock material color
  Simulation    Voxel enable/disable, voxel resolution
  Rohteil       Stock shape, Z-offset, height, radius
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QVBoxLayout, QWidget,
)

from controller.sim.core.settings import AppSettings
from controller.sim.ui.viewport   import PathMode, ToolMode

try:
    from controller.sim.voxel.controller import VoxelSimController as _VSC  # noqa: F401
    _VOXEL_AVAILABLE = True
    del _VSC
except ImportError:
    _VOXEL_AVAILABLE = False

_SHAPE_LABELS = ["Boundary Box", "Rund"]
_SHAPE_KEYS   = ["bounding_box", "round"]

# (icon_name, tooltip) for each section, in display order — shared by
# settings_panel.py (sim widget overlay) and settings_page.py (app page)
# so both build the exact same nav, in the same order.
SECTION_ICONS: list[tuple[str, str]] = [
    ("settings",   "Darstellung"),
    ("file-3d",    "Optik"),
    ("scan-cube",  "Simulation"),
    ("workpieces", "Rohteil"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Reusable helpers
# ══════════════════════════════════════════════════════════════════════════════

class _ColorSwatch(QWidget):
    """Small colored rectangle showing a hex color."""

    def __init__(self, hex_color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self._color = hex_color
        self._update_style()

    def set_color(self, hex_color: str) -> None:
        self._color = hex_color
        self._update_style()

    def _update_style(self) -> None:
        self.setStyleSheet(
            f"background: {self._color}; border: 1px solid rgba(255,255,255,0.15);"
            f" border-radius: 4px;"
        )


def _hdr(text: str) -> QLabel:
    lbl = QLabel(text)
    font = lbl.font()
    font.setBold(True)
    lbl.setFont(font)
    return lbl


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ── Two-way binding helpers ─────────────────────────────────────────────────
# "write" side (widget -> AppSettings) is a plain one-line signal connection
# at each call site; these only handle the "read" side (AppSettings -> widget)
# so a change made through ANY instance of a section is reflected by every
# other live instance, without echoing back and re-triggering itself.

def _sync_checkbox(chk: QCheckBox, value: bool) -> None:
    if chk.isChecked() != value:
        chk.blockSignals(True)
        chk.setChecked(value)
        chk.blockSignals(False)


def _sync_combo_text(combo: QComboBox, text: str) -> None:
    idx = combo.findText(text)
    if idx >= 0 and idx != combo.currentIndex():
        combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)


def _sync_spin(spin: QDoubleSpinBox, value: float) -> None:
    if abs(spin.value() - value) > 1e-9:
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Darstellung
# ══════════════════════════════════════════════════════════════════════════════

class _DisplayTab(QWidget):
    """Tool/path display mode, info-bar toggles, viewport toggles."""

    _TOOL_LABELS = {ToolMode.CYLINDER: "Endmill",
                    ToolMode.POINT:    "Point",
                    ToolMode.NONE:     "None"}
    _PATH_LABELS = {PathMode.FULL:        "Complete",
                    PathMode.PROGRESSIVE: "Progressive",
                    PathMode.NONE:        "None"}

    def __init__(self, s: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._s = s

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Tool / path display mode ─────────────────────────────────────────
        root.addWidget(_hdr("Werkzeug & Pfad"))
        tf = QFormLayout(); tf.setSpacing(8)
        self._tool_combo = QComboBox()
        self._tool_combo.addItems(["Endmill", "Point", "None"])
        tf.addRow("Werkzeug-Anzeige", self._tool_combo)

        self._path_combo = QComboBox()
        self._path_combo.addItems(["Complete", "Progressive", "None"])
        tf.addRow("Pfad-Modus", self._path_combo)

        self._chk_holder = QCheckBox()
        tf.addRow("Werkzeugaufnahme anzeigen", self._chk_holder)
        root.addLayout(tf)

        # Info bar ──────────────────────────────────────────────────────────
        root.addWidget(_hdr("Info-Leiste"))
        inf = QFormLayout(); inf.setSpacing(8)
        self._chk_datum    = QCheckBox()
        self._chk_gcode    = QCheckBox()
        self._chk_tool     = QCheckBox()
        self._chk_feedrate = QCheckBox()
        inf.addRow("WCS / Nullpunkt",  self._chk_datum)
        inf.addRow("G-code Zeile",     self._chk_gcode)
        inf.addRow("Werkzeug",         self._chk_tool)
        inf.addRow("Vorschub",         self._chk_feedrate)
        root.addLayout(inf)

        # Viewport ───────────────────────────────────────────────────────────
        root.addWidget(_hdr("Viewport"))
        vp = QFormLayout(); vp.setSpacing(8)
        self._chk_axes         = QCheckBox()
        self._chk_grid         = QCheckBox()
        self._chk_datum_symbol = QCheckBox()
        vp.addRow("Achsen (X/Y/Z)",    self._chk_axes)
        vp.addRow("Gitter (XY-Ebene)", self._chk_grid)
        vp.addRow("Nullpunkt-Symbol",  self._chk_datum_symbol)
        root.addLayout(vp)
        root.addStretch()

        # Load saved state
        for chk, val in [
            (self._chk_datum,         s.show_datum),
            (self._chk_gcode,         s.show_gcode_line),
            (self._chk_tool,          s.show_tool),
            (self._chk_feedrate,      s.show_feedrate),
            (self._chk_axes,          s.show_axes),
            (self._chk_grid,          s.show_grid),
            (self._chk_datum_symbol,  s.show_datum_symbol),
            (self._chk_holder,        s.show_tool_holder),
        ]:
            chk.blockSignals(True)
            chk.setChecked(val)
            chk.blockSignals(False)

        _sync_combo_text(self._tool_combo, s.tool_mode)
        _sync_combo_text(self._path_combo, s.path_mode)

        # Write side: widget -> AppSettings
        self._tool_combo.currentIndexChanged.connect(
            lambda _i: setattr(s, "tool_mode", self._tool_combo.currentText()))
        self._path_combo.currentIndexChanged.connect(
            lambda _i: setattr(s, "path_mode", self._path_combo.currentText()))
        self._chk_datum.toggled.connect(        lambda v: setattr(s, "show_datum",        v))
        self._chk_gcode.toggled.connect(        lambda v: setattr(s, "show_gcode_line",   v))
        self._chk_tool.toggled.connect(         lambda v: setattr(s, "show_tool",         v))
        self._chk_feedrate.toggled.connect(     lambda v: setattr(s, "show_feedrate",     v))
        self._chk_axes.toggled.connect(         lambda v: setattr(s, "show_axes",         v))
        self._chk_grid.toggled.connect(         lambda v: setattr(s, "show_grid",         v))
        self._chk_datum_symbol.toggled.connect( lambda v: setattr(s, "show_datum_symbol", v))
        self._chk_holder.toggled.connect(       lambda v: setattr(s, "show_tool_holder",  v))

        # Read side: AppSettings -> widget (keeps every open instance in sync).
        # Bound methods, deliberately NOT inline lambdas: build_sections() is
        # instantiated fresh every time a ProgramDetailPage is opened (its own
        # DatumSimWidget -> SettingsPanel -> build_sections(), see this
        # module's docstring) and torn down via deleteLater() on pop — while
        # AppSettings itself is an app-lifetime singleton. A lambda closing
        # over `self._chk_xxx` has no `__self__`, so Qt has no QObject to tie
        # the connection's lifetime to and never auto-disconnects it; the
        # next *_changed emission after teardown then runs the lambda against
        # an already-deleted C++ checkbox (libshiboken RuntimeError). A bound
        # method of `self` (a QWidget) IS recognized as belonging to that
        # QObject, so Qt drops the connection automatically once `self`'s C++
        # side is destroyed — same reasoning already applied to
        # _on_bg_settings_changed/_on_enabled_changed/_on_shape_changed below.
        s.tool_mode_changed.connect(self._on_tool_mode_changed)
        s.path_mode_changed.connect(self._on_path_mode_changed)
        s.show_datum_changed.connect(        self._on_show_datum_changed)
        s.show_gcode_line_changed.connect(   self._on_show_gcode_line_changed)
        s.show_tool_changed.connect(         self._on_show_tool_changed)
        s.show_feedrate_changed.connect(     self._on_show_feedrate_changed)
        s.show_axes_changed.connect(         self._on_show_axes_changed)
        s.show_grid_changed.connect(         self._on_show_grid_changed)
        s.show_datum_symbol_changed.connect( self._on_show_datum_symbol_changed)
        s.show_tool_holder_changed.connect(  self._on_show_tool_holder_changed)

    def _on_tool_mode_changed(self, v: str) -> None:
        _sync_combo_text(self._tool_combo, v)

    def _on_path_mode_changed(self, v: str) -> None:
        _sync_combo_text(self._path_combo, v)

    def _on_show_datum_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_datum, v)

    def _on_show_gcode_line_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_gcode, v)

    def _on_show_tool_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_tool, v)

    def _on_show_feedrate_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_feedrate, v)

    def _on_show_axes_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_axes, v)

    def _on_show_grid_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_grid, v)

    def _on_show_datum_symbol_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_datum_symbol, v)

    def _on_show_tool_holder_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_holder, v)


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Optik (all color settings)
# ══════════════════════════════════════════════════════════════════════════════

class _AppearanceTab(QWidget):
    """All sim-widget color settings in one place: background theme, tool
    cutting-edge color, stock material color — fixed presets, no free
    color picker."""

    def __init__(self, s: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._s = s

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(_hdr("Farben"))
        form = QFormLayout(); form.setSpacing(8)

        self._bg_combo  = QComboBox()
        self._bg_swatch = _ColorSwatch("#1c1c1c")
        for name, (inner, _outer) in AppSettings.BG_COLORS.items():
            self._bg_combo.addItem(name, userData=inner)
        bg_row = QHBoxLayout(); bg_row.setSpacing(8)
        bg_row.addWidget(self._bg_swatch)
        bg_row.addWidget(self._bg_combo)
        form.addRow("Hintergrund", bg_row)

        self._tool_color_combo  = QComboBox()
        self._tool_color_swatch = _ColorSwatch("#ffd600")
        for name, rgb in AppSettings.TOOL_COLORS.items():
            self._tool_color_combo.addItem(name, userData=_rgb_to_hex(rgb))
        tool_row = QHBoxLayout(); tool_row.setSpacing(8)
        tool_row.addWidget(self._tool_color_swatch)
        tool_row.addWidget(self._tool_color_combo)
        form.addRow("Werkzeug-Schneide", tool_row)

        self._mat_color_combo  = QComboBox()
        self._mat_color_swatch = _ColorSwatch("#f2ae1f")
        for name, rgb in AppSettings.VOXEL_COLORS.items():
            self._mat_color_combo.addItem(name, userData=_rgb_to_hex(rgb))
        mat_row = QHBoxLayout(); mat_row.setSpacing(8)
        mat_row.addWidget(self._mat_color_swatch)
        mat_row.addWidget(self._mat_color_combo)
        form.addRow("Rohteil-Material", mat_row)

        root.addLayout(form)
        root.addStretch()

        # Load saved state
        _sync_combo_text(self._bg_combo, s.bg_theme_name())
        self._update_bg_swatch()
        _sync_combo_text(self._tool_color_combo, s.tool_cutting_color)
        self._update_tool_swatch()
        _sync_combo_text(self._mat_color_combo, s.voxel_color)
        self._update_mat_swatch()

        # Write side
        self._bg_combo.currentIndexChanged.connect(self._on_bg_changed)
        self._tool_color_combo.currentIndexChanged.connect(self._on_tool_color_changed)
        self._mat_color_combo.currentIndexChanged.connect(self._on_mat_color_changed)

        # Read side
        s.bg_color_changed.connect(self._on_bg_settings_changed)
        s.bg_color_2_changed.connect(self._on_bg_settings_changed)
        s.tool_cutting_color_changed.connect(self._on_tool_color_settings_changed)
        s.voxel_color_changed.connect(self._on_mat_color_settings_changed)

    def _on_bg_changed(self, _idx: int) -> None:
        self._s.apply_bg_theme(self._bg_combo.currentText())
        self._update_bg_swatch()

    def _on_bg_settings_changed(self, _v: str) -> None:
        _sync_combo_text(self._bg_combo, self._s.bg_theme_name())
        self._update_bg_swatch()

    def _update_bg_swatch(self) -> None:
        hex_val = self._bg_combo.currentData()
        if hex_val:
            self._bg_swatch.set_color(hex_val)

    def _on_tool_color_changed(self, _idx: int) -> None:
        self._s.tool_cutting_color = self._tool_color_combo.currentText()
        self._update_tool_swatch()

    def _update_tool_swatch(self) -> None:
        hex_val = self._tool_color_combo.currentData()
        if hex_val:
            self._tool_color_swatch.set_color(hex_val)

    def _on_mat_color_changed(self, _idx: int) -> None:
        self._s.voxel_color = self._mat_color_combo.currentText()
        self._update_mat_swatch()

    def _update_mat_swatch(self) -> None:
        hex_val = self._mat_color_combo.currentData()
        if hex_val:
            self._mat_color_swatch.set_color(hex_val)

    def _on_tool_color_settings_changed(self, v: str) -> None:
        _sync_combo_text(self._tool_color_combo, v)
        self._update_tool_swatch()

    def _on_mat_color_settings_changed(self, v: str) -> None:
        _sync_combo_text(self._mat_color_combo, v)
        self._update_mat_swatch()


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Simulation
# ══════════════════════════════════════════════════════════════════════════════

# Keys match AppSettings.SIM_BACKENDS; labels are what the combo shows.
_BACKEND_KEYS   = ["voxel", "dexel"]
_BACKEND_LABELS = ["Voxel", "Tri-Dexel"]


class _VoxelSimTab(QWidget):
    """Voxel simulation enable switch and resolution."""

    def __init__(self, s: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._s = s

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(_hdr("Abtragssimulation"))
        form = QFormLayout(); form.setSpacing(8)

        self._chk = QCheckBox()
        if not _VOXEL_AVAILABLE:
            self._chk.setEnabled(False)
            self._chk.setToolTip(
                "Nicht verfügbar — numpy / moderngl nicht importierbar."
            )
        form.addRow("Aktivieren", self._chk)

        self._backend_combo = QComboBox()
        self._backend_combo.addItems(_BACKEND_LABELS)
        self._backend_combo.setToolTip(
            "Materialmodell der Abtragssimulation.\n"
            "Voxel: dichtes Würfelgitter, in allen Richtungen auf die Auflösung gerastert.\n"
            "Tri-Dexel: Strahlen in X/Y/Z mit exakten Materialgrenzen entlang\n"
            "jedes Strahls — glattere Flächen, deutlich weniger Speicher bei\n"
            "feiner Auflösung.\n"
            "Änderung baut die Simulation sofort neu auf."
        )
        form.addRow("Modell", self._backend_combo)

        if not _VOXEL_AVAILABLE:
            warn = QLabel("⚠ Voxel-Module nicht verfügbar")
            warn.setStyleSheet("color: #e0a040; font-size: 11px;")
            warn.setWordWrap(True)
            form.addRow("", warn)

        self._size_spin = QDoubleSpinBox()
        self._size_spin.setRange(0.05, 5.0)
        self._size_spin.setSingleStep(0.1)
        self._size_spin.setDecimals(2)
        self._size_spin.setSuffix(" mm")
        self._size_spin.setToolTip(
            "Kantenlänge eines Voxels bzw. Strahlabstand (Tri-Dexel).\n"
            "Kleiner = feiner, aber mehr Speicher + langsamer.\n"
            "Änderung baut die Simulation sofort neu auf."
        )
        form.addRow("Auflösung", self._size_spin)
        root.addLayout(form)

        root.addWidget(_hdr("Sicherheit"))
        safety_form = QFormLayout(); safety_form.setSpacing(8)
        self._chk_collision = QCheckBox()
        self._chk_collision.setToolTip(
            "Prüft jede Bewegung auf Kollision: Eilgänge dürfen kein "
            "Material berühren, Zustellungen nur bis zur Schneidenlänge — "
            "Schaft/Aufnahme darüber gelten als Kollision."
        )
        safety_form.addRow("Kollisionserkennung", self._chk_collision)

        self._safe_z_spin = QDoubleSpinBox()
        self._safe_z_spin.setRange(0.0, 500.0)
        self._safe_z_spin.setSingleStep(5.0)
        self._safe_z_spin.setDecimals(1)
        self._safe_z_spin.setSuffix(" mm")
        self._safe_z_spin.setToolTip(
            "Angenommene Werkzeug-Z-Position (über dem Werkstück-Nullpunkt) "
            "vor dem ersten Programmschritt — verhindert eine fälschliche "
            "G00-Kollisionsmeldung beim Programmstart.\n"
            "Wirkt erst beim nächsten Laden der Datei, nicht live."
        )
        safety_form.addRow("Start-Sicherheitsabstand Z", self._safe_z_spin)
        root.addLayout(safety_form)

        root.addStretch()

        # Load saved
        self._chk.blockSignals(True)
        self._chk.setChecked(s.voxel_enabled and _VOXEL_AVAILABLE)
        self._chk.blockSignals(False)
        _sync_spin(self._size_spin, s.voxel_size)
        self._backend_combo.blockSignals(True)
        self._backend_combo.setCurrentIndex(_BACKEND_KEYS.index(s.sim_backend))
        self._backend_combo.blockSignals(False)
        self._sync_size_state()
        self._chk_collision.blockSignals(True)
        self._chk_collision.setChecked(s.collision_detection_enabled)
        self._chk_collision.blockSignals(False)
        _sync_spin(self._safe_z_spin, s.start_safe_z_mm)

        # Write side
        self._chk.toggled.connect(self._on_enabled)
        self._size_spin.valueChanged.connect(lambda v: setattr(s, "voxel_size", v))
        self._backend_combo.currentIndexChanged.connect(self._on_backend)
        self._chk_collision.toggled.connect(
            lambda v: setattr(s, "collision_detection_enabled", v))
        self._safe_z_spin.valueChanged.connect(lambda v: setattr(s, "start_safe_z_mm", v))

        # Read side — bound methods, not lambdas: see _DisplayTab's read-side
        # block above for why (this exact checkbox is the one that crashed
        # with a libshiboken "already deleted" RuntimeError before this fix).
        s.voxel_enabled_changed.connect(self._on_enabled_changed)
        s.voxel_size_changed.connect(self._on_voxel_size_settings_changed)
        s.sim_backend_changed.connect(self._on_backend_settings_changed)
        s.collision_detection_enabled_changed.connect(self._on_collision_settings_changed)
        s.start_safe_z_mm_changed.connect(self._on_safe_z_settings_changed)

    def _sync_size_state(self) -> None:
        on = _VOXEL_AVAILABLE and self._chk.isChecked()
        self._size_spin.setEnabled(on)
        self._backend_combo.setEnabled(on)

    def _on_backend(self, idx: int) -> None:
        self._s.sim_backend = _BACKEND_KEYS[idx]

    def _on_backend_settings_changed(self, key: str) -> None:
        idx = _BACKEND_KEYS.index(key) if key in _BACKEND_KEYS else 0
        if idx != self._backend_combo.currentIndex():
            self._backend_combo.blockSignals(True)
            self._backend_combo.setCurrentIndex(idx)
            self._backend_combo.blockSignals(False)

    def _on_enabled(self, checked: bool) -> None:
        self._s.voxel_enabled = checked
        self._sync_size_state()

    def _on_enabled_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk, v and _VOXEL_AVAILABLE)
        self._sync_size_state()

    def _on_voxel_size_settings_changed(self, v: float) -> None:
        _sync_spin(self._size_spin, v)

    def _on_collision_settings_changed(self, v: bool) -> None:
        _sync_checkbox(self._chk_collision, v)

    def _on_safe_z_settings_changed(self, v: float) -> None:
        _sync_spin(self._safe_z_spin, v)


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Rohteil
# ══════════════════════════════════════════════════════════════════════════════

class _StockTab(QWidget):
    """Workpiece shape and dimensional overrides."""

    def __init__(self, s: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._s = s

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(_hdr("Rohteil"))
        form = QFormLayout(); form.setSpacing(8)

        self._shape_combo = QComboBox()
        self._shape_combo.addItems(_SHAPE_LABELS)
        form.addRow("Form", self._shape_combo)

        self._z_spin = QDoubleSpinBox()
        self._z_spin.setRange(-500.0, 500.0)
        self._z_spin.setSingleStep(0.5)
        self._z_spin.setDecimals(2)
        self._z_spin.setSuffix(" mm")
        self._z_spin.setToolTip(
            "Abstand vom Werkzeug-Nullpunkt (Z=0) zur Rohteiloberfläche.\n"
            "0 mm = Oberfläche liegt bei Z=0 (Standard)."
        )
        form.addRow("Z-Oberfläche", self._z_spin)

        self._h_spin = QDoubleSpinBox()
        self._h_spin.setRange(0.0, 2000.0)
        self._h_spin.setSingleStep(1.0)
        self._h_spin.setDecimals(1)
        self._h_spin.setSuffix(" mm")
        self._h_spin.setSpecialValueText("Auto")
        self._h_spin.setToolTip("Rohteilhöhe. 0 = Auto aus G-code-Pfad.")
        form.addRow("Höhe", self._h_spin)

        self._r_label = QLabel("Radius")
        self._r_spin  = QDoubleSpinBox()
        self._r_spin.setRange(1.0, 2000.0)
        self._r_spin.setSingleStep(5.0)
        self._r_spin.setDecimals(1)
        self._r_spin.setSuffix(" mm")
        self._r_spin.setToolTip("Zylinderradius um den Werkstück-Nullpunkt.")
        form.addRow(self._r_label, self._r_spin)

        # Bounding-box-only: explicit size + corner offset. Hidden for ROUND
        # (see _sync_radius_row(), which despite its name now also drives
        # these rows' visibility — inverse of the radius row's).
        self._box_rows: list[tuple[QLabel, QWidget]] = []

        self._w_label = QLabel("Breite (X)")
        self._w_spin  = QDoubleSpinBox()
        self._w_spin.setRange(0.0, 2000.0)
        self._w_spin.setSingleStep(5.0)
        self._w_spin.setDecimals(1)
        self._w_spin.setSuffix(" mm")
        self._w_spin.setSpecialValueText("Auto")
        self._w_spin.setToolTip("Rohteilbreite (X). 0 = Auto aus G-code-Pfad.")
        form.addRow(self._w_label, self._w_spin)
        self._box_rows.append((self._w_label, self._w_spin))

        self._d_label = QLabel("Tiefe (Y)")
        self._d_spin  = QDoubleSpinBox()
        self._d_spin.setRange(0.0, 2000.0)
        self._d_spin.setSingleStep(5.0)
        self._d_spin.setDecimals(1)
        self._d_spin.setSuffix(" mm")
        self._d_spin.setSpecialValueText("Auto")
        self._d_spin.setToolTip("Rohteiltiefe (Y). 0 = Auto aus G-code-Pfad.")
        form.addRow(self._d_label, self._d_spin)
        self._box_rows.append((self._d_label, self._d_spin))

        self._xo_label = QLabel("X-Offset (Ecke)")
        self._xo_spin  = QDoubleSpinBox()
        self._xo_spin.setRange(-1000.0, 1000.0)
        self._xo_spin.setSingleStep(1.0)
        self._xo_spin.setDecimals(1)
        self._xo_spin.setSuffix(" mm")
        self._xo_spin.setToolTip(
            "Abstand vom Werkstück-Nullpunkt zur unteren X-Kante des Rohteils.\n"
            "Nur wirksam bei fester Breite (nicht Auto). 0 mm = Nullpunkt liegt\n"
            "genau auf der Rohteilecke."
        )
        form.addRow(self._xo_label, self._xo_spin)
        self._box_rows.append((self._xo_label, self._xo_spin))

        self._yo_label = QLabel("Y-Offset (Ecke)")
        self._yo_spin  = QDoubleSpinBox()
        self._yo_spin.setRange(-1000.0, 1000.0)
        self._yo_spin.setSingleStep(1.0)
        self._yo_spin.setDecimals(1)
        self._yo_spin.setSuffix(" mm")
        self._yo_spin.setToolTip("Abstand vom Werkstück-Nullpunkt zur unteren Y-Kante. Siehe X-Offset.")
        form.addRow(self._yo_label, self._yo_spin)
        self._box_rows.append((self._yo_label, self._yo_spin))

        root.addLayout(form)
        root.addStretch()

        if not _VOXEL_AVAILABLE:
            for w in (self._shape_combo, self._z_spin, self._h_spin, self._r_spin,
                      self._w_spin, self._d_spin, self._xo_spin, self._yo_spin):
                w.setEnabled(False)

        # Load saved
        try:
            shape_idx = _SHAPE_KEYS.index(s.stock_shape)
        except ValueError:
            shape_idx = 0
        self._shape_combo.blockSignals(True)
        self._shape_combo.setCurrentIndex(shape_idx)
        self._shape_combo.blockSignals(False)

        for spin, val in [
            (self._z_spin,  s.stock_z_offset_mm),
            (self._h_spin,  s.stock_height_mm),
            (self._r_spin,  s.stock_round_radius_mm),
            (self._w_spin,  s.stock_width_mm),
            (self._d_spin,  s.stock_depth_mm),
            (self._xo_spin, s.stock_x_offset_mm),
            (self._yo_spin, s.stock_y_offset_mm),
        ]:
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        self._sync_radius_row()

        # Write side
        self._shape_combo.currentIndexChanged.connect(self._on_shape)
        self._z_spin.valueChanged.connect(lambda v: setattr(s, "stock_z_offset_mm",    v))
        self._h_spin.valueChanged.connect(lambda v: setattr(s, "stock_height_mm",      v))
        self._r_spin.valueChanged.connect(lambda v: setattr(s, "stock_round_radius_mm", v))
        self._w_spin.valueChanged.connect( lambda v: setattr(s, "stock_width_mm",     v))
        self._d_spin.valueChanged.connect( lambda v: setattr(s, "stock_depth_mm",     v))
        self._xo_spin.valueChanged.connect(lambda v: setattr(s, "stock_x_offset_mm",  v))
        self._yo_spin.valueChanged.connect(lambda v: setattr(s, "stock_y_offset_mm",  v))

        # Read side — bound methods, not lambdas: see _DisplayTab's read-side
        # block above for why.
        s.stock_shape_changed.connect(self._on_shape_changed)
        s.stock_z_offset_changed.connect(self._on_z_settings_changed)
        s.stock_height_changed.connect(  self._on_h_settings_changed)
        s.stock_round_radius_changed.connect(self._on_r_settings_changed)
        s.stock_width_changed.connect(   self._on_w_settings_changed)
        s.stock_depth_changed.connect(   self._on_d_settings_changed)
        s.stock_x_offset_changed.connect(self._on_xo_settings_changed)
        s.stock_y_offset_changed.connect(self._on_yo_settings_changed)

    def _on_shape(self, idx: int) -> None:
        if 0 <= idx < len(_SHAPE_KEYS):
            self._s.stock_shape = _SHAPE_KEYS[idx]
            self._sync_radius_row()

    def _on_shape_changed(self, v: str) -> None:
        try:
            idx = _SHAPE_KEYS.index(v)
        except ValueError:
            return
        if idx != self._shape_combo.currentIndex():
            self._shape_combo.blockSignals(True)
            self._shape_combo.setCurrentIndex(idx)
            self._shape_combo.blockSignals(False)
        self._sync_radius_row()

    def _on_z_settings_changed(self, v: float) -> None:
        _sync_spin(self._z_spin, v)

    def _on_h_settings_changed(self, v: float) -> None:
        _sync_spin(self._h_spin, v)

    def _on_r_settings_changed(self, v: float) -> None:
        _sync_spin(self._r_spin, v)

    def _on_w_settings_changed(self, v: float) -> None:
        _sync_spin(self._w_spin, v)

    def _on_d_settings_changed(self, v: float) -> None:
        _sync_spin(self._d_spin, v)

    def _on_xo_settings_changed(self, v: float) -> None:
        _sync_spin(self._xo_spin, v)

    def _on_yo_settings_changed(self, v: float) -> None:
        _sync_spin(self._yo_spin, v)

    def _sync_radius_row(self) -> None:
        is_round = (self._shape_combo.currentIndex() == _SHAPE_KEYS.index("round"))
        self._r_label.setVisible(is_round)
        self._r_spin.setVisible(is_round)
        # Width/depth/corner-offset are BOUNDING_BOX-only — inverse of the
        # radius row above.
        for label, spin in self._box_rows:
            label.setVisible(not is_round)
            spin.setVisible(not is_round)


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_sections(parent: QWidget | None = None) -> list[QWidget]:
    """Build a fresh set of the 4 section widgets, in SECTION_ICONS order.

    Safe to call more than once (e.g. once for the sim widget's overlay
    panel, once for the app-wide SettingsPage) — every returned set is
    independently, bidirectionally bound to the same AppSettings singleton.
    """
    s = AppSettings.instance()
    return [
        _DisplayTab(s, parent),
        _AppearanceTab(s, parent),
        _VoxelSimTab(s, parent),
        _StockTab(s, parent),
    ]
