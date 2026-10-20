"""
sim/dexel/stock.py — DexelStock: the three ray fields of one workpiece.

Reuses voxel/stock.py's StockDefinition/BoundingBox unchanged (shape, bbox,
resolution); ``voxel_size`` is the ray spacing h here. The ray grid covers
the same (Nx, Ny, Nz) = bbox.grid_shape(h) cells the voxel grid would, so
both backends sample the stock at identical cell centres.

Initial intervals are exact: BOUNDING_BOX rays span the bbox, ROUND (a
cylinder around the work origin) gives X/Y rays their exact chord and Z rays
the full height inside the circle.

Pure numpy, no GL — usable from background threads (material build,
collision pre-pass) exactly like gpu_grid.solid_material().
"""
from __future__ import annotations

import numpy as np

from controller.sim.dexel.field import PLANE_AXES, DexelField
from controller.sim.voxel.stock import BoundingBox, StockDefinition, StockShape

# Default interval slots per ray. A 3-axis part's Z rays mostly hold one
# interval, but a tool whose cutting length is shorter than the depth it
# is driven to (a shank collision the simulation still carves) leaves a
# band of material above each pass — workpieces/Gcode.cnc's 1 mm cutting
# length end mill needs 3+. X/Y rays cross every pocket wall.
DEFAULT_K_Z:  int = 4
DEFAULT_K_XY: int = 8


class DexelStock:
    """Three DexelFields (index 0/1/2 = rays along X/Y/Z) + stock geometry."""

    def __init__(
        self,
        stock: StockDefinition,
        k_xy:  int = DEFAULT_K_XY,
        k_z:   int = DEFAULT_K_Z,
    ) -> None:
        if stock.bbox is None:
            raise RuntimeError("Call stock.build_bbox(path) before building a DexelStock")
        self.stock = stock
        self.bbox: BoundingBox = stock.bbox
        self.voxel_size: float = stock.voxel_size
        nx, ny, nz = stock.grid_shape
        self.shape = (nx, ny, nz)

        origin = self.bbox.origin()
        h = self.voxel_size
        dims = (nx, ny, nz)
        self.fields: tuple[DexelField, DexelField, DexelField] = tuple(  # type: ignore[assignment]
            DexelField(
                axis, origin, h,
                nu=dims[PLANE_AXES[axis][0]],
                nv=dims[PLANE_AXES[axis][1]],
                k=k_z if axis == 2 else k_xy,
            )
            for axis in (0, 1, 2)
        )
        self.reset()

    # ── Initial material ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Refill all rays with the solid stock shape (marks everything dirty)."""
        b = self.bbox
        lo = (b.x_min, b.y_min, b.z_min)
        hi = (b.x_max, b.y_max, b.z_max)
        for f in self.fields:
            u = f.u_centers()[np.newaxis, :]     # (1, Nu)
            v = f.v_centers()[:, np.newaxis]     # (Nv, 1)
            start = np.full((f.nv, f.nu), lo[f.axis], dtype="f4")
            end   = np.full((f.nv, f.nu), hi[f.axis], dtype="f4")
            if self.stock.shape == StockShape.ROUND:
                cx, cy = b.xy_center()
                r = self.stock.round_radius_mm
                if f.axis == 2:
                    # Z rays: inside the circle → full height, else empty.
                    outside = (u - cx) ** 2 + (v - cy) ** 2 > r * r
                    start = np.where(outside, np.float32(1.0), start)
                    end   = np.where(outside, np.float32(0.0), end)
                else:
                    # X rays sit at (y=u, z=v), Y rays at (x=u, z=v): the
                    # other horizontal coordinate is u in both cases.
                    c_other = cy if f.axis == 0 else cx
                    c_ray   = cx if f.axis == 0 else cy
                    d2 = (u - c_other) ** 2 + np.zeros_like(v)
                    half = np.sqrt(np.maximum(r * r - d2, 0.0)).astype("f4")
                    inside = d2 <= r * r
                    start = np.where(inside, c_ray - half, np.float32(1.0)).astype("f4")
                    end   = np.where(inside, c_ray + half, np.float32(0.0)).astype("f4")
            f.fill(start, end)

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def z_field(self) -> DexelField:
        return self.fields[2]

    def volume_mm3(self, axis: int = 2) -> float:
        """Material volume as seen by one ray set (h² × summed lengths)."""
        return self.fields[axis].total_length() * self.voxel_size ** 2

    def occupancy(self) -> np.ndarray:
        """(Nz, Ny, Nx) bool: material at each cell centre according to the
        Z rays — same layout as GpuVoxelGrid.material, for comparisons/tests."""
        nx, ny, nz = self.shape
        f = self.z_field
        zs = self.bbox.origin()[2] + (np.arange(nz, dtype="f4") + 0.5) * self.voxel_size
        s = f.start[np.newaxis]                       # (1, Ny, Nx, K)
        e = f.end[np.newaxis]
        z = zs[:, np.newaxis, np.newaxis, np.newaxis]  # (Nz, 1, 1, 1)
        return np.any((s <= z) & (z <= e), axis=-1)

    # ── Dirty state (shared by all three fields) ──────────────────────────────

    @property
    def is_dirty(self) -> bool:
        return any(f.is_dirty for f in self.fields)


def solid_dexels(stock: StockDefinition) -> DexelStock:
    """Dexel counterpart of gpu_grid.solid_material(): a fresh, fully solid
    DexelStock for *stock* (bbox must be set). GL-free — call it from a
    background thread."""
    return DexelStock(stock)
