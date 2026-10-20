"""
sim/dexel/gpu_fields.py — GpuDexelGrid: DexelStock + ModernGL texture arrays.

Each ray set becomes one ``sampler2DArray`` texture, RG32F, size
(Nu, Nv, K): texel (iu, iv) of layer k holds (start, end) of slot k of
ray (iv, iu). Unused slots carry field.EMPTY_START/EMPTY_END.

Mirrors GpuVoxelGrid's role and interface for VoxelSimController /
DatumSimWidget (bbox, voxel_size, is_dirty, reset(), upload_if_dirty(),
release()); carving goes through DexelCarver(grid.dexels). Dirty tiles are
tracked per field (see field.py) and uploaded tile by tile — same
thread-safety contract as gpu_grid.py: one carve writer at a time,
upload_if_dirty() on the GL thread only.
"""
from __future__ import annotations

import moderngl
import numpy as np

from controller.sim.dexel.field import DexelField
from controller.sim.dexel.stock import DexelStock
from controller.sim.voxel.stock import BoundingBox, StockDefinition


def _texels(field: DexelField, iv0: int, iv1: int, iu0: int, iu1: int) -> bytes:
    """(K, rows, cols, 2) float32 bytes of a ray block — texture-array order."""
    s = field.start[iv0:iv1, iu0:iu1]
    e = field.end[iv0:iv1, iu0:iu1]
    return np.ascontiguousarray(np.stack([s, e], axis=-1).transpose(2, 0, 1, 3)).tobytes()


class GpuDexelGrid:
    """CPU DexelStock + one GPU texture array per ray set."""

    def __init__(
        self,
        ctx:    moderngl.Context,
        stock:  StockDefinition,
        dexels: DexelStock | None = None,
    ) -> None:
        self._ctx = ctx
        self._dexels = dexels if dexels is not None else DexelStock(stock)
        self._textures: list[moderngl.TextureArray] = []
        for f in self._dexels.fields:
            tex = ctx.texture_array(
                (f.nu, f.nv, f.k), components=2,
                data=_texels(f, 0, f.nv, 0, f.nu), dtype="f4",
            )
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            tex.repeat_x = False
            tex.repeat_y = False
            self._textures.append(tex)
            f.clear_dirty()

    # ── Read-only properties (GpuVoxelGrid-compatible) ────────────────────────

    @property
    def dexels(self) -> DexelStock:
        return self._dexels

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._dexels.shape

    @property
    def bbox(self) -> BoundingBox:
        return self._dexels.bbox

    @property
    def voxel_size(self) -> float:
        return self._dexels.voxel_size

    @property
    def textures(self) -> list[moderngl.TextureArray]:
        """X, Y, Z ray textures — bind these in the renderer."""
        return self._textures

    @property
    def is_dirty(self) -> bool:
        return self._dexels.is_dirty

    # ── Reset / GPU sync ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Refill with solid stock; the next upload_if_dirty() pushes it all."""
        self._dexels.reset()

    def upload_if_dirty(self) -> bool:
        """Upload all dirty ray tiles. GL thread only, context current.
        Returns True if anything was uploaded."""
        uploaded = False
        tile = DexelField.TILE
        for f, tex in zip(self._dexels.fields, self._textures):
            for tv, tu in f.take_dirty_tiles():
                iv0, iu0 = tv * tile, tu * tile
                iv1, iu1 = min(iv0 + tile, f.nv), min(iu0 + tile, f.nu)
                tex.write(
                    _texels(f, iv0, iv1, iu0, iu1),
                    viewport=(iu0, iv0, 0, iu1 - iu0, iv1 - iv0, f.k),
                )
                uploaded = True
        return uploaded

    def release(self) -> None:
        for tex in self._textures:
            tex.release()
        self._textures = []
