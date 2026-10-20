"""
sim/dexel/carver.py — DexelCarver: tool-sweep subtraction on a DexelStock.

Drop-in for VoxelCarver as far as VoxelSimController and prepass.py are
concerned: same ``carve_segment(start, end, tool)`` signature, same cutting
zone (tip up to ``cutting_length``, same fallbacks). Changed rays are marked
dirty on their DexelField; GpuDexelGrid.upload_if_dirty() pushes them.
"""
from __future__ import annotations

import numpy as np

from controller.sim.dexel.stock import DexelStock
from controller.sim.dexel.sweep import (
    ProfileInverse,
    cutting_length,
    sweep_horizontal,
    sweep_z,
)
from controller.sim.simulation.tool_definition import ToolDefinition


class DexelCarver:
    """Carves tool-swept volumes into all three ray sets of a DexelStock."""

    def __init__(self, dexels: DexelStock) -> None:
        self._dexels = dexels
        # ProfileInverse per tool geometry — building one costs ~10k profile
        # evaluations, far more than a typical short segment's carve.
        self._inverse_cache: dict[tuple, ProfileInverse] = {}

    def _inverse(self, tool: ToolDefinition, cut_len: float) -> ProfileInverse:
        key = (
            tool.tool_type, tool.diameter, tool.corner_radius,
            tool.tip_angle, tool.taper_angle, cut_len,
        )
        inv = self._inverse_cache.get(key)
        if inv is None:
            inv = self._inverse_cache[key] = ProfileInverse(tool.profile_radius_at_array, cut_len)
        return inv

    def carve_segment(
        self,
        start: np.ndarray,     # (3,) float32 — segment start, world mm
        end:   np.ndarray,     # (3,) float32 — segment end,   world mm
        tool:  ToolDefinition,
    ) -> None:
        """Subtract the swept volume of *tool* moving from *start* to *end*."""
        cut_len = cutting_length(tool)
        radius_at = tool.profile_radius_at_array

        fx, fy, fz = self._dexels.fields
        results = [
            (fz, sweep_z(fz, start, end, self._inverse(tool, cut_len), 0.0, cut_len)),
            (fx, sweep_horizontal(fx, start, end, radius_at, tool.radius, 0.0, cut_len)),
            (fy, sweep_horizontal(fy, start, end, radius_at, tool.radius, 0.0, cut_len)),
        ]
        for field, res in results:
            if res is None:
                continue
            (iv0, iv1, iu0, iu1), lo, hi = res
            field.subtract(iv0, iv1, iu0, iu1, lo, hi)
