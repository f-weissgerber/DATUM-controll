"""
sim/dexel — Tri-dexel material-removal simulation (alternative backend).

The workpiece is stored as three orthogonal sets of parallel rays (X, Y, Z
"dexels"). Each ray holds up to K material intervals [start, end] as float32,
so the stock surface is exact along every ray direction and only quantised
crosswise (ray spacing = ``StockDefinition.voxel_size``). Memory grows with
O(N²·K) instead of the voxel grid's O(N³).

Pure Python + numpy, rendered by GLSL ray casting on the Z dexels (walls
refined from the X/Y dexels) — same OpenGL 3.3 / ModernGL requirements as
sim/voxel, which stays available as the other selectable backend
(``AppSettings.sim_backend``).

Sub-modules
-----------
field       — DexelField: one ray set + vectorised interval subtraction
stock       — DexelStock (three fields) + solid_dexels(stock)
sweep       — per-ray intersection of a tool sweep with a ray set
carver      — DexelCarver: same carve_segment() interface as VoxelCarver
collision   — check_segment() with the same rule as voxel/collision.py
gpu_fields  — GpuDexelGrid: DexelStock + ModernGL texture arrays
renderer    — DexelRenderer: GLSL ray casting, Blinn-Phong shading

Known limits
------------
• Crosswise resolution is still ``voxel_size``: oblique surfaces gain mostly
  in their normals, not in their position.
• The per-segment sweep is taken as one interval per ray, which is exact for
  a convex swept volume — true for every ToolType profile (all of them grow
  monotonically from the tip). A non-monotone profile would be carved as its
  monotone hull.
• A ray that would need more than K intervals keeps material (the last slot
  is widened) and is counted in ``DexelField.overflow_count`` + logged.
"""
