### ChimeraX Export Utility (`tomopanda/utils/chimerax_export.py`)

Generate a ChimeraX `.cxc` script from RELION `.star` particle tables. The script draws a sphere at each particle coordinate and a direction indicator derived from Euler angles (implemented as cylinder shaft + cone tip).

---

#### Features / 功能
- Spheres at `rlnCoordinateX/Y/Z`
- Orientation indicator from Euler angles `rlnAngleTilt/Psi/Rot` (RELION convention)
  - Implemented as `shape cylinder` (shaft) + `shape cone` (tip)
- Adjustable sphere radius, arrow length, colors, coordinate scale, and basis axis used to derive direction

---

#### Requirements / 依赖
- Python environment with TomoPANDA installed
- A RELION `particles.star` table with at least `rlnCoordinateX`, `rlnCoordinateY`, `rlnCoordinateZ` columns (Euler angle columns are optional)
- UCSF ChimeraX to run the generated `.cxc` script

---

#### Usage (CLI) / 使用方法（命令行）

- English
```bash
python -m tomopanda.utils.chimerax_export \
  --star /home/muwang/Documents/GitHub/TomoPANDA/results/particles.star \
  --out  /home/muwang/Documents/GitHub/TomoPANDA/results/particles.cxc \
  --sphere-radius 5 \
  --arrow-length 20 \
  --sphere-color "cornflower blue" \
  --arrow-color "orangered" \
  --coordinate-scale 1.0 \
  --arrow-axis z
```

```

Open in ChimeraX / 在 ChimeraX 中打开：
```bash
chimerax /home/muwang/Documents/GitHub/TomoPANDA/results/particles.cxc
```

---

#### Parameters / 参数说明
- `--star` (str): Path to RELION `particles.star` file / `particles.star` 文件路径
- `--out` (str): Output `.cxc` file path / 输出 `.cxc` 路径
- `--sphere-radius` (float): Sphere radius / 球半径（模型单位）
- `--arrow-length` (float): Arrow length / 箭头长度（模型单位）
- `--sphere-color` (str): Sphere color name / 球体颜色
- `--arrow-color` (str): Arrow color name / 箭头颜色
- `--coordinate-scale` (float): Scale factor for coordinates (use if your STAR coordinates need unit conversion) / 坐标缩放因子（用于单位换算）
- `--arrow-axis` (`x|y|z`): Basis axis rotated by Euler angles to form direction vector / 由欧拉角旋转的基轴，用于生成方向向量
- `--arrow-shaft-radius` (float, optional): Shaft cylinder radius; default ≈ `max(0.1, sphere_radius * 0.3)`
- `--arrow-tip-radius` (float, optional): Tip cone base radius; default ≈ `1.8 * shaft_radius`
- `--arrow-shaft-ratio` (float): Fraction of total length used for shaft (rest for cone tip); default `0.8`

---

#### Notes / 注意事项
- If Euler angles are missing, the direction defaults to the chosen basis axis with zero rotation.
- RELION Euler convention is applied via the existing `RELIONConverter` helpers.
- The generated `.cxc` uses `shape sphere`, `shape cylinder`, and `shape cone` (since `shape arrow` is not supported in ChimeraX). See: [ChimeraX shape command docs](https://www.rbvi.ucsf.edu/chimerax/docs/user/commands/shape.html).


