"""
ChimeraX export utilities

Generate ChimeraX .cxc command scripts from RELION .star particle tables.

Usage (CLI):
  python -m tomopanda.utils.chimerax_export \
    --star /home/muwang/Documents/GitHub/TomoPANDA/results/particles.star \
    --out  /home/muwang/Documents/GitHub/TomoPANDA/results/particles.cxc \
    --sphere-radius 5 --arrow-length 20

Then open in ChimeraX:
  chimerax /home/muwang/Documents/GitHub/TomoPANDA/results/particles.cxc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from .relion_utils import read_star, parse_particles_from_star, RELIONConverter


def write_cxc_for_particles(
    star_path: Path,
    out_path: Path,
    *,
    sphere_radius: float = 5.0,
    arrow_length: float = 20.0,
    arrow_shaft_radius: Optional[float] = None,
    arrow_tip_radius: Optional[float] = None,
    arrow_shaft_ratio: float = 0.8,
    sphere_color: str = 'cornflower blue',
    arrow_color: str = 'orangered',
    coordinate_scale: float = 1.0,
    arrow_axis: str = 'z',
) -> None:
    """
    Create a ChimeraX .cxc script from a RELION particles .star file.

    Each particle produces:
      - a sphere at (X,Y,Z)
      - an arrow from center toward the Euler-derived direction.
    """
    df = read_star(star_path)
    parsed = parse_particles_from_star(df)
    centers: np.ndarray = parsed['centers']
    eulers: np.ndarray = parsed['eulers']

    centers_scaled = centers.astype(float) * float(coordinate_scale)

    lines = []
    lines.append('graphics silhouettes true')
    lines.append('lighting soft')

    for i, center in enumerate(centers_scaled):
        x, y, z = center.tolist()
        # sphere for position (shape sphere)
        lines.append(
            f"shape sphere center {x:.3f},{y:.3f},{z:.3f} radius {sphere_radius:.3f} color '{sphere_color}'"
        )

        tilt, psi, rot = eulers[i].tolist() if len(eulers) > i else (0.0, 0.0, 0.0)
        direction = RELIONConverter.rotation_matrix_to_direction(tilt, psi, rot, axis=arrow_axis)
        end = center + direction * float(arrow_length)

        # Decompose arrow as cylinder (shaft) + cone (tip) since 'shape arrow' is not supported
        shaft_len = float(arrow_length) * float(arrow_shaft_ratio)
        tip_len = max(0.0, float(arrow_length) - shaft_len)
        shaft_end = center + direction * shaft_len

        # Defaults if not provided
        shaft_radius = float(arrow_shaft_radius) if arrow_shaft_radius is not None else max(0.1, sphere_radius * 0.3)
        tip_radius = float(arrow_tip_radius) if arrow_tip_radius is not None else shaft_radius * 1.8

        sx, sy, sz = shaft_end.tolist()
        ex, ey, ez = end.tolist()

        # shape cylinder fromPoint->toPoint (shaft)
        lines.append(
            f"shape cylinder fromPoint {x:.3f},{y:.3f},{z:.3f} toPoint {sx:.3f},{sy:.3f},{sz:.3f} "
            f"radius {shaft_radius:.3f} caps true color '{arrow_color}'"
        )

        if tip_len > 0.0:
            # shape cone fromPoint(base)->toPoint(tip)
            lines.append(
                f"shape cone fromPoint {sx:.3f},{sy:.3f},{sz:.3f} toPoint {ex:.3f},{ey:.3f},{ez:.3f} "
                f"radius {tip_radius:.3f} topRadius 0.0 caps true color '{arrow_color}'"
            )

    lines.append('view')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Export ChimeraX .cxc from RELION .star particles')
    p.add_argument('--star', required=True, type=str, help='Path to particles.star')
    p.add_argument('--out', required=True, type=str, help='Output .cxc file path')
    p.add_argument('--sphere-radius', type=float, default=5.0, help='Sphere radius (model units)')
    p.add_argument('--arrow-length', type=float, default=20.0, help='Arrow length (model units)')
    p.add_argument('--sphere-color', type=str, default='cornflower blue', help='Sphere color')
    p.add_argument('--arrow-color', type=str, default='orangered', help='Arrow color')
    p.add_argument('--coordinate-scale', type=float, default=1.0, help='Scale factor for coordinates')
    p.add_argument('--arrow-axis', type=str, default='z', choices=['x','y','z'], help='Basis axis rotated by Euler to form arrow direction')
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    star = Path(args.star)
    out = Path(args.out)
    write_cxc_for_particles(
        star,
        out,
        sphere_radius=float(args.sphere_radius),
        arrow_length=float(args.arrow_length),
        sphere_color=str(args.sphere_color),
        arrow_color=str(args.arrow_color),
        coordinate_scale=float(args.coordinate_scale),
        arrow_axis=str(args.arrow_axis),
    )
    print(f"Wrote ChimeraX script: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


