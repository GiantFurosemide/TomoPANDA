## Mesh Geodesic Sampling for CryoET Particle Picking / 网格测地采样（中英双语）

This document describes the developer-level design and implementation details of the Mesh-Geodesic sampling pipeline implemented in `tomopanda/core/mesh_geodesic.py`. It focuses on APIs, algorithmic steps, parameter semantics, data formats, and practical guidance for extending and integrating the module.

本文档面向开发者，系统性阐述 `tomopanda/core/mesh_geodesic.py` 的实现，包括 API、算法流程、参数语义、数据格式与工程实践建议，便于扩展与集成。

---

### 1. High-level Overview / 高层概览

- **Goal / 目标**: Convert a binary membrane mask into an Open3D triangular mesh, align face normals with SDF gradients, and sample particle centers with spacing governed by expected particle size. Output standard files for downstream tools (RELION, ChimeraX, etc.).
- **Pipeline / 流水线**:
  1) Preprocess mask with Gaussian blur → Signed Distance Field (SDF)
  2) Marching Cubes on φ=0 → Open3D triangle mesh
  3) Adaptive mesh processing (smoothing/decimation/subdivision)
  4) Compute SDF-aligned face normals and centers
  5) Distance-constrained sampling of centers
  6) Boundary feasibility check and file export

---

### 2. Public API / 公共 API

All public APIs are implemented in `mesh_geodesic.py` and re-exported for convenient use.

所有公共 API 均实现在 `mesh_geodesic.py`，便于直接调用。

```python
class MeshGeodesicSampler:
    def __init__(
        self,
        smoothing_sigma: float = 1.5,
        taubin_iterations: int = 10,
        expected_particle_size: Optional[float] = None,
        random_seed: Optional[int] = None,
        add_noise: bool = False,
        noise_scale_factor: float = 0.1,
    ) -> None: ...

    def create_signed_distance_field(
        self, mask: np.ndarray, target_resolution: float = None
    ) -> np.ndarray: ...

    def extract_mesh_from_sdf(
        self, phi: np.ndarray, spacing: Tuple[float, float, float] = None
    ) -> o3d.geometry.TriangleMesh: ...

    def sample_membrane_points(
        self,
        mask: np.ndarray,
        particle_radius: float = 10.0,
        volume_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def get_all_triangle_centers_and_normals(
        self, mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray: ...
```

Factory/helper functions / 工厂与便捷函数：

```python
def create_mesh_geodesic_sampler(
    smoothing_sigma: float = 1.5,
    taubin_iterations: int = 10,
    expected_particle_size: Optional[float] = None,
    random_seed: Optional[int] = None,
    add_noise: bool = False,
    noise_scale_factor: float = 0.1,
) -> MeshGeodesicSampler: ...

def run_mesh_geodesic_sampling(
    mask: np.ndarray,
    *,
    smoothing_sigma: float = 1.5,
    taubin_iterations: int = 10,
    particle_radius: float = 10.0,
    volume_shape: Optional[Tuple[int, int, int]] = None,
    expected_particle_size: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]: ...

def get_triangle_centers_and_normals(
    mask: np.ndarray,
    *,
    expected_particle_size: Optional[float] = None,
    smoothing_sigma: float = 1.5,
    taubin_iterations: int = 10,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    random_seed: Optional[int] = None,
) -> np.ndarray: ...

def save_sampling_outputs(
    output_dir: Union[str, Path],
    centers: np.ndarray,
    normals: np.ndarray,
    *,
    mrc_path: Optional[Union[str, Path]] = None,
    tomogram_name: str = "tomogram",
    particle_diameter: float = 200.0,
    voxel_size: Optional[Tuple[float, float, float]] = None,
    sigma_tilt: float = 30.0,
    sigma_psi: float = 30.0,
    sigma_rot: float = 30.0,
    create_vis_script: bool = False,
    use_simplified_relion: bool = True,
) -> Tuple[Path, Path, Path, Optional[Path]]: ...
```

---

### 3. Parameter Semantics / 参数语义

- **expected_particle_size (px)**: Primary control knob. Governs sampling distance and mesh density.
  - Sampling distance: `max(5.0, expected_particle_size / 2.0)`
  - SDF target resolution: `max(0.5, min(3.0, expected_particle_size / 5.0))`
  - Marching Cubes spacing: `(scale, scale, scale)`, where `scale = max(0.5, min(5.0, expected_particle_size / 10.0))`
  - Taubin iterations: adaptively mapped by size brackets (see Section 4)
- **taubin_iterations**: Used if `expected_particle_size` is None; manual smoothing control.
- **smoothing_sigma**: Gaussian blur on the mask before SDF.
- **random_seed**: Controls randomness for reproducible meshes (noise injection, sampling order).
- **add_noise / noise_scale_factor**: Optional Gaussian noise added to smoothed mask prior to SDF; use to produce mesh variants.

参数要点：若设置 `expected_particle_size`，系统自动推导采样距离、网格密度与平滑强度；否则使用 `taubin_iterations` 进行手动控制。

---

### 4. Algorithms / 算法细节

#### 4.1 Signed Distance Field (SDF) / 符号距离场

- Mask smoothing → threshold at 0.5 → inside/outside binary.
- SDF computed as: `edt(1 - inside) - edt(inside)` so that φ>0 outside, φ<0 inside.
- Optional down/up-sampling based on `target_resolution` derived from `expected_particle_size`.

#### 4.2 Mesh Extraction via Marching Cubes / Marching Cubes 网格提取

- Extract isosurface at φ=0 to get `(verts, faces)`.
- Convert to Open3D `TriangleMesh` with coordinate order corrected to (x,y,z).
- Clean-up: remove duplicated vertices and degenerate triangles.

#### 4.3 Adaptive Mesh Processing / 自适应网格处理

- If `expected_particle_size` is set:
  - Compute a target triangle size ≈ `expected_particle_size / 3.0`.
  - Map size to `taubin_iterations` bracket:
    - ≤1 → 3; ≤5 → 5; ≤10 → 8; ≤20 → 12; ≤50 → 18; ≤100 → 25; ≤200 → 30; else 35
  - For very large particles (target_triangle_size > 15): quadric decimation to target triangle count `~ 1000 / (particle_size / 10)` (min 50)
  - For very small particles (target_triangle_size < 3): one iteration midpoint subdivision
- Else: use fixed `taubin_iterations`.
- Vertex normals are computed after processing.

#### 4.4 Normal Alignment with SDF Gradient / 法向与 SDF 梯度对齐

- Compute SDF gradients `(gx, gy, gz)` on the φ grid.
- For each face center (or vertex), sample nearest gradient vector and normalize.
- Flip normals when dot(normal, gradient) < 0 to point outward (increasing φ).
- A global orientation step may flip triangle winding when mean alignment is negative.

#### 4.5 Face Sampling with Distance Constraint / 距离约束的面采样

- Compute all face centers and aligned normals.
- Progressive Poisson-like sampling using a spatial hash grid:
  - Grid size = `max(1, int(min_distance))`
  - Accept a center if it is at least `min_distance` away from existing samples
- Apply Non-Maximum Suppression (NMS) as a final pruning step using KD-tree radius queries.

#### 4.6 Feasibility Check / 可行性检查

- Discard samples that are closer than `particle_radius` to any volume boundary `(0, volume_shape)` in all axes.

---

### 5. I/O Formats / 输入输出格式

`save_sampling_outputs` produces standard artifacts under `output_dir`:

- `sampling_coordinates.csv` (float64 CSV): columns `[x, y, z, nx, ny, nz]`
- `particles.star` (simplified RELION STAR when `use_simplified_relion=True`):
  - `_rlnCoordinateX`, `_rlnCoordinateY`, `_rlnCoordinateZ`
  - `_rlnAngleTilt`, `_rlnAnglePsi`, `_rlnAngleRot` (normals converted via `RELIONConverter`)
  - `_rlnTomoName`, `_rlnTomoParticleId`, `_rlnTomoParticleDiameter`
- `coordinates.csv` (float64 CSV): position and normal vectors (for scripts/pipelines)
- `prior_angles.csv`: Euler priors derived from normals
- Optional: `visualize_results.py` (matplotlib-based quick viewer)

When `mrc_path` is provided and `voxel_size` is not, voxel size is read from MRC header and used to scale coordinates for RELION outputs.

---

### 6. Usage Examples / 使用示例

#### 6.1 Python

```python
from tomopanda.core.mesh_geodesic import (
    create_mesh_geodesic_sampler,
    run_mesh_geodesic_sampling,
    save_sampling_outputs,
)

sampler = create_mesh_geodesic_sampler(
    smoothing_sigma=1.5,
    expected_particle_size=20.0,  # drives spacing & sampling distance
    random_seed=42,
)

centers, normals = run_mesh_geodesic_sampling(
    mask,
    particle_radius=10.0,
    expected_particle_size=20.0,
    random_seed=42,
)

save_sampling_outputs(
    output_dir="results/my_run",
    centers=centers,
    normals=normals,
    mrc_path=None,
    tomogram_name="demo_tomo",
    particle_diameter=200.0,
    create_vis_script=True,
    use_simplified_relion=True,
)
```

#### 6.2 CLI (via `tomopanda/cli`) / 命令行

```bash
tomopanda sample mesh-geodesic \
  --mask membrane_mask.mrc \
  --expected-particle-size 20.0 \
  --output results/
```

---

### 7. Performance & Limits / 性能与限制

- Complexity scales with number of triangles generated by Marching Cubes.
- Use `expected_particle_size` to coarsen meshes for large structures (fewer faces → faster sampling).
- Memory usage is roughly linear in number of faces; tune expected size and spacing accordingly.
- KD-tree and hash-grid based sampling are efficient for typical CryoET volumes.

---

### 8. Edge Cases & Troubleshooting / 边界与排错

- SDF empty or extremely thin structures: φ=0 may produce sparse or no triangles. Consider smaller `smoothing_sigma` or different `expected_particle_size`.
- Normals orientation appears inverted: ensure SDF sign convention is preserved; try without noise; verify φ>0 outside.
- Visualization issues (e.g., Matplotlib `Poly3DCollection`): always convert triangle indices to vertex coordinates `(N, 3, 3)` before plotting.
- Boundary filtering removes most points: check `particle_radius` and `volume_shape` consistency `(x,y,z)`.
- RELION scaling: provide `mrc_path` or explicit `voxel_size` to ensure physical units.

---

### 9. Extensibility Notes / 可扩展性说明

- Mesh processing hooks: integrate curvature-aware decimation/subdivision by inspecting `target_triangle_size`.
- Alternative sampling strategies: replace `_sample_faces_with_distance_constraint` with blue-noise or curvature-weighted schemes.
- Normal alignment: plug in robust gradient sampling or smooth gradient fields to reduce flip-noise on rough meshes.

---

### 10. References / 参考

1) Lorensen & Cline (1987) Marching Cubes
2) Taubin (1995) Signal processing approach to fair surface design
3) Peyré & Cohen (2006) Geodesic methods for shape and surface processing
4) Crane et al. (2013) Geodesics in heat

