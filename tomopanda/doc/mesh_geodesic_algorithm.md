# Mesh Geodesic Sampling Algorithm for CryoET Particle Picking

## Overview

Mesh geodesic sampling is a membrane protein sampling algorithm based on mesh geodesic distance, specifically designed for particle picking in cryoET. The algorithm converts membrane segmentation masks to triangular meshes and performs uniform sampling using geodesic distance, providing high-quality initial candidate points for 3D classification.

## Algorithm Principles

### 1. Signed Distance Field (SDF)
- Create signed distance field from binary membrane mask
- φ > 0 indicates outside membrane, φ < 0 indicates inside membrane
- Use Gaussian smoothing to reduce noise
- **Adaptive resolution**: Adjust SDF resolution based on expected particle size

### 2. Mesh Extraction
- Use Marching Cubes algorithm to extract triangular mesh at φ=0 isosurface
- **Multi-level mesh control**:
  - **Level 1**: Adaptive SDF resolution based on particle size
  - **Level 2**: Adaptive Marching Cubes spacing
  - **Level 3**: Taubin smoothing with particle-size-dependent iterations
  - **Level 4**: Mesh decimation for large particles / subdivision for small particles
- Support random noise injection to generate different mesh variants
- Compute vertex normals

### 3. Triangle Extraction
- Calculate all triangle center coordinates
- Compute surface normals for each triangle
- Align normal directions with SDF gradient
- Output N*6 format data: [x, y, z, nx, ny, nz]

### 4. Adaptive Mesh Density Control

The algorithm now supports **direct control of mesh hole size** based on expected particle size:

| Particle Size | Triangle Size | Taubin Iterations | Mesh Processing | Result |
|---------------|---------------|-------------------|-----------------|---------|
| ≤ 5 pixels | 1.7 pixels | 3-5 | Subdivision | Very fine mesh |
| 6-20 pixels | 2-6.7 pixels | 5-12 | Standard | Fine mesh |
| 21-50 pixels | 7-16.7 pixels | 12-18 | Standard | Medium mesh |
| 51-100 pixels | 17-33.3 pixels | 18-25 | Decimation | Coarse mesh |
| > 100 pixels | > 33.3 pixels | 25-35 | Decimation | Very coarse mesh |

## Core Classes and Methods

### MeshGeodesicSampler Class

```python
class MeshGeodesicSampler:
    def __init__(self, 
                 smoothing_sigma=1.5, 
                 taubin_iterations=10, 
                 expected_particle_size=None,  # Direct mesh density control
                 random_seed=None)
    def sample_membrane_points(self, mask, particle_radius=10.0, volume_shape=None)
    def get_all_triangle_centers_and_normals(self, mask)  # Extract all triangles
```

### Key Methods

1. **create_signed_distance_field(mask, target_resolution=None)**: Create SDF with adaptive resolution
2. **extract_mesh_from_sdf(phi, spacing=None)**: Extract mesh with adaptive spacing
3. **get_triangle_centers_and_normals(mesh, phi)**: Extract triangle centers and normals
4. **get_all_triangle_centers_and_normals(mask)**: Complete extraction pipeline

### New Features

- **Adaptive mesh density**: Direct control based on `expected_particle_size`
- **Multi-level processing**: 4 levels of mesh optimization
- **Automatic parameter selection**: No need to manually tune taubin iterations

## Input/Output Formats

### Input
- **Membrane mask**: MRC format binary mask (0/1)
- **Expected particle size**: Parameter to control mesh density (optional)
- **Random seed**: Parameter to control mesh variants (optional)

### Output
- **Triangle data**: N*6 array, each row contains [x, y, z, nx, ny, nz]
- **Coordinate file**: CSV format with x,y,z coordinates and nx,ny,nz normal vectors
- **RELION STAR file**: Standard RELION format with particle coordinates and Euler angles
- **Prior angles file**: Angle priors for 3D classification

## Parameters

### Core Parameters
- **smoothing_sigma**: Gaussian smoothing parameter (default: 1.5)
- **taubin_iterations**: Taubin smoothing iterations (default: 10) - **mutually exclusive with expected_particle_size**
- **expected_particle_size**: Expected particle size in pixels for mesh density control - **automatically calculates sampling distance and taubin iterations** (mutually exclusive with taubin_iterations)
- **random_seed**: Random seed for mesh generation (None for deterministic)

### Parameter Relationships

**Important**: `expected_particle_size` and `taubin_iterations` are mutually exclusive:

| Parameter | Effect | When to Use |
|-----------|--------|-------------|
| `expected_particle_size` | Automatic mesh density control | When you know particle size |
| `taubin_iterations` | Manual mesh smoothing control | When you want precise control |

**Automatic mapping** when using `expected_particle_size`:
- Small particles (≤20px) → Fine mesh (3-12 iterations) + Dense sampling (distance = particle_size/2)
- Medium particles (21-50px) → Medium mesh (12-18 iterations) + Medium sampling
- Large particles (>50px) → Coarse mesh (18-35 iterations) + Sparse sampling

## 使用示例 (Usage Examples)

### 基本使用

```python
from tomopanda.core.mesh_geodesic import get_triangle_centers_and_normals
from tomopanda.utils.mrc_utils import load_membrane_mask
from tomopanda.utils.relion_utils import convert_to_relion_star

# 提取所有三角形中心坐标和法向量
triangle_data = get_triangle_centers_and_normals(
    mask=mask,
    expected_particle_size=100.0,
    random_seed=42
)
# triangle_data 是 N*6 数组: [x, y, z, nx, ny, nz]

# 分离坐标和法向量
centers = triangle_data[:, :3]  # x, y, z
normals = triangle_data[:, 3:]  # nx, ny, nz

# 转换为RELION格式
convert_to_relion_star(centers, normals, "particles.star")
```

### 命令行使用

```bash
# 提取所有三角形中心
tomopanda sample mesh-geodesic --create-synthetic --output results/

# 使用真实膜掩码
tomopanda sample mesh-geodesic --mask membrane_mask.mrc --output results/

# 控制mesh密度和随机性
tomopanda sample mesh-geodesic --create-synthetic --expected-particle-size 20.0 --random-seed 42 --output results/

# 生成不同mesh变体
tomopanda sample mesh-geodesic --create-synthetic --random-seed 42 --output variant_1
tomopanda sample mesh-geodesic --create-synthetic --random-seed 123 --output variant_2
```

## 参数说明 (Parameter Description)

### 核心参数
- **smoothing_sigma**: 高斯平滑参数，默认1.5
- **taubin_iterations**: Taubin平滑迭代次数，默认10（与expected_particle_size互斥）
- **expected_particle_size**: 期望颗粒大小（像素），自动控制mesh密度和采样距离，默认None
- **random_seed**: 随机种子，用于生成不同的mesh变体，默认None

### 输出参数
- **tomogram_name**: 断层扫描名称
- **particle_diameter**: 粒子直径（埃）
- **confidence**: 置信度分数

## 算法优势 (Algorithm Advantages)

1. **完整覆盖**: 提取所有三角形中心，确保膜表面完整覆盖
2. **方向信息**: 提供表面法向量，可用于姿态先验
3. **mesh变体**: 支持随机种子生成不同的mesh变体
4. **密度控制**: 通过expected_particle_size控制mesh细密程度
5. **格式兼容**: 支持多种输出格式，特别是RELION

## 技术细节 (Technical Details)

### 依赖库
- **mrcfile**: MRC文件读写
- **open3d**: 网格处理和可视化
- **scikit-image**: Marching Cubes算法
- **gdist**: 测地距离计算
- **pandas**: 数据格式转换

### 性能考虑
- 三角形提取比测地采样更高效
- 内存使用量与网格复杂度成正比
- 支持大规模数据集处理

## 应用场景 (Application Scenarios)

1. **膜蛋白检测**: 基于膜分割的蛋白质粒子挑选
2. **3D分类**: 为3D分类提供高质量的初始候选
3. **mesh变体**: 生成多个mesh变体提高检测覆盖率
4. **结构生物学**: 膜蛋白复合物的结构分析

## 未来改进 (Future Improvements)

1. **并行化**: 支持多进程/多线程处理
2. **自适应密度**: 根据局部曲率调整mesh密度
3. **机器学习集成**: 结合深度学习模型优化mesh生成
4. **实时处理**: 支持流式数据处理

## 参考文献 (References)

1. Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3D surface construction algorithm.
2. Taubin, G. (1995). A signal processing approach to fair surface design.
3. Peyré, G., & Cohen, L. D. (2006). Geodesic methods for shape and surface processing.
4. Crane, K., Weischedel, C., & Wardetzky, M. (2013). Geodesics in heat: A new approach to computing distance based on heat flow.
