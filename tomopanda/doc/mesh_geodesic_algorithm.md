# Mesh Geodesic Sampling Algorithm for CryoET Particle Picking

## 概述 (Overview)

Mesh geodesic sampling是一种基于网格测地距离的膜蛋白采样算法，专门用于cryoET中的粒子挑选。该算法将膜分割掩码转换为三角网格，然后使用测地距离进行均匀采样，为3D分类提供高质量的初始候选点。

## 算法原理 (Algorithm Principles)

### 1. 签名距离场 (Signed Distance Field)
- 从二值膜掩码创建签名距离场 (SDF)
- φ > 0 表示膜外，φ < 0 表示膜内
- 使用高斯平滑减少噪声

### 2. 网格提取 (Mesh Extraction)
- 使用Marching Cubes算法在φ=0等值面上提取三角网格
- 应用Taubin平滑算法优化网格质量
- 计算顶点法向量

### 3. 测地采样 (Geodesic Sampling)
- 使用测地最远点采样 (Geodesic Farthest Point Sampling)
- 近似Poisson-disk采样，确保采样点间的最小测地距离
- 使用gdist库计算测地距离

### 4. 后处理 (Post-processing)
- 非极大值抑制 (NMS) 去除重叠采样点
- 边界检查确保粒子可放置性
- 输出采样点坐标和表面法向量

## 核心类和方法 (Core Classes and Methods)

### MeshGeodesicSampler类

```python
class MeshGeodesicSampler:
    def __init__(self, smoothing_sigma=1.5, taubin_iterations=10, min_distance=20.0)
    def sample_membrane_points(self, mask, particle_radius=10.0, volume_shape=None)
```

### 主要方法

1. **create_signed_distance_field(mask)**: 创建签名距离场
2. **extract_mesh_from_sdf(phi)**: 从SDF提取网格
3. **geodesic_farthest_point_sampling(vertices, faces, radius)**: 测地采样
4. **apply_non_maximum_suppression(centers, normals, min_distance)**: NMS
5. **check_placement_feasibility(centers, normals, volume_shape, particle_radius)**: 边界检查

## 输入输出格式 (Input/Output Formats)

### 输入 (Input)
- **膜掩码**: MRC格式的二值掩码 (0/1)
- **粒子半径**: 用于边界检查的粒子半径
- **最小距离**: 采样点间的最小距离

### 输出 (Output)
- **坐标文件**: CSV格式，包含x,y,z坐标和nx,ny,nz法向量
- **RELION STAR文件**: 标准RELION格式，包含粒子坐标和欧拉角
- **先验角度文件**: 用于3D分类的角度先验

## 使用示例 (Usage Examples)

### 基本使用

```python
from tomopanda.core.mesh_geodesic import create_mesh_geodesic_sampler
from tomopanda.utils.mrc_utils import load_membrane_mask
from tomopanda.utils.relion_utils import convert_to_relion_star

# 创建采样器
sampler = create_mesh_geodesic_sampler(min_distance=20.0)

# 加载膜掩码
mask = load_membrane_mask("membrane_mask.mrc")

# 执行采样
centers, normals = sampler.sample_membrane_points(mask, particle_radius=10.0)

# 转换为RELION格式
convert_to_relion_star(centers, normals, "particles.star")
```

### 命令行使用

```bash
python examples/mesh_geodesic_example.py \
    --mask membrane_mask.mrc \
    --output sampling_results/ \
    --min-distance 20.0 \
    --particle-radius 10.0
```

## 参数说明 (Parameter Description)

### 核心参数
- **smoothing_sigma**: 高斯平滑参数，默认1.5
- **taubin_iterations**: Taubin平滑迭代次数，默认10
- **min_distance**: 采样点间最小距离，默认20.0像素
- **particle_radius**: 粒子半径，用于边界检查

### 输出参数
- **tomogram_name**: 断层扫描名称
- **particle_diameter**: 粒子直径（埃）
- **confidence**: 置信度分数

## 算法优势 (Algorithm Advantages)

1. **几何一致性**: 基于测地距离确保采样点在膜表面上的均匀分布
2. **方向信息**: 提供表面法向量，可用于姿态先验
3. **边界安全**: 自动检查粒子放置的可行性
4. **格式兼容**: 支持多种输出格式，特别是RELION

## 技术细节 (Technical Details)

### 依赖库
- **mrcfile**: MRC文件读写
- **open3d**: 网格处理和可视化
- **scikit-image**: Marching Cubes算法
- **gdist**: 测地距离计算
- **pandas**: 数据格式转换

### 性能考虑
- 测地距离计算是算法的主要瓶颈
- 建议对大型数据集使用并行处理
- 内存使用量与网格复杂度成正比

## 应用场景 (Application Scenarios)

1. **膜蛋白检测**: 基于膜分割的蛋白质粒子挑选
2. **3D分类**: 为3D分类提供高质量的初始候选
3. **质量控制**: 基于几何的采样质量评估
4. **结构生物学**: 膜蛋白复合物的结构分析

## 未来改进 (Future Improvements)

1. **并行化**: 支持多进程/多线程处理
2. **自适应采样**: 根据局部曲率调整采样密度
3. **机器学习集成**: 结合深度学习模型优化采样
4. **实时处理**: 支持流式数据处理

## 参考文献 (References)

1. Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3D surface construction algorithm.
2. Taubin, G. (1995). A signal processing approach to fair surface design.
3. Peyré, G., & Cohen, L. D. (2006). Geodesic methods for shape and surface processing.
4. Crane, K., Weischedel, C., & Wardetzky, M. (2013). Geodesics in heat: A new approach to computing distance based on heat flow.
