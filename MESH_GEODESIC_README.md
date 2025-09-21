# Mesh Geodesic Sampling for CryoET Particle Picking

## 简介 (Introduction)

Mesh geodesic sampling是TomoPANDA中用于cryoET膜蛋白粒子挑选的算法。该算法基于网格测地距离，从膜分割掩码中生成均匀分布的采样点，为3D分类提供高质量的初始候选。

## 快速开始 (Quick Start)

### 1. 安装依赖 (Install Dependencies)

```bash
# 安装项目依赖（包含mesh geodesic依赖）
pip install -r requirements.txt

# 或手动安装mesh geodesic相关依赖
pip install mrcfile open3d scikit-image pandas scipy numpy gdist
```

### 2. 基本使用 (Basic Usage)

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

# 保存为RELION格式
convert_to_relion_star(centers, normals, "particles.star")
```

### 2. CLI使用 (CLI Usage)

```bash
# 使用TomoPANDA CLI - 推荐方式
# 使用合成数据测试
tomopanda sample mesh-geodesic --create-synthetic --output results/

# 使用真实膜掩码
tomopanda sample mesh-geodesic --mask membrane_mask.mrc --output results/

# 自定义参数
tomopanda sample mesh-geodesic \
    --mask membrane_mask.mrc \
    --output results/ \
    --min-distance 25.0 \
    --particle-radius 12.0 \
    --smoothing-sigma 2.0 \
    --verbose
```

### 3. Python API使用 (Python API Usage)

```python
# 直接使用Python API
from tomopanda.core.mesh_geodesic import create_mesh_geodesic_sampler
from tomopanda.utils.mrc_utils import load_membrane_mask
from tomopanda.utils.relion_utils import convert_to_relion_star

# 创建采样器
sampler = create_mesh_geodesic_sampler(min_distance=20.0)

# 加载膜掩码
mask = load_membrane_mask("membrane_mask.mrc")

# 执行采样
centers, normals = sampler.sample_membrane_points(mask, particle_radius=10.0)

# 保存为RELION格式
convert_to_relion_star(centers, normals, "particles.star")
```

## 算法原理 (Algorithm Principles)

1. **签名距离场**: 从二值掩码创建SDF
2. **网格提取**: 使用Marching Cubes提取三角网格
3. **测地采样**: 基于测地距离的均匀采样
4. **后处理**: NMS和边界检查

## 输出格式 (Output Formats)

- **CSV坐标文件**: 包含x,y,z坐标和nx,ny,nz法向量
- **RELION STAR文件**: 标准RELION格式，包含粒子坐标和欧拉角
- **先验角度文件**: 用于3D分类的角度先验

## 参数说明 (Parameters)

- `min_distance`: 采样点间最小距离（像素）
- `particle_radius`: 粒子半径，用于边界检查
- `smoothing_sigma`: 高斯平滑参数
- `taubin_iterations`: Taubin平滑迭代次数

## 文件结构 (File Structure)

```
tomopanda/
├── core/
│   └── mesh_geodesic.py          # 核心算法
├── utils/
│   ├── mrc_utils.py              # MRC文件处理
│   └── relion_utils.py           # RELION格式转换
├── examples/
│   └── mesh_geodesic_example.py  # 使用示例
└── doc/
    └── mesh_geodesic_algorithm.md # 详细算法文档
```

## 依赖库 (Dependencies)

- **mrcfile**: MRC文件读写
- **open3d**: 3D网格处理
- **scikit-image**: Marching Cubes算法
- **gdist**: 测地距离计算
- **pandas**: 数据格式转换

## 性能优化 (Performance Tips)

1. 对于大型数据集，考虑使用并行处理
2. 调整`min_distance`参数平衡采样密度和计算时间
3. 使用合适的`particle_radius`避免边界冲突

## 故障排除 (Troubleshooting)

### 常见问题

1. **ImportError: No module named 'gdist'**
   ```bash
   pip install gdist
   ```

2. **ImportError: No module named 'open3d'**
   ```bash
   pip install open3d
   ```

3. **内存不足**
   - 减小输入数据尺寸
   - 增加`min_distance`参数

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行采样算法
sampler = create_mesh_geodesic_sampler()
centers, normals = sampler.sample_membrane_points(mask)
```

## 贡献 (Contributing)

欢迎提交Issue和Pull Request来改进算法！

## 许可证 (License)

本项目采用MIT许可证。
