# TomoPANDA

基于SE(3)等变变换器的CryoET膜蛋白检测工具

## 简介

TomoPANDA是一个专门用于cryoET（冷冻电子断层扫描）膜蛋白检测的工具包。它采用SE(3)等变变换器架构，能够有效处理3D断层扫描数据中的膜蛋白识别和定位任务。

## 主要特性

- **SE(3)等变变换器**: 基于几何深度学习的先进架构
- **膜蛋白检测**: 专门针对膜蛋白的检测和定位算法
- **网格测地采样**: 基于测地距离的膜蛋白采样算法
- **多格式支持**: 支持MRC、RELION等标准格式
- **命令行接口**: 简单易用的CLI工具
- **模块化设计**: 高度模块化的代码架构

## 安装

### 依赖安装

```bash
# 安装基础依赖

# 创建虚拟环境
python -m venv tomopanda

# 激活虚拟环境
# Windows
tomopanda\Scripts\activate
# Linux/Mac
source tomopanda/bin/activate

# 安装依赖
pip install -r requirements.txt

```

### 开发安装

```bash
git clone https://github.com/your-org/TomoPANDA.git
cd TomoPANDA
pip install -e .
```

## 快速开始

### 基本使用

```bash
# 查看所有可用命令
tomopanda --help

# 使用mesh geodesic采样进行粒子挑选
tomopanda sample mesh-geodesic --create-synthetic --output results/

# 使用真实膜掩码进行采样
tomopanda sample mesh-geodesic --mask membrane_mask.mrc --output results/
```

### Python API使用

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

## 命令行接口

### Sample命令 - 粒子采样

```bash
# 基本用法
tomopanda sample mesh-geodesic [OPTIONS]

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

### 其他命令

```bash
# 粒子检测
tomopanda detect --tomogram tomogram.mrc --output detections/

# 模型训练
tomopanda train --config config.yaml --data data/

# 可视化
tomopanda visualize --input results/ --output plots/

# 数据分析
tomopanda analyze --input results/ --output analysis/

# 配置管理
tomopanda config --show
tomopanda config --set parameter=value

# 版本信息
tomopanda version
```

## 项目结构

```
tomopanda/
├── cli/                        # 命令行接口
│   ├── commands/              # 命令模块
│   │   ├── sample.py          # 采样命令
│   │   ├── detect.py          # 检测命令
│   │   ├── train.py           # 训练命令
│   │   └── ...
│   └── main.py                # CLI主入口
├── core/                       # 核心算法
│   ├── mesh_geodesic.py       # 网格测地采样
│   ├── se3_transformer.py     # SE(3)变换器
│   └── ...
├── utils/                      # 工具模块
│   ├── mrc_utils.py           # MRC文件处理
│   ├── relion_utils.py        # RELION格式转换
│   └── ...
├── examples/                   # 使用示例
│   └── mesh_geodesic_example.py
└── doc/                        # 文档
    └── mesh_geodesic_algorithm.md
```

## 核心算法

### Mesh Geodesic采样

Mesh geodesic采样是一种基于网格测地距离的膜蛋白采样算法：

1. **签名距离场**: 从二值膜掩码创建SDF
2. **网格提取**: 使用Marching Cubes提取三角网格
3. **测地采样**: 基于测地距离的均匀采样
4. **后处理**: NMS和边界检查

详细算法说明请参考 [mesh_geodesic_algorithm.md](tomopanda/doc/mesh_geodesic_algorithm.md)

## 输出格式

- **CSV坐标文件**: 包含x,y,z坐标和nx,ny,nz法向量
- **RELION STAR文件**: 标准RELION格式，包含粒子坐标和欧拉角
- **先验角度文件**: 用于3D分类的角度先验

## 参数说明

### Mesh Geodesic采样参数

- `--min-distance`: 采样点间最小距离（像素）
- `--particle-radius`: 粒子半径，用于边界检查
- `--smoothing-sigma`: 高斯平滑参数
- `--taubin-iterations`: Taubin平滑迭代次数

### 输出参数

- `--tomogram-name`: 断层扫描名称
- `--particle-diameter`: 粒子直径（埃）
- `--confidence`: 置信度分数

## 性能优化

1. 对于大型数据集，考虑使用并行处理
2. 调整`min_distance`参数平衡采样密度和计算时间
3. 使用合适的`particle_radius`避免边界冲突

## 故障排除

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

## 贡献

欢迎提交Issue和Pull Request来改进项目！

## 许可证

本项目采用MIT许可证。

## 参考文献

1. Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3D surface construction algorithm.
2. Taubin, G. (1995). A signal processing approach to fair surface design.
3. Peyré, G., & Cohen, L. D. (2006). Geodesic methods for shape and surface processing.
