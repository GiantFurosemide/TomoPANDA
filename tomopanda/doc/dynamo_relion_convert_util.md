# Dynamo-RELION 转换工具

这个工具提供了将 RELION STAR 文件转换为 Dynamo 格式的功能，包括 `.tbl` (table) 和 `.vll` (volume list) 文件。

## 功能概述

### 输入格式
- **RELION 5 STAR 文件**: 包含粒子坐标和取向信息
- 支持的列包括：
  ```
  _rlnCoordinateX #1 
  _rlnCoordinateY #2 
  _rlnCoordinateZ #3 
  _rlnTomoSubtomogramRot #4 
  _rlnTomoSubtomogramTilt #5 
  _rlnTomoSubtomogramPsi #6 
  _rlnTomoName #7 
  _rlnTomoParticleId #8 
  _rlnOpticsGroup #9
  ```

### 输出格式

#### 1. Dynamo .tbl 文件
- **格式**: 35列格式，每行一个粒子
- **文档**: https://www.dynamo-em.org/w/index.php?title=Table
- **关键列说明**:
  - 列1: 颗粒序号 (从 `start_tag` 开始递增)
  - 列2: 对齐标志 (默认为 "1")
  - 列7-9: 欧拉角 (度) - phi [-180, +180], theta [0, 180], psi [-180, +180]
  - 列20: 对应的 .vll 文件中的 tomogram 行号 (从1开始)
  - 列24-26: 坐标 (float, 体素单位) - 原始 tomogram 中的坐标
  - 列31: Dynamo 模型索引 (从1开始)

#### 2. Dynamo .vll 文件
- 包含 STAR 文件中的 `_rlnTomoName`
- 顺序与行号必须与 .tbl 文件中的粒子对应

## 原始设计规范

### 函数设计: `relion_star2dynamo_vll_tbl()`

**输入:**
1. particle.star file
   - relion5 particle.star 包含以下列:
   ```
   loop_
   _rlnCoordinateX #1 
   _rlnCoordinateY #2 
   _rlnCoordinateZ #3 
   _rlnTomoSubtomogramRot #4 
   _rlnTomoSubtomogramTilt #5 
   _rlnTomoSubtomogramPsi #6 
   _rlnTomoName #7 
   _rlnTomoParticleId #8 
   _rlnOpticsGroup #9
   ```

**输出:**

1. **dynamo .tbl 文件格式**
   - 文档: https://www.dynamo-em.org/w/index.php?title=Table
   - 每行一个粒子
   - 列说明:
     - 列1: 颗粒序号
     - 列2: 1 particle for alignment
     - 列7-9: 欧拉角（in degree）phi -180，+180  thet 0，180 psi -180，+180 
     - 列20: 对应的vll 中的tomogram的行号，也就是index （从1开始）
     - 列24-26：坐标 （float，in voxel）这是特指在原始tomo中的坐标
     - 列31: 1， dynamo 中model的index（从1开始），因为一个volume可以包含多个model
   
   - 数据示例: `"1  1   0   0   0   0   td   tilt   narot   0   0   0   0   0   0   0   0   0   0   1   0   0   0   x   y   z   0   0   0   0   1 0 0 0"`

   - **设计规范**:
     - 35 columns (1-based in docs)
     - Each row is a list[str]; the table is list[list[str]]
     - Defaults: column 2 (aligned) = "1", column 31 (otag) = "1"
     - Column 1 (tag) increments per row starting from `start_tag`
     - Provide a function to map an N×M matrix into specified 1-based columns and write a `.tbl` file.

2. **dynamo .vll 文件**
   - 就是 star file 的 _rlnTomoName
   - 注意顺序与行号要与 .tbl 的particle 对应好

## 使用方法

### 基本转换

```python
from tomopanda.utils.dynamo_relion_convert_util import relion_star_to_dynamo_tbl_vll

# 转换 RELION STAR 文件到 Dynamo 格式
relion_star_to_dynamo_tbl_vll(
    star_file="particles.star",
    tbl_output="particles.tbl", 
    vll_output="particles.vll",
    start_tag=1,
    model_index=1
)
```

### 便捷转换函数

```python
from tomopanda.utils.dynamo_relion_convert_util import convert_relion_to_dynamo

# 自动命名输出文件
tbl_path, vll_path = convert_relion_to_dynamo(
    star_file="particles.star",
    output_dir="output",
    base_name="particles",
    start_tag=1,
    model_index=1
)
```

### 自定义矩阵映射

```python
from tomopanda.utils.dynamo_relion_convert_util import convert_matrix_to_tbl
import numpy as np

# 创建自定义数据矩阵
data = np.array([
    [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],  # x, y, z, phi, theta, psi
    [4.0, 5.0, 6.0, 15.0, 25.0, 35.0],
])

# 映射到 Dynamo 列: 坐标到列24-26，角度到列7-9
target_columns = [24, 25, 26, 7, 8, 9]  # x, y, z, phi, theta, psi

convert_matrix_to_tbl(
    matrix=data,
    target_columns=target_columns,
    output_path="custom.tbl",
    start_tag=1
)
```

## 高级功能

### 欧拉角转换
工具自动处理 RELION 和 Dynamo 之间的欧拉角转换：
- RELION 使用 ZYZ 约定: R = Rz(rot) * Ry(tilt) * Rz(psi)
- Dynamo 使用 ZYZ 约定但角度定义不同
- 自动确保角度在正确范围内

### 批量处理
```python
import glob
from pathlib import Path

# 处理多个 STAR 文件
star_files = glob.glob("data/*.star")
output_dir = Path("dynamo_output")
output_dir.mkdir(exist_ok=True)

for i, star_file in enumerate(star_files):
    base_name = Path(star_file).stem
    convert_relion_to_dynamo(
        star_file=star_file,
        output_dir=output_dir,
        base_name=base_name,
        start_tag=i * 1000 + 1  # 避免标签冲突
    )
```

## 示例文件

查看 `examples/` 目录中的示例文件：
- `dynamo_conversion_basic.py`: 基本转换示例
- `dynamo_conversion_advanced.py`: 高级功能示例
- `dynamo_conversion_batch.py`: 批量处理示例

## 注意事项

1. **坐标单位**: 确保输入坐标的单位与 Dynamo 期望的单位一致
2. **角度范围**: 工具会自动处理角度范围转换
3. **文件对应**: .tbl 和 .vll 文件必须保持对应关系
4. **标签唯一性**: 确保粒子标签在单个项目中唯一

## 技术细节

### 35列格式说明
```
列1:  标签 (tag)
列2:  对齐标志 (aligned) = "1"
列7-9: 欧拉角 (phi, theta, psi)
列20: tomogram 索引
列24-26: 坐标 (x, y, z)
列31: 模型索引 (otag) = "1"
```

### 数据示例
```
1  1   0   0   0   0   td   tilt   narot   0   0   0   0   0   0   0   0   0   0   1   0   0   0   x   y   z   0   0   0   0   1 0 0 0
```