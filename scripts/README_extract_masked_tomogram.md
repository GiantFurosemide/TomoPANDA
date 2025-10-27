# Masked Tomogram 提取工具

这个工具使用 `mrcfile` 和 `numpy` 来扩展mask并提取tomogram的特定区域，同时用高斯白噪声填充背景。

## 功能特点

- **纯Python实现**：不依赖RELION，使用mrcfile和numpy进行所有计算
- **高斯模糊mask扩展**：使用scipy的高斯滤波器平滑扩展mask边界
- **智能噪声生成**：根据tomogram的统计特性生成匹配的高斯白噪声
- **边界平滑**：可选的形态学操作进一步平滑mask边界
- **详细类型注解**：所有函数都有完整的输入输出类型说明

## 安装依赖

```bash
pip install mrcfile numpy scipy
```

## 使用方法

### 基本用法

```bash
python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc
```

### 高级用法

```bash
# 自定义mask扩展程度
python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc --sigma 3.0

# 调整噪声水平
python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc --noise_std 0.2

# 指定扩展像素数（覆盖sigma参数）
python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc --expand_pixels 5

# 增加边界平滑
python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc --smooth_iterations 2

# 覆盖已存在的输出文件
python extract_masked_tomogram.py --input_tomo input_tomo.mrc --input_mask input_mask.mrc --output_tomo output.mrc --overwrite
```

## 参数说明

### 必需参数
- `--input_tomo`: 输入tomogram文件路径 (.mrc)
- `--input_mask`: 输入mask文件路径 (.mrc，0和1的二值图像)
- `--output_tomo`: 输出tomogram文件路径 (.mrc)

### 可选参数
- `--sigma`: 高斯模糊标准差，用于mask扩展 (默认: 2.0)
- `--threshold`: mask二值化阈值 (默认: 0.5)
- `--noise_std`: 背景噪声标准差，相对于tomogram标准差的比例 (默认: 0.1)
- `--background_value`: 背景基础值 (默认: 0.0)
- `--expand_pixels`: 直接指定扩展像素数，覆盖sigma参数
- `--smooth_iterations`: mask边界平滑迭代次数 (默认: 1)
- `--overwrite`: 覆盖已存在的输出文件

## 函数说明

### 核心函数

#### `read_mrc_file(filepath: str) -> Tuple[np.ndarray, dict]`
- **输入**: MRC文件路径
- **输出**: (数据数组, 元数据字典)
- **功能**: 读取MRC文件并返回3D numpy数组和元数据

#### `write_mrc_file(filepath: str, data: np.ndarray, metadata: dict, overwrite: bool = False) -> None`
- **输入**: 输出文件路径, 3D数据数组, 元数据字典, 是否覆盖
- **输出**: None
- **功能**: 写入MRC文件

#### `expand_mask_gaussian(mask: np.ndarray, sigma: float = 2.0, threshold: float = 0.5) -> np.ndarray`
- **输入**: 二值mask数组, 高斯模糊标准差, 二值化阈值
- **输出**: 扩展后的mask数组
- **功能**: 使用高斯模糊扩展mask

#### `generate_gaussian_noise(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, dtype: np.dtype = np.float32) -> np.ndarray`
- **输入**: 噪声数组形状, 噪声均值, 噪声标准差, 数据类型
- **输出**: 高斯噪声数组
- **功能**: 生成高斯白噪声

#### `extract_masked_tomogram(tomogram: np.ndarray, mask: np.ndarray, noise_std: float = 0.1, background_value: float = 0.0) -> np.ndarray`
- **输入**: 原始tomogram数据, mask数组, 背景噪声标准差, 背景基础值
- **输出**: 处理后的tomogram
- **功能**: 提取mask区域内的tomogram并用噪声填充背景

#### `smooth_mask_boundary(mask: np.ndarray, iterations: int = 1) -> np.ndarray`
- **输入**: 输入mask, 平滑迭代次数
- **输出**: 平滑后的mask
- **功能**: 平滑mask边界

## 工作流程

1. **读取文件**: 使用mrcfile读取tomogram和mask文件
2. **扩展mask**: 使用高斯模糊和二值化扩展mask
3. **平滑边界**: 可选的形态学操作平滑mask边界
4. **提取区域**: 提取mask区域内的tomogram数据
5. **添加噪声**: 在背景区域添加高斯白噪声
6. **保存结果**: 将处理后的数据保存为MRC文件

## 输出信息

脚本会显示详细的处理信息和统计结果：

- 文件形状和数据类型
- 像素大小信息
- 原始和扩展后的mask体积
- 扩展比例
- 结果tomogram的统计信息

## 注意事项

1. 确保输入tomogram和mask文件形状匹配
2. 噪声水平是相对于tomogram标准差的相对值
3. 使用`--overwrite`参数可以覆盖已存在的输出文件
4. 脚本设置了随机种子以确保结果可重复
5. 所有计算都在内存中进行，大文件可能需要较多内存

## 示例输出

```
开始处理...
输入tomogram: input_tomo.mrc
输入mask: input_mask.mrc
输出文件: output.mrc
步骤1: 读取输入文件...
Tomogram形状: (100, 200, 200)
Tomogram数据类型: float32
Tomogram像素大小: (1.0, 1.0, 1.0)
Mask形状: (100, 200, 200)
Mask数据类型: float32
步骤2: 扩展mask...
使用高斯模糊扩展 (sigma=2.0)
步骤3: 平滑mask边界 (迭代次数: 1)...
步骤4: 提取masked tomogram并添加噪声背景...
噪声水平: 0.1 (相对于tomogram标准差)
背景值: 0.0
步骤5: 保存结果...

处理完成！
输出文件: output.mrc
输出形状: (100, 200, 200)
输出数据类型: float32
输出像素大小: (1.0, 1.0, 1.0)

统计信息:
原始mask体积: 50000 体素
扩展后mask体积: 75000 体素
扩展比例: 1.50x
结果tomogram均值: 0.1234
结果tomogram标准差: 0.5678
```
