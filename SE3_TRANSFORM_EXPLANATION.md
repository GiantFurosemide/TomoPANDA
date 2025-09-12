# SE(3)变换详解

## 1. 什么是SE(3)？

**SE(3)** 是**特殊欧几里得群 (Special Euclidean Group)** 在3维空间中的表示，记作 SE(3)。

### 数学定义
SE(3) = { (R, t) | R ∈ SO(3), t ∈ ℝ³ }

其中：
- **SO(3)**: 3维特殊正交群（3D旋转群）
- **R**: 3×3旋转矩阵
- **t**: 3维平移向量

## 2. SE(3)的组成

### 2.1 旋转部分 SO(3)
SO(3)包含所有3×3的旋转矩阵，满足：
- **正交性**: R^T R = I
- **行列式为1**: det(R) = 1
- **保持长度**: ||Rx|| = ||x||

### 2.2 平移部分 ℝ³
ℝ³是3维实向量空间，表示空间中的平移。

## 3. SE(3)变换的表示

### 3.1 齐次坐标表示
SE(3)变换可以用4×4齐次变换矩阵表示：

```
T = [R  t]  = [r₁₁  r₁₂  r₁₃  tₓ]
    [0  1]    [r₂₁  r₂₂  r₂₃  tᵧ]
              [r₃₁  r₃₂  r₃₃  tᵧ]
              [0    0    0    1 ]
```

### 3.2 对点的变换
对于3D点 p = (x, y, z)，SE(3)变换为：
```
p' = Tp = [R  t] [x]  = [Rx + t]
          [0  1] [y]    [  1   ]
                  [z]
                  [1]
```

## 4. SE(3)等变性 (SE(3) Equivariance)

### 4.1 等变性定义
函数 f 是SE(3)等变的，如果对于任意SE(3)变换 g：
```
f(g·x) = g·f(x)
```

### 4.2 在CryoET中的意义
- **旋转等变性**: 蛋白质旋转后，检测结果也相应旋转
- **平移等变性**: 蛋白质平移后，检测结果也相应平移
- **物理合理性**: 符合物理世界的对称性

## 5. SE(3)变换的参数化

### 5.1 欧拉角表示
```
R = Rz(γ) Ry(β) Rx(α)
```
其中 α, β, γ 是绕x, y, z轴的旋转角度。

### 5.2 四元数表示
四元数 q = w + xi + yj + zk，其中 w² + x² + y² + z² = 1

### 5.3 轴角表示
旋转轴 n (单位向量) + 旋转角度 θ

### 5.4 李代数表示
se(3) = { (ω, v) | ω ∈ so(3), v ∈ ℝ³ }

## 6. SE(3)在TomoPANDA中的应用

### 6.1 膜蛋白检测
```python
# 伪代码示例
class SE3Transformer:
    def __init__(self):
        self.rotation_group = SO3()
        self.translation_space = R3()
    
    def transform_protein(self, protein_coords, se3_params):
        R, t = se3_params
        # 应用SE(3)变换
        transformed_coords = R @ protein_coords + t
        return transformed_coords
```

### 6.2 等变特征提取
- **球谐函数**: 在球面上定义的函数，具有旋转等变性
- **不可约表示**: 将SE(3)群作用分解为不可约子空间
- **群卷积**: 在SE(3)群上的卷积操作

## 7. SE(3)变换的计算

### 7.1 组合变换
两个SE(3)变换的组合：
```
T₃ = T₂ T₁ = [R₂  t₂] [R₁  t₁] = [R₂R₁  R₂t₁ + t₂]
             [0   1 ] [0   1 ]   [0     1        ]
```

### 7.2 逆变换
SE(3)变换的逆：
```
T⁻¹ = [R^T  -R^T t]
      [0    1     ]
```

### 7.3 李群指数映射
从李代数se(3)到李群SE(3)的映射：
```
exp: se(3) → SE(3)
```

## 8. 数值实现考虑

### 8.1 旋转矩阵的正交化
由于数值误差，需要定期正交化旋转矩阵：
```python
def orthogonalize_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt
```

### 8.2 四元数归一化
```python
def normalize_quaternion(q):
    return q / np.linalg.norm(q)
```

## 9. SE(3)等变神经网络架构

### 9.1 核心组件
- **SE(3)卷积层**: 在SE(3)群上的卷积
- **等变激活函数**: 保持等变性的激活函数
- **球谐函数基**: 用于表示旋转等变函数

### 9.2 网络结构
```
输入 → SE(3)卷积 → 等变激活 → SE(3)卷积 → ... → 输出
```

## 10. 在CryoET中的优势

### 10.1 数据增强
- 自动生成旋转和平移的数据增强
- 保持物理意义的变换

### 10.2 泛化能力
- 对任意方向的蛋白质都能正确检测
- 减少对训练数据方向分布的依赖

### 10.3 计算效率
- 利用群结构减少参数数量
- 共享权重提高效率

## 11. 实际应用示例

### 11.1 膜蛋白姿态估计
```python
# 估计膜蛋白的6DOF姿态 (3个旋转 + 3个平移)
def estimate_pose(protein_density):
    # 使用SE(3)等变网络
    se3_params = se3_network(protein_density)
    rotation, translation = decompose_se3(se3_params)
    return rotation, translation
```

### 11.2 粒子检测
```python
# 检测膜蛋白粒子位置和方向
def detect_particles(tomogram):
    # SE(3)等变特征提取
    features = se3_feature_extractor(tomogram)
    # 等变检测
    particles = se3_detector(features)
    return particles
```

## 12. 总结

SE(3)变换是TomoPANDA的核心数学基础，它：

1. **数学严谨**: 基于李群理论，数学基础扎实
2. **物理合理**: 符合3D空间的几何变换规律
3. **计算高效**: 利用群结构优化计算
4. **应用广泛**: 适用于各种3D几何问题

在CryoET膜蛋白检测中，SE(3)等变性确保了模型能够：
- 正确处理任意方向的蛋白质
- 保持检测结果的几何一致性
- 提高模型的泛化能力和鲁棒性
