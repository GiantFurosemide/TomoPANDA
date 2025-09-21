# TomoPANDA Module List

## 核心架构模块 (Core Architecture Modules)

### 1. 数据输入与预处理模块 (Data Input & Preprocessing)
- **tomogram_loader**: CryoET断层扫描数据加载器
- **preprocessing**: 数据预处理（去噪、归一化、增强）
- **coordinate_system**: 坐标系转换和SE(3)变换处理
- **data_augmentation**: SE(3)等变数据增强

### 2. SE(3)等变变换器核心模块 (SE(3)-Equivariant Transformer Core)
- **se3_transformer**: SE(3)等变变换器主架构
- **irreducible_representations**: 不可约表示计算
- **spherical_harmonics**: 球谐函数计算
- **group_convolution**: 群卷积操作
- **equivariant_attention**: 等变注意力机制

### 3. 特征提取与编码模块 (Feature Extraction & Encoding)
- **membrane_detector**: 膜结构检测器
- **protein_encoder**: 蛋白质特征编码器
- **spatial_encoder**: 空间特征编码器
- **multi_scale_features**: 多尺度特征提取

### 4. 粒子检测与定位模块 (Particle Detection & Localization)
- **particle_detector**: 粒子检测器
- **pose_estimator**: 姿态估计器
- **confidence_scorer**: 置信度评分器
- **non_maximum_suppression**: 非极大值抑制
- **mesh_geodesic**: 基于网格测地距离的膜蛋白采样算法

### 5. 后处理与优化模块 (Post-processing & Optimization)
- **refinement**: 粒子位置精化
- **filtering**: 结果过滤和验证
- **clustering**: 粒子聚类分析
- **quality_assessment**: 质量评估

## 支持模块 (Supporting Modules)

### 6. 数学与几何计算模块 (Mathematical & Geometric Computing)
- **quaternion_ops**: 四元数操作
- **rotation_matrices**: 旋转矩阵计算
- **lie_algebra**: 李代数操作
- **geometric_utils**: 几何工具函数

### 7. 神经网络基础模块 (Neural Network Foundation)
- **layers**: 自定义层实现
- **activations**: 激活函数
- **normalization**: 归一化层
- **loss_functions**: 损失函数

### 8. 训练与优化模块 (Training & Optimization)
- **trainer**: 训练管理器
- **optimizer**: 优化器配置
- **scheduler**: 学习率调度器
- **checkpointing**: 模型检查点管理

### 9. 评估与验证模块 (Evaluation & Validation)
- **metrics**: 评估指标计算
- **validation**: 模型验证
- **cross_validation**: 交叉验证
- **performance_analysis**: 性能分析

### 10. 可视化模块 (Visualization)
- **tomogram_visualizer**: 断层扫描可视化
- **particle_visualizer**: 粒子检测结果可视化
- **trajectory_plotter**: 轨迹绘制
- **statistics_plotter**: 统计图表

### 11. 输入输出模块 (I/O Modules)
- **file_handlers**: 文件格式处理
- **data_exporters**: 数据导出器
- **config_parser**: 配置文件解析
- **log_manager**: 日志管理
- **mrc_utils**: MRC文件读写工具
- **relion_utils**: RELION格式转换工具

### 12. 工具与实用程序模块 (Utilities & Tools)
- **memory_manager**: 内存管理
- **parallel_processing**: 并行处理
- **caching**: 缓存系统
- **profiling**: 性能分析工具

## 配置与接口模块 (Configuration & Interface Modules)

### 13. 配置管理模块 (Configuration Management)
- **config_manager**: 配置管理器
- **parameter_validation**: 参数验证
- **default_configs**: 默认配置
- **environment_setup**: 环境设置

### 14. API接口模块 (API Interface)
- **python_api**: Python API接口
- **command_line**: 命令行接口
- **batch_processing**: 批处理接口
- **web_interface**: Web界面（可选）

### 15. 测试模块 (Testing)
- **unit_tests**: 单元测试
- **integration_tests**: 集成测试
- **performance_tests**: 性能测试
- **test_data**: 测试数据

## 依赖关系 (Dependencies)

### 外部依赖 (External Dependencies)
- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **SciPy**: 科学计算
- **OpenCV**: 计算机视觉
- **MDAnalysis**: 分子动力学分析
- **e3nn**: SE(3)等变神经网络
- **torch-geometric**: 几何深度学习

### 内部依赖关系 (Internal Dependencies)
```
核心模块 → 支持模块 → 工具模块
    ↓         ↓         ↓
配置模块 ← API接口 ← 测试模块
```

## 模块优先级 (Module Priority)

### 高优先级 (High Priority)
1. SE(3)等变变换器核心模块
2. 数据输入与预处理模块
3. 粒子检测与定位模块
4. 数学与几何计算模块

### 中优先级 (Medium Priority)
5. 特征提取与编码模块
6. 后处理与优化模块
7. 训练与优化模块
8. 评估与验证模块

### 低优先级 (Low Priority)
9. 可视化模块
10. 输入输出模块
11. 工具与实用程序模块
12. 配置与接口模块
13. 测试模块

## 开发建议 (Development Recommendations)

1. **分阶段开发**: 按优先级顺序开发模块
2. **模块化设计**: 确保模块间低耦合高内聚
3. **接口标准化**: 定义清晰的模块接口
4. **测试驱动**: 每个模块都应有对应的测试
5. **文档完善**: 为每个模块提供详细文档
6. **性能优化**: 特别关注SE(3)计算的效率
7. **内存管理**: 考虑大规模CryoET数据的处理需求
