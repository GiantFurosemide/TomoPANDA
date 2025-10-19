# TomoPANDA CLI架构设计

## 概述

TomoPANDA采用**多命令架构 (Multi-Command Architecture)**，提供统一的命令行接口：

```bash
tomopanda <command> <variable1> ...
```

## 架构特点

### 1. 模块化设计
- **主入口点**: `tomopanda/cli/main.py`
- **命令分发器**: `TomoPandaCLI` 类
- **命令基类**: `BaseCommand` 抽象基类
- **具体命令**: 继承自 `BaseCommand` 的各个命令类

### 2. 命令结构
```
tomopanda/
├── cli/
│   ├── main.py              # 主入口点
│   └── commands/
│       ├── base.py          # 命令基类
│       ├── detect.py        # 粒子检测命令
│       ├── train.py         # 模型训练命令
│       ├── visualize.py     # 可视化命令
│       ├── analyze.py       # 数据分析命令
│       ├── config.py        # 配置管理命令
│       ├── version.py       # 版本信息命令
│       └── sample.py        # 粒子采样命令
└── config/
    ├── manager.py           # 配置管理器
    └── defaults.py          # 默认配置
```

## 支持的命令

### 1. 粒子检测 (`detect`)
```bash
tomopanda detect <input> [options]
```
- **功能**: 检测断层扫描数据中的膜蛋白粒子
- **参数**: 输入文件、模型、阈值、输出格式等
- **输出**: 检测结果文件

### 2. 模型训练 (`train`)
```bash
tomopanda train <config> [options]
```
- **功能**: 训练SE(3)等变变换器模型
- **参数**: 配置文件、数据目录、训练参数等
- **输出**: 训练好的模型

### 3. 数据可视化 (`visualize`)
```bash
tomopanda visualize <input> [options]
```
- **功能**: 可视化断层扫描数据和检测结果
- **参数**: 输入文件、可视化类型、渲染参数等
- **输出**: 可视化图像

### 4. 数据分析 (`analyze`)
```bash
tomopanda analyze <input> [options]
```
- **功能**: 分析断层扫描数据和检测结果
- **参数**: 输入文件、分析类型、统计参数等
- **输出**: 分析报告

### 5. 配置管理 (`config`)
```bash
tomopanda config <action> [options]
```
- **功能**: 管理TomoPANDA配置
- **子命令**: `init`, `show`, `validate`
- **输出**: 配置文件

### 6. 粒子采样 (`sample`)
```bash
tomopanda sample <method> [options]
```
- **功能**: 从断层扫描数据中采样粒子候选
- **子命令**: `mesh-geodesic` - 基于网格测地距离的膜蛋白采样
- **参数**: 膜掩码、采样参数、输出格式等
- **输出**: 采样坐标、RELION格式文件

#### 6.1 Mesh Geodesic采样
```bash
# 使用合成数据测试
tomopanda sample mesh-geodesic --create-synthetic --output results/

# 使用真实膜掩码
tomopanda sample mesh-geodesic --mask membrane_mask.mrc --output results/

# 自定义参数 (使用expected particle size - taubin iterations会自动计算)
tomopanda sample mesh-geodesic \
    --mask membrane_mask.mrc \
    --output results/ \
    --expected-particle-size 25.0 \
    --smoothing-sigma 2.0 \
    --verbose

# 替代方案: 使用手动taubin iterations (不使用expected particle size)
tomopanda sample mesh-geodesic \
    --mask membrane_mask.mrc \
    --output results/ \
    --smoothing-sigma 2.0 \
    --taubin-iterations 15 \
    --verbose
```

### 7. 版本信息 (`version`)
```bash
tomopanda version [options]
```
- **功能**: 显示版本和系统信息
- **参数**: `--short`, `--verbose`
- **输出**: 版本信息

## 架构优势

### 1. 可扩展性
- 新命令只需继承 `BaseCommand` 并实现必要方法
- 命令间相互独立，易于维护
- 支持命令别名和子命令

### 2. 一致性
- 所有命令共享通用参数 (`--verbose`, `--config`, `--output`)
- 统一的错误处理和日志记录
- 标准化的帮助信息格式

### 3. 易用性
- 清晰的命令结构
- 详细的帮助信息
- 智能的参数验证

### 4. 配置管理
- 统一的配置系统
- 支持多种配置格式 (JSON, YAML)
- 配置验证和模板生成

## 使用示例

### 基本使用
```bash
# 显示帮助
tomopanda --help

# 显示版本
tomopanda version

# 初始化配置
tomopanda config init --template detect

# 检测粒子
tomopanda detect input.mrc --model default --threshold 0.5

# 训练模型
tomopanda train config.json --data-dir ./data --epochs 100

# 可视化结果
tomopanda visualize output.json --type particles --format png

# 分析数据
tomopanda analyze results.json --analysis density distribution
```

### 高级使用
```bash
# 使用配置文件
tomopanda detect input.mrc --config my_config.json

# 批量处理
tomopanda detect *.mrc --batch-size 4 --device cuda

# 自定义输出
tomopanda detect input.mrc --output ./results --format pdb

# 详细输出
tomopanda detect input.mrc --verbose
```

## 开发指南

### 添加新命令
1. 在 `commands/` 目录下创建新文件
2. 继承 `BaseCommand` 类
3. 实现必要的方法：
   - `get_name()`: 命令名称
   - `get_description()`: 命令描述
   - `add_parser()`: 添加参数解析器
   - `execute()`: 执行命令逻辑
4. 在 `main.py` 中注册新命令

### 示例：添加新命令
```python
# commands/new_command.py
from .base import BaseCommand

class NewCommand(BaseCommand):
    def get_name(self):
        return "new"
    
    def get_description(self):
        return "新命令描述"
    
    def add_parser(self, subparsers):
        parser = subparsers.add_parser(self.name, help=self.description)
        # 添加参数
        return parser
    
    def execute(self, args):
        # 实现命令逻辑
        return 0
```

## 配置系统

### 配置文件格式
支持 JSON 和 YAML 格式：

```json
{
  "detection": {
    "model": "default",
    "threshold": 0.5,
    "min_size": 10,
    "max_size": 100
  },
  "training": {
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 1e-4
  }
}
```

### 配置管理
```bash
# 初始化配置
tomopanda config init --template detect

# 显示配置
tomopanda config show

# 验证配置
tomopanda config validate config.json
```

## 错误处理

### 返回码
- `0`: 成功
- `1`: 一般错误
- `2`: 参数错误
- `3`: 文件错误
- `4`: 配置错误

### 错误信息
- 清晰的错误消息
- 可选的详细错误信息 (`--verbose`)
- 建议的解决方案

## 性能优化

### 并行处理
- 支持批处理
- 多进程/多线程支持
- GPU加速选项

### 内存管理
- 流式处理大文件
- 内存使用监控
- 缓存机制

## 测试

### 单元测试
```bash
# 运行测试
python -m pytest tests/

# 测试覆盖率
python -m pytest --cov=tomopanda tests/
```

### 集成测试
```bash
# 测试命令行接口
python -m tomopanda.cli.main version
python -m tomopanda.cli.main detect --help
```

## 部署

### 安装
```bash
# 开发安装
pip install -e .

# 生产安装
pip install tomopanda
```

### 打包
```bash
# 构建包
python setup.py sdist bdist_wheel

# 上传到PyPI
twine upload dist/*
```

## 总结

TomoPANDA的多命令架构提供了：

1. **清晰的命令结构**: 每个功能对应一个命令
2. **统一的接口**: 所有命令共享相同的使用模式
3. **灵活的配置**: 支持多种配置方式
4. **易于扩展**: 新功能可以轻松添加为新命令
5. **用户友好**: 详细的帮助信息和错误提示

这种架构特别适合科学计算工具，既保持了功能的独立性，又提供了统一的用户体验。
