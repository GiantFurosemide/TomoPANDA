"""
Configuration management command
tomopanda config <action> [options]
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from .base import BaseCommand


class ConfigCommand(BaseCommand):
    """Configuration management command"""
    
    def get_name(self) -> str:
        return "config"
    
    def get_description(self) -> str:
        return "Manage TomoPANDA configuration"
    
    def add_parser(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help=self.description,
            description="Manage TomoPANDA configuration files"
        )
        
        # Subcommands
        subparsers_config = parser.add_subparsers(
            dest='action',
            help='Configuration operations',
            metavar='<action>'
        )
        
        # Initialize configuration
        init_parser = subparsers_config.add_parser(
            'init',
            help='Initialize configuration file'
        )
        init_parser.add_argument(
            '--template',
            choices=['detect', 'train', 'analyze'],
            default='detect',
            help='Configuration template (default: detect)'
        )
        init_parser.add_argument(
            '--force',
            action='store_true',
            help='Overwrite existing configuration file'
        )
        
        # Show configuration
        show_parser = subparsers_config.add_parser(
            'show',
            help='Show current configuration'
        )
        show_parser.add_argument(
            '--section',
            type=str,
            help='Show specific configuration section'
        )
        
        # Validate configuration
        validate_parser = subparsers_config.add_parser(
            'validate',
            help='Validate configuration file'
        )
        validate_parser.add_argument(
            'config_file',
            type=str,
            help='Configuration file to validate'
        )
        
        # Add common arguments
        self.add_common_args(parser)
        
        return parser
    
    def execute(self, args: argparse.Namespace) -> int:
        """执行配置命令"""
        try:
            if not args.action:
                print("错误: 请指定配置操作 (init, show, validate)")
                return 1
            
            if args.action == 'init':
                return self._init_config(args)
            elif args.action == 'show':
                return self._show_config(args)
            elif args.action == 'validate':
                return self._validate_config(args)
            else:
                print(f"错误: 未知的配置操作: {args.action}")
                return 1
                
        except Exception as e:
            print(f"配置操作过程中发生错误: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _init_config(self, args: argparse.Namespace) -> int:
        """初始化配置文件"""
        config_file = Path('tomopanda_config.json')
        
        if config_file.exists() and not args.force:
            print(f"配置文件已存在: {config_file}")
            print("使用 --force 覆盖现有配置")
            return 1
        
        # 生成配置模板
        config = self._generate_config_template(args.template)
        
        # 保存配置文件
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"配置文件已创建: {config_file}")
        return 0
    
    def _show_config(self, args: argparse.Namespace) -> int:
        """显示配置"""
        config_file = Path('tomopanda_config.json')
        
        if not config_file.exists():
            print("配置文件不存在，请先运行 'tomopanda config init'")
            return 1
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if args.section:
            if args.section in config:
                print(f"{args.section} 配置:")
                print(json.dumps(config[args.section], indent=2, ensure_ascii=False))
            else:
                print(f"配置节 '{args.section}' 不存在")
                return 1
        else:
            print("当前配置:")
            print(json.dumps(config, indent=2, ensure_ascii=False))
        
        return 0
    
    def _validate_config(self, args: argparse.Namespace) -> int:
        """验证配置文件"""
        config_file = Path(args.config_file)
        
        if not config_file.exists():
            print(f"配置文件不存在: {args.config_file}")
            return 1
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # TODO: 实现配置验证逻辑
            print(f"配置文件验证通过: {args.config_file}")
            return 0
            
        except json.JSONDecodeError as e:
            print(f"配置文件格式错误: {e}")
            return 1
    
    def _generate_config_template(self, template: str) -> dict:
        """生成配置模板"""
        templates = {
            'detect': {
                "detection": {
                    "model": "default",
                    "threshold": 0.5,
                    "min_size": 10,
                    "max_size": 100,
                    "batch_size": 1,
                    "device": "auto"
                },
                "output": {
                    "format": "json",
                    "save_visualization": False
                }
            },
            'train': {
                "training": {
                    "epochs": 100,
                    "batch_size": 8,
                    "learning_rate": 1e-4,
                    "validation_split": 0.2
                },
                "model": {
                    "type": "se3_transformer",
                    "hidden_dim": 128,
                    "num_layers": 6
                },
                "data": {
                    "data_dir": "",
                    "augmentation": True
                }
            },
            'analyze': {
                "analysis": {
                    "types": ["density", "distribution"],
                    "bins": 50,
                    "percentiles": [25, 50, 75]
                },
                "filtering": {
                    "min_confidence": 0.0,
                    "min_size": None,
                    "max_size": None
                },
                "output": {
                    "format": "json",
                    "include_plots": False
                }
            }
        }
        
        return templates.get(template, templates['detect'])
