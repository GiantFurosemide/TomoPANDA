"""
基础命令类
所有命令的基类
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCommand(ABC):
    """命令基类"""
    
    def __init__(self):
        self.name = self.get_name()
        self.description = self.get_description()
    
    @abstractmethod
    def get_name(self) -> str:
        """获取命令名称"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """获取命令描述"""
        pass
    
    @abstractmethod
    def add_parser(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """添加命令参数解析器"""
        pass
    
    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """执行命令"""
        pass
    
    def add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """添加通用参数"""
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='显示详细输出'
        )
        parser.add_argument(
            '--config', '-c',
            type=str,
            help='配置文件路径'
        )
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='输出目录或文件路径'
        )
