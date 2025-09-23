#!/usr/bin/env python3
"""
TomoPANDA主命令行入口点
支持多命令架构: tomopanda <command> <args...>
"""

import sys
import argparse
from typing import List, Optional

from .commands.detect import DetectCommand
from .commands.train import TrainCommand
from .commands.visualize import VisualizeCommand
from .commands.analyze import AnalyzeCommand
from .commands.config import ConfigCommand
from .commands.version import VersionCommand
from .commands.sample import SampleCommand


class TomoPandaCLI:
    """TomoPANDA命令行接口主类"""
    
    def __init__(self):
        self.commands = {
            'detect': DetectCommand(),
            'train': TrainCommand(),
            'visualize': VisualizeCommand(),
            'analyze': AnalyzeCommand(),
            'config': ConfigCommand(),
            'version': VersionCommand(),
            'sample': SampleCommand(),
        }
    
    def create_parser(self) -> argparse.ArgumentParser:
        """创建主参数解析器"""
        parser = argparse.ArgumentParser(
            prog='tomopanda',
            description='TomoPANDA - CryoET膜蛋白检测工具',
            epilog='使用 "tomopanda <command> --help" 查看特定命令的帮助信息',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # 添加子命令
        subparsers = parser.add_subparsers(
            dest='command', # 告诉主解析器：子命令存储在 'command' 属性中
            help='commands',
            metavar='<command>'
        )
        
        # 为每个命令创建子解析器
        for name, command in self.commands.items():
            command.add_parser(subparsers)
        
        return parser
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """运行命令行接口"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # 如果没有提供命令，显示帮助
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        # 执行对应的命令
        command = self.commands[parsed_args.command]
        return command.execute(parsed_args)


def main(args: Optional[List[str]] = None) -> int:
    """主入口函数"""
    cli = TomoPandaCLI()
    return cli.run(args)


if __name__ == '__main__':
    sys.exit(main())
