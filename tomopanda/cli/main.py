#!/usr/bin/env python3
"""
TomoPANDA main command line entry point
Supports multi-command architecture: tomopanda <command> <args...>
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
    """TomoPANDA main command line interface class"""
    
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
        """Create main argument parser"""
        parser = argparse.ArgumentParser(
            prog='tomopanda',
            description='TomoPANDA - CryoET membrane protein detection tool',
            epilog='Use "tomopanda <command> --help" to view help for specific commands',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Add subcommands
        subparsers = parser.add_subparsers(
            dest='command', # Tell main parser: subcommands are stored in 'command' attribute
            help='commands',
            metavar='<command>'
        )
        
        # Create subparser for each command
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
