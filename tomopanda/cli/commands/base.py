"""
Base command class
Base class for all commands
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCommand(ABC):
    """Base command class"""
    
    def __init__(self):
        self.name = self.get_name()
        self.description = self.get_description()
    
    @abstractmethod
    def get_name(self) -> str:
        """Get command name"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get command description"""
        pass
    
    @abstractmethod
    def add_parser(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """Add command argument parser"""
        pass
    
    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute command"""
        pass
    
    def add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """Add common arguments"""
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show verbose output'
        )
        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Configuration file path'
        )
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output directory or file path'
        )
