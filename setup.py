#!/usr/bin/env python3
"""
TomoPANDA安装脚本
"""

from setuptools import setup, find_packages
import pathlib
import re
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 读取requirements文件
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

def read_version():
    init_path = pathlib.Path("tomopanda/__init__.py")
    text = init_path.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise RuntimeError("Version not found in tomopanda/__init__.py")
    return m.group(1)


setup(
    name="tomopanda",
    version=read_version(),
    author="TomoPANDA Team",
    author_email="contact@tomopanda.org",
    description="基于SE(3)等变变换器的CryoET膜蛋白检测工具",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tomopanda/tomopanda",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "tomopanda=tomopanda.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
