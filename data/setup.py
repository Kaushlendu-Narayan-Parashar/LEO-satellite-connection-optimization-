#!/usr/bin/env python3
"""
Setup script for LEO Satellite Communication Simulation Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="leo-satellite-simulation",
    version="1.0.0",
    author="LEO Simulation Framework",
    author_email="your.email@example.com",
    description="Comprehensive LEO satellite communication link simulation framework for AI research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/leo-satellite-simulation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "leo-simulate=leo_simulation.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "leo_simulation": [
            "data/*.yaml",
            "data/*.json",
        ],
    },
    keywords=[
        "satellite",
        "communication",
        "LEO",
        "simulation",
        "AI",
        "machine learning",
        "adaptive coding",
        "modulation",
        "DVB-S2",
        "orbital mechanics",
        "atmospheric propagation",
        "link budget"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/leo-satellite-simulation/issues",
        "Source": "https://github.com/yourusername/leo-satellite-simulation",
        "Documentation": "https://leo-satellite-simulation.readthedocs.io/",
    },
) 