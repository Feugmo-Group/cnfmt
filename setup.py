"""
Setup script for CNFMT package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cnfmt",
    version="1.0.0",
    author="Cony",
    description="Conditional Neural Fundamental Measure Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/cnfmt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "equinox>=0.11.0",
        "optax>=0.1.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cnfmt-train=cnfmt.scripts.train:main",
        ],
    },
)
