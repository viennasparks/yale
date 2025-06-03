from setuptools import setup, find_packages

setup(
    name="aptamer-discovery-platform",
    version="1.0.0",
    description="Advanced Aptamer Analytics and Discovery Platform",
    author="Aptamer Discovery Team",
    author_email="info@aptamer-discovery.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scipy>=1.10.1",
        "scikit-learn>=1.3.0",
        "torch>=2.0.1",
        "xgboost>=1.7.5",
        "biopython>=1.81",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "plotly>=5.15.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0.1",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "aptamer-discovery=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
