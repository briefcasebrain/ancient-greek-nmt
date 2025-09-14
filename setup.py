"""
Setup script for Ancient Greek Neural Machine Translation Library

This setup file configures the package for installation and distribution.
It includes all necessary dependencies, package metadata, and entry points.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')
else:
    long_description = """
    Ancient Greek Neural Machine Translation Library

    A comprehensive library for training and deploying neural machine translation
    models between Ancient Greek and modern languages.
    """

# Core dependencies required for the library to function
REQUIRED_PACKAGES = [
    "transformers>=4.42.0",
    "datasets>=2.16.0",
    "accelerate>=0.33.0",
    "torch>=2.0.0",
    "sacrebleu>=2.3.0",
    "ftfy>=6.1.0",
    "regex>=2023.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
]

# Optional dependencies for additional features
EXTRAS_REQUIRE = {
    "visualization": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
    ],
    "notebook": [
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
        "notebook>=7.0.0",
    ],
    "development": [
        "pytest>=7.3.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.3.0",
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
    ],
    "all": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
        "notebook>=7.0.0",
    ]
}

setup(
    # Package metadata
    name="ancient-greek-nmt",
    version="1.0.0",
    author="Aansh Shah",
    author_email="aansh@briefcasebrain.com",
    description="Neural Machine Translation for Ancient Greek",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/briefcasebrain/ancient-greek-nmt",
    project_urls={
        "Source": "https://github.com/briefcasebrain/ancient-greek-nmt",
        "Tracker": "https://github.com/briefcasebrain/ancient-greek-nmt/issues",
    },

    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    package_data={
        "ancient_greek_nmt": [
            "configs/*.json",
            "configs/*.yaml",
            "data/*.txt",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",

    # Dependencies
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,

    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "ancient-greek-translate=ancient_greek_nmt.cli:main",
            "ancient-greek-train=ancient_greek_nmt.cli:train",
            "ancient-greek-evaluate=ancient_greek_nmt.cli:evaluate",
        ],
    },

    # Classification metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Natural Language :: Greek",
    ],

    # Keywords for searchability
    keywords=[
        "ancient greek",
        "neural machine translation",
        "nmt",
        "transformers",
        "deep learning",
        "natural language processing",
        "nlp",
        "classics",
        "digital humanities",
        "translation",
        "mbart",
        "nllb",
    ],

    # License
    license="MIT",

    # Testing configuration
    test_suite="tests",
    tests_require=[
        "pytest>=7.3.0",
        "pytest-cov>=4.0.0",
    ],

    # Zip safety
    zip_safe=False,
)