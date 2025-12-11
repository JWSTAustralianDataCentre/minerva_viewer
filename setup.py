"""
MINERVA Galaxy Viewer - Setup Configuration

Installation:
    pip install -e .        # Development mode (editable)
    pip install .           # Standard installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="minerva-viewer",
    version="0.1.0",
    author="MINERVA Team",
    author_email="your.email@institution.edu",
    description="Interactive viewer for JWST MINERVA survey galaxies with photo-z and spectroscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/minerva_viewer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "astropy>=5.0.0",
        "eazy-py>=0.6.0",
        "grizli>=1.9.0",
        "panel>=1.0.0",
        "param>=1.12.0",
        "pillow>=9.0.0",
        "requests>=2.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "ipython>=8.0",
            "jupyter>=1.0",
            "notebook>=6.4",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    package_data={
        "": ["templates/*", "templates/**/*"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "minerva-viewer=minerva_viewer:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/minerva_viewer/issues",
        "Source": "https://github.com/yourusername/minerva_viewer",
        "Documentation": "https://github.com/yourusername/minerva_viewer/blob/main/README.md",
    },
    keywords="astronomy jwst galaxies photometric-redshift spectroscopy visualization",
    zip_safe=False,
)
