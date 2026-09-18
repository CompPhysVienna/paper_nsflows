from setuptools import setup, find_packages

setup(
    name="nsflows",
    version="0.1.0",
    description="Generative nested sampling of atomistic thermodynamic landscapes with normalizing flows",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.12",
        "torch>=2.2",
        "matplotlib>=3.8",
        "einops>=0.7",
        "tqdm>=4.66",
    ],
    entry_points={},
)
