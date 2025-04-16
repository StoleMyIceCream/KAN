from setuptools import setup, find_packages

setup(
    name="kan-implementation",
    version="0.1.0",
    description="PyTorch implementation of Kolmogorov-Arnold Networks (KAN)",
    author="Luca Nogueira Calçado",
    author_email="luca.n.calcado@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)