from setuptools import setup, find_packages

setup(
    name="snana_processing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    author="Pablo Cornejo",
    description="Processing tools for SNANA (SuperNova ANAlysis) data",
    python_requires=">=3.6",
) 