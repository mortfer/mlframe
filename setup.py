from setuptools import setup, find_packages

setup(
    name="mlframe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "keras",
        "matplotlib",
    ],
    description="A machine learning framework for training and evaluating models",
    long_description=open("README.md").read(),
    url="https://github.com/mortfer/mlframe",
    python_requires=">=3.8",
)
