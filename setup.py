from setuptools import find_packages, setup

setup(
    name="auto_eval",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "absl-py",
        "ml_collections",
        "tensorflow",
        "wandb",
        "einops",
    ],
)
