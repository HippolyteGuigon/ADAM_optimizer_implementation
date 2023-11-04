from setuptools import setup, find_packages

setup(
    name="adam_optimizer_implementation",
    version="0.1.0",
    packages=find_packages(
        include=["adam_optimizer_implementation", "adam_optimizer_implementation.*"]
    ),
    description="Python programm for creating a replica\
        of the principal Pytorch optimizerss",
    author="Hippolyte Guigon",
)
