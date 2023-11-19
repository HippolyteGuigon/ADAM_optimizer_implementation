from setuptools import setup, find_packages

setup(
    name="optimizer_implementation",
    version="0.1.0",
    packages=find_packages(
        include=["optimizer_implementation", "optimizer_implementation.*"]
    ),
    description="Python programm for creating a replica\
        of the principal Pytorch optimizerss",
    setup_requires=["wheel"],
    author="Hippolyte Guigon",
)
