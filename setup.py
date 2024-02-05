from setuptools import setup, find_packages


# TODO updated to src layout
setup(
    name="cnwigaps",
    version="1.0.0",
    description="A module for the CNWI-GAPS project",
    packages=find_packages(include=["cnwigaps.*", "cnwigaps"]),
)
