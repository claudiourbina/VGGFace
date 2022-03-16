from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = []
with open('requirements.txt') as file:
    REQUIRED_PACKAGES = [line.strip() for line in file]

setup(
    name="resnet50",
    version="v1.0.0",
    description="ResNet50 for TFv2",
    url="https://github.com/claudiourbina/resnet50",
    author="Claudio Urbina",
    author_email="claudiourbina.vc@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    install_requires=[REQUIRED_PACKAGES],
    extras_require={
        "cpu": ["tensorflow==2.4.4"],
        "gpu": ["tensorflow-gpu==2.4.4"]
    }
)