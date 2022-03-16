from setuptools import find_packages
from setuptools import setup

setup(
    name="vggface",
    version="v1.0.0",
    description="VGGFace for TFv2",
    url="https://github.com/claudiourbina/VGGFace",
    author="Claudio Urbina",
    author_email="claudiourbina.vc@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    install_requires=["pillow==9.0.1"],
    extras_require={
        "cpu": ["tensorflow==2.4.4"],
        "gpu": ["tensorflow-gpu==2.4.4"]
    }
)