from setuptools import setup, find_packages
import os
import subprocess


setup(
    name="vent",
    author="vent team",
    author_email="vent@vents.com",
    description="some description of how we made a ventilator",
    keywords="vents ventilators etc",
    url="https://ventilator.readthedocs.io",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "scipy",
        "sklearn",
        "matplotlib",
        "jupyter",
        "ipython",
        "pathos",
        "pigpio"
    ],
    python_requires="==3.7.*"
)
