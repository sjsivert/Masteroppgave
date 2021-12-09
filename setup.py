#!/usr/bin/env python

from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name="Masterproject",
    version="1.0",
    description="Sindre and Sanders Master project",
    author="Sindre J. I. Sivertsen, Sander Kilen",
    author_email="sindrejohan1@gmail.com",
    url="https://github.com/NikZy/Masteroppgave/",
    packages=find_packages(include=["src"]),
)
