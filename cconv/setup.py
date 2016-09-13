#!/usr/bin/env python

import os
import numpy as np
from setuptools import setup, find_packages, Extension


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

packages = ['cconv']
setup(
    name = 'myconv',
    version = '1.0',
    author = 'Igor Mandricnenko',
    description = "Basic functions to speed up numpy-based convolutional networks",
    packages = packages,
    install_requires = ['numpy', 'scipy'],
    ext_modules = [Extension('cconv',['cconv/cconv.c'])],
    include_dirs = [np.get_include()]
)

