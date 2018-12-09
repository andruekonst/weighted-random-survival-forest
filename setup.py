"""
Compile Cython files
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='Weighted RSF implementation',
      ext_modules=cythonize("fast_implementations.pyx"),
      zip_safe=False,
      include_dirs=[numpy.get_include()])