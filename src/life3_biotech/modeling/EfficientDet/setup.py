from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("src/life3_biotech/modeling/EfficientDet/utils/compute_overlap.pyx"),
    include_dirs=[numpy.get_include()]
)
