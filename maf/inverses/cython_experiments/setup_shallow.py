from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("shallow_op_cython.pyx", annotate=True),
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)
