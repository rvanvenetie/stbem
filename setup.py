# This file optionally cythonizes the source code, to gain some additional
# speedup. Do not forget to recompile if you change the source.
#
# Run as `python3 setup.py build_ext --inplace`
import os

from Cython.Build import cythonize
from setuptools import setup

os.environ['CFLAGS'] = '-O3'
setup(ext_modules=cythonize(
    [
        'src/quadrature_rules.py', 'src/quadrature.py',
        'src/parametrization.py', 'src/single_layer_exact.py',
        'src/initial_potential.py', 'src/error_estimator.py',
        'src/hierarchical_error_estimator.py', 'src/mesh.py',
        'src/initial_mesh.py', 'src/parametrization.py', 'src/norms.py',
        'src/single_layer.py'
    ],
    compiler_directives={'language_level': "3"},
))
