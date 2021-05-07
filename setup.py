# Run as `python3 setup.py build_ext --inplace`
import os
from glob import glob
from setuptools import setup
from Cython.Build import cythonize

os.environ['CFLAGS'] = '-O3'
setup(ext_modules=cythonize(
    [
        'quadrature_rules.py', 'quadrature.py', 'parametrization.py',
        'single_layer_exact.py', 'initial_potential.py', 'error_estimator.py',
        'hierarchical_error_estimator.py', 'mesh.py', 'initial_mesh.py',
        'parametrization.py', 'norms.py', 'single_layer.py'
    ],
    compiler_directives={'language_level': "3"},
))
