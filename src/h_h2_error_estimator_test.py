import random

import numpy as np

from .h_h2_error_estimator import HH2ErrorEstimator
from .mesh import MeshParametrized
from .parametrization import UnitSquare
from .single_layer import SingleLayerOperator


def test_error_estimator():
    mesh = MeshParametrized(UnitSquare())
    random.seed(5)
    for _ in range(10):
        elem = random.choice(list(mesh.leaf_elements))
        mesh.refine_axis(elem, random.random() < 0.5)

    SL = SingleLayerOperator(mesh)

    def RHS(elems):
        rhs = np.zeros(shape=len(elems))
        for i, elem_test in enumerate(elems):
            rhs[i] = elem_test.h_t * elem_test.h_x

        return rhs

    h_h2_error_estimator = HH2ErrorEstimator(SL=SL, g=RHS, use_mp=False)

    elems = list(mesh.leaf_elements)
    mat = SL.bilform_matrix(elems, elems, use_mp=False)
    rhs = RHS(elems)
    Phi = np.linalg.solve(mat, rhs)
    print(h_h2_error_estimator.estimate(elems, Phi))
