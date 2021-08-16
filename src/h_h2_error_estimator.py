import time

import numpy as np

from .hierarchical_error_estimator import DummyElement


class HH2ErrorEstimator:
    def __init__(self, SL, M0=None, g=None, use_mp=True):
        self.SL = SL
        self.M0 = M0
        self.g = g
        self.use_mp = use_mp

    def estimate(self, elems, Phi, problem=None):
        """ Returns the hierarchical basis estimator for given function Phi. """

        # Calcualte uniform refinement of the mesh.
        elems_coarse = elems
        elem_2_children = DummyElement.uniform_refinement(elems_coarse)

        # Flatten list and calculate mapping of indices.
        elems_fine = [
            child for children in elem_2_children for child in children
        ]
        #elem_2_idx_fine = {k: v for v, k in enumerate(elems_fine)}

        # Evaluate SL matrix on the fine mesh.
        mat_fine = self.SL.bilform_matrix(elems_test=elems_fine,
                                          elems_trial=elems_fine,
                                          use_mp=self.use_mp)

        # Evaluate rhs on the fine mesh.
        rhs = np.zeros(len(elems_fine))

        # Evaluate the dirichlet data
        if self.g:
            rhs += self.g(elems_fine)

        # Evaluate the RHS on the fine mesh.
        if self.M0:
            rhs -= self.M0.linform_vector(elems=elems_fine, use_mp=self.use_mp)

        # Solve
        time_solve_begin = time.time()
        Phi_fine = np.linalg.solve(mat_fine, rhs)
        print('Solving fine matrix took {}s'.format(time.time() -
                                                    time_solve_begin))

        # Prolongate the normal phi.
        Phi_prolong = np.repeat(Phi, 4)
        assert Phi_prolong[0] == Phi_prolong[1]

        # Calculate error.
        diff = Phi_fine - Phi_prolong
        return np.sqrt(diff.T @ mat_fine @ diff)
