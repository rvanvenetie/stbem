from mesh import Vertex
import numpy as np


class DummyElement:
    """ Needed for calculation of the Hierarchical Error Estimator. """
    def __init__(self, vertices, parent):
        self.vertices = vertices
        self.parent = parent
        self.gamma_space = parent.gamma_space

        self.time_interval = float(self.vertices[0].t), float(
            self.vertices[2].t)
        self.space_interval = float(self.vertices[0].x), float(
            self.vertices[2].x)
        self.h_t = float(abs(self.vertices[2].t - self.vertices[0].t))
        self.h_x = float(abs(self.vertices[2].x - self.vertices[0].x))

    def __repr__(self):
        return "Elem(t={}, x={})".format(self.time_interval,
                                         self.space_interval)


class HierarchicalErrorEstimator:
    def __init__(self, SL, M0=None):
        self.SL = SL
        self.M0 = M0

    def __uniform_refinement(self, elems):
        """ Returns the uniform refinement of the given elements. """
        result = []
        for elem_coarse in elems:
            v0, v1, v2, v3 = elem_coarse.vertices
            v01 = Vertex(t=(v0.t + v1.t) / 2, x=(v0.x + v1.x) / 2, idx=-1)
            v12 = Vertex(t=(v1.t + v2.t) / 2, x=(v1.x + v2.x) / 2, idx=-1)
            v23 = Vertex(t=(v2.t + v3.t) / 2, x=(v2.x + v3.x) / 2, idx=-1)
            v30 = Vertex(t=(v3.t + v0.t) / 2, x=(v3.x + v0.x) / 2, idx=-1)
            vi = Vertex(t=(v0.t + v2.t) / 2, x=(v0.x + v2.x) / 2, idx=-1)

            children = [
                DummyElement(vertices=[v0, v01, vi, v30], parent=elem_coarse),
                DummyElement(vertices=[v01, v1, v12, vi], parent=elem_coarse),
                DummyElement(vertices=[v30, vi, v23, v3], parent=elem_coarse),
                DummyElement(vertices=[vi, v12, v2, v23], parent=elem_coarse),
            ]

            result.append(children)
        return result

    def estimate(self, elems, Phi):
        """ Returns the hierarchical basis estimator for given function Phi. """

        # Calcualte uniform refinement of the mesh.
        elems_coarse = elems
        elem_2_children = self.__uniform_refinement(elems_coarse)

        # Flatten list and calculate mapping of indices.
        elems_fine = [
            child for children in elem_2_children for child in children
        ]
        elem_2_idx_fine = {k: v for v, k in enumerate(elems_fine)}

        # Evaluate SL matrix tested with the fine mesh.
        mat = self.SL.bilform_matrix(elems_test=elems_fine,
                                     elems_trial=elems_coarse,
                                     cache_dir='data',
                                     use_mp=True)
        VPhi = mat @ Phi

        # Evaluate the RHS on the fine mesh.
        rhs = -self.M0.linform_vector(
            elems=elems_fine, cache_dir='data', use_mp=True)

        estim = np.zeros(len(elems_coarse))
        for i, elem_coarse in enumerate(elems_coarse):
            S = self.SL.bilform_matrix(elem_2_children[i], elem_2_children[i])
            children = [elem_2_idx_fine[elem] for elem in elem_2_children[i]]
            #scaling = sum(mat[j, i] for j in children)

            estim_loc = np.zeros(3)
            for k, coefs in enumerate([[1, 1, -1, -1], [1, -1, 1, -1],
                                       [1, -1, -1, 1]]):
                rhs_estim = 0
                V_estim = 0
                for j, c in zip(children, coefs):
                    rhs_estim += rhs[j] * c
                    V_estim += VPhi[j] * c
                coefs = np.array(coefs)
                scaling_estim = coefs @ (S @ coefs.T)
                estim_loc[k] = abs(rhs_estim - V_estim)**2 / scaling_estim
            estim[i] = np.sum(estim_loc)

        return np.sqrt(np.sum(estim)), estim
