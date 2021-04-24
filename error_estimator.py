import random
import numpy as np
from quadrature import gauss_quadrature_scheme, DuffyScheme2D, ProductScheme2D
from pprint import pprint
from mesh import MeshParametrized
from parametrization import UnitSquare


class ErrorEstimator:
    def __init__(self, mesh, N_poly=5):
        self.mesh = mesh

        self.gauss = gauss_quadrature_scheme(N_poly)
        self.gauss_2d = ProductScheme2D(self.gauss)
        self.duff_2d = DuffyScheme2D(self.gauss_2d, symmetric=True)

    def WeightedL2(self, elem, residual):
        """ Residual takes arguments t, x_hat, x. """
        # Evaluate squared integral.
        t_a, x_a = elem.time_interval[0], elem.space_interval[0]
        t = np.array(t_a + elem.h_t * self.gauss_2d.points[0])
        x_hat = np.array(x_a + elem.h_x * self.gauss_2d.points[1])
        x = elem.gamma_space(x_hat)

        res_sqr = np.asarray(residual(t, x_hat, x))**2
        res_l2 = elem.h_x * elem.h_t * np.dot(res_sqr, self.gauss_2d.weights)

        # Return the weighted l2 norm.
        return (elem.h_x**(-1) + elem.h_t**(-1 / 2)) * res_l2

    def Sobolev(self, elem, residual):
        time_neighbours = []
        for edge in elem.edges_axis(0):
            time_neighbours += edge.neighbour_elements()

        space_neighbours = []
        for edge in elem.edges_axis(1):
            space_neighbours += edge.neighbour_elements()

        print(elem)
        for time_nbr in time_neighbours:
            t_a, t_b = max(time_nbr.vertices[0].t,
                           elem.vertices[0].t), min(time_nbr.vertices[2].t,
                                                    elem.vertices[2].t)
            print('\t', time_nbr, float(t_a), float(t_b))
            if time_nbr.gamma_space == elem.gamma_space:
                x_a, x_b = min(time_nbr.vertices[0].x,
                               elem.vertices[0].x), max(
                                   time_nbr.vertices[2].x, elem.vertices[2].x)
                print('\t same gamma', float(x_a), float(x_b))

                val = np.zeros(gauss.weights.shape)
                gamma = elem.gamma_space
                for i, t in enumerate(gauss.points):

                    def slo(xy):
                        x = gamma(xy[0])
                        y = gamma(xy[1])
                        xy_sqr = np.sum((x - y)**2, axis=0)
                        return (residual(t, xy[0]) -
                                residual(t, xy[1]))**2 / xy_sqr

                    val[i] = gauss_duff_2d.integrate(slo, x_a, x_b, x_a, x_b)

                approx = np.dot(val, gauss.weights)
                print(approx)
