import random
import math
import numpy as np
from quadrature import gauss_quadrature_scheme, DuffyScheme2D, ProductScheme2D, gauss_sqrtinv_quadrature_scheme
from pprint import pprint
from mesh import MeshParametrized
from parametrization import UnitSquare
from norms import Slobodeckij


class ErrorEstimator:
    def __init__(self, mesh, N_poly=5):
        assert mesh.glue_space
        self.gamma_len = mesh.gamma_space.gamma_length

        self.gauss = gauss_quadrature_scheme(N_poly)
        self.gauss_2d = ProductScheme2D(self.gauss)
        self.duff_2d = DuffyScheme2D(self.gauss_2d, symmetric=True)

        self.slobodeckij = Slobodeckij(N_poly)

    def __integrate_h_1_2(self, residual, t_a, t_b, x_a, x_b, gamma):
        val = np.zeros(self.gauss.weights.shape)
        h_t = float(t_b - t_a)

        points = t_a + h_t * self.gauss.points
        for i, t in enumerate(points):

            def slo(xy):
                x = gamma(xy[0])
                y = gamma(xy[1])
                xy_sqr = np.sum((x - y)**2, axis=0)
                t_np = np.repeat(t, len(xy[0]))
                return (residual(t_np, xy[0], x) -
                        residual(t_np, xy[1], y))**2 / xy_sqr

            val[i] = self.duff_2d.integrate(slo, x_a, x_b, x_a, x_b)

        approx = h_t * np.dot(val, self.gauss.weights)
        return approx

    def __integrate_h_1_4(self, residual, t_a, t_b, x_a, x_b, gamma):
        val = np.zeros(self.gauss.weights.shape)
        h_x = float(x_b - x_a)

        points = x_a + h_x * self.gauss.points
        for i, x_hat in enumerate(points):
            x = gamma(x_hat)

            def slo(t):
                return residual(t, x_hat, x)

            val[i] = self.slobodeckij.seminorm_h_1_4(slo, t_a, t_b)

        approx = h_x * np.dot(val, self.gauss.weights)
        return approx

    def Sobolev_time(self, elem, residual):
        """ This calculates the sobolev time error estimator. """
        time_neighbours = []
        for edge in elem.edges_axis(0):
            time_neighbours += edge.neighbour_elements()

        ips = []
        for time_nbr in time_neighbours:
            t_a = max(time_nbr.vertices[0].t, elem.vertices[0].t)
            t_b = min(time_nbr.vertices[2].t, elem.vertices[2].t)
            assert t_a < t_b

            # Determine the space parametrization.
            if time_nbr.gamma_space == elem.gamma_space:
                x_a = min(time_nbr.vertices[0].x, elem.vertices[0].x)
                x_b = max(time_nbr.vertices[2].x, elem.vertices[2].x)
                gamma = elem.gamma_space
            elif time_nbr.vertices[2].x == self.gamma_len and elem.vertices[
                    0].x == 0:
                x_a = -time_nbr.h_x
                x_b = float(elem.vertices[2].x)
                gamma = lambda x_hat: np.select([x_hat < 0, x_hat >= 0], [
                    time_nbr.gamma_space(x_hat + self.gamma_len),
                    elem.gamma_space(x_hat)
                ])
            elif elem.vertices[2].x == self.gamma_len and time_nbr.vertices[
                    0].x == 0:
                x_a = -elem.h_x
                x_b = float(time_nbr.vertices[2].x)
                gamma = lambda x_hat: np.select([x_hat < 0, x_hat >= 0], [
                    elem.gamma_space(x_hat + self.gamma_len),
                    time_nbr.gamma_space(x_hat)
                ])
            elif elem.vertices[0].x < time_nbr.vertices[0].x:
                middle = float(time_nbr.vertices[0].x)
                x_a = float(elem.vertices[0].x)
                x_b = float(time_nbr.vertices[2].x)

                gamma = lambda x_hat: np.select([
                    x_hat < middle, x_hat >= middle
                ], [elem.gamma_space(x_hat),
                    time_nbr.gamma_space(x_hat)])
            elif time_nbr.vertices[0].x < elem.vertices[0].x:
                middle = float(elem.vertices[0].x)
                x_a = float(time_nbr.vertices[0].x)
                x_b = float(elem.vertices[2].x)

                gamma = lambda x_hat: np.select([
                    x_hat < middle, x_hat >= middle
                ], [time_nbr.gamma_space(x_hat),
                    elem.gamma_space(x_hat)])
            else:
                assert False

            #assert np.allclose(gamma(middle - 1e-10), gamma(middle + 1e-10))
            assert x_a < x_b
            ips.append((time_nbr,
                        self.__integrate_h_1_2(residual, t_a, t_b, x_a, x_b,
                                               gamma)))
        return math.fsum([val for elem, val in ips]), ips

    def Sobolev_space(self, elem, residual):
        """ This calculates the sobolev space error estimator. """
        space_neighbours = []
        for edge in elem.edges_axis(1):
            space_neighbours += edge.neighbour_elements()

        ips = []
        for space_nbr in space_neighbours:
            assert elem.gamma_space == space_nbr.gamma_space
            # Intersection.
            x_a = max(space_nbr.vertices[0].x, elem.vertices[0].x)
            x_b = min(space_nbr.vertices[2].x, elem.vertices[2].x)
            assert x_a < x_b

            # Union.
            t_a = min(space_nbr.vertices[0].t, elem.vertices[0].t)
            t_b = max(space_nbr.vertices[2].t, elem.vertices[2].t)

            ips.append((space_nbr,
                        self.__integrate_h_1_4(residual, t_a, t_b, x_a, x_b,
                                               ellem.gamma_space)))
        return math.fsum([val for elem, val in ips]), ips

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
        return self.Sobolev_time(elem, residual)


if __name__ == "__main__":

    def residual(t, x_hat, x):
        return t * x_hat

    mesh = MeshParametrized(UnitSquare())
    mesh.uniform_refine()
    mesh.uniform_refine()
    error_estimator = ErrorEstimator(mesh, N_poly=11)
    for elem in mesh.leaf_elements:
        print(elem, error_estimator.Sobolev(elem, residual)[1])
