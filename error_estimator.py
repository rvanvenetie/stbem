import random
import multiprocessing as mp
import time
import math
import numpy as np
from quadrature import gauss_quadrature_scheme, DuffyScheme2D, ProductScheme2D, gauss_sqrtinv_quadrature_scheme
from pprint import pprint
from mesh import MeshParametrized
from parametrization import UnitSquare
from norms import Slobodeckij


def MP_estim_l2(i):
    global __elems, __error_estimator, __residual
    return __error_estimator.weighted_l2(__elems[i], __residual)


def MP_estim_sobolev(i):
    global __elems, __error_estimator, __residual
    return __error_estimator.sobolev(__elems[i], __residual)


class ErrorEstimator:
    def __init__(self, mesh, N_poly=5):
        assert mesh.glue_space
        self.gamma_len = mesh.gamma_space.gamma_length

        self.gauss = gauss_quadrature_scheme(N_poly)
        self.gauss_2d = ProductScheme2D(gauss_quadrature_scheme(N_poly))
        self.slobodeckij = Slobodeckij(N_poly)

    def __integrate_h_1_2(self, residual, t_a, t_b, elem_left, elem_right):
        val = np.zeros(self.gauss.weights.shape)
        h_t = float(t_b - t_a)
        points = t_a + h_t * self.gauss.points
        for i, t in enumerate(points):
            residual_t = lambda x_hat, x: residual(np.repeat(t, len(x_hat)),
                                                   x_hat, x)

            if elem_left.gamma_space is elem_right.gamma_space:
                gamma = elem_left.gamma_space
                #assert np.all(
                #    gamma(elem_left.space_interval[1]) == gamma(
                #        elem_right.space_interval[0]))
                val[i] = self.slobodeckij.seminorm_h_1_2(
                    residual_t, elem_left.space_interval[0],
                    elem_right.space_interval[1], gamma)
            else:
                val[i] = self.slobodeckij.seminorm_h_1_2_pw(
                    residual_t, *elem_left.space_interval,
                    elem_left.gamma_space, *elem_right.space_interval,
                    elem_right.gamma_space)

        approx = h_t * np.dot(val, self.gauss.weights)
        return approx

    def __integrate_h_1_4(self, residual, t_a, t_b, x_a, x_b, gamma):
        val = np.zeros(self.gauss.weights.shape)
        h_x = float(x_b - x_a)

        points = x_a + h_x * self.gauss.points
        for i, x_hat in enumerate(points):

            def slo(t):
                return residual(t, np.repeat(x_hat, len(t)), gamma)

            val[i] = self.slobodeckij.seminorm_h_1_4(slo, t_a, t_b)

        approx = h_x * np.dot(val, self.gauss.weights)
        return approx

    def sobolev_space(self, elem, residual):
        """ This calculates the sobolev space error estimator.
            
        That is, for every neighbour along a time axis, we evaluate
            |r|^2_{L(J cap J'; H^{1/2}(K cup K'))}.
        """
        time_neighbours = []
        for edge in elem.edges_axis(0):
            time_neighbours += edge.neighbour_elements()

        ips = []
        for time_nbr in time_neighbours:
            t_a = max(time_nbr.time_interval[0], elem.time_interval[0])
            t_b = min(time_nbr.time_interval[1], elem.time_interval[1])
            assert t_a < t_b

            # Determine the space parametrization.
            if time_nbr.vertices[2].x == self.gamma_len and elem.vertices[
                    0].x == 0:
                elem_left = time_nbr
                elem_right = elem
            elif elem.vertices[2].x == self.gamma_len and time_nbr.vertices[
                    0].x == 0:
                elem_left = elem
                elem_right = time_nbr
            elif elem.vertices[0].x < time_nbr.vertices[0].x:
                elem_left = elem
                elem_right = time_nbr
            elif time_nbr.vertices[0].x < elem.vertices[0].x:
                elem_left = time_nbr
                elem_right = elem
            else:
                assert False

            ips.append((time_nbr,
                        self.__integrate_h_1_2(residual, t_a, t_b, elem_left,
                                               elem_right)))
        assert len(ips) >= 2
        return math.fsum([val for elem, val in ips]), ips

    def sobolev_time(self, elem, residual):
        """ This calculates the sobolev time error estimator. 

            That is, for every neighbour along a space axis, we evaluate
            |r|^2_{H^{1/4}(J cup J'; L_2(K cap  K')}.
        """
        space_neighbours = []
        for edge in elem.edges_axis(1):
            space_neighbours += edge.neighbour_elements()

        ips = []
        for space_nbr in space_neighbours:
            assert elem.gamma_space == space_nbr.gamma_space
            # Intersection.
            x_a = max(space_nbr.space_interval[0], elem.space_interval[0])
            x_b = min(space_nbr.space_interval[1], elem.space_interval[1])
            assert x_a < x_b

            # Union.
            t_a = min(space_nbr.time_interval[0], elem.time_interval[0])
            t_b = max(space_nbr.time_interval[1], elem.time_interval[1])

            ips.append((space_nbr,
                        self.__integrate_h_1_4(residual, t_a, t_b, x_a, x_b,
                                               elem.gamma_space)))

        assert len(ips) >= 2
        return math.fsum([val for elem, val in ips]), ips

    def weighted_l2(self, elem, residual):
        """ Residual takes arguments t, x_hat, x. """
        # Evaluate squared integral.
        t_a, x_a = elem.time_interval[0], elem.space_interval[0]
        t = np.array(t_a + elem.h_t * self.gauss_2d.points[0])
        x_hat = np.array(x_a + elem.h_x * self.gauss_2d.points[1])

        res_sqr = np.asarray(residual(t, x_hat, elem.gamma_space))**2
        res_l2 = elem.h_x * elem.h_t * np.dot(res_sqr, self.gauss_2d.weights)

        # Return the weighted l2 norm.
        return elem.h_t**(-1 / 2) * res_l2, elem.h_x**(-1) * res_l2

    def sobolev(self, elem, residual):
        return self.sobolev_time(elem, residual)[0], self.sobolev_space(
            elem, residual)[0]

    def residual_pw(self, elems, Phi, SL, M0u0):
        """ Returns the residual function for a pw polygonal domain. """
        SL._init_elems()

        def residual(t, x_hat, gamma):
            assert len(t) == len(x_hat)
            x = gamma(x_hat)
            result = np.zeros(len(t))
            for i, (t, x_hat, x) in enumerate(zip(t, x_hat, x.T)):
                # Evaluate the SL for our trial function.
                VPhi = 0
                for j, elem_trial in enumerate(elems):
                    if t <= elem_trial.time_interval[0]: continue
                    if elem_trial.gamma_space is gamma:
                        VPhi += Phi[j] * SL.evaluate_pw(elem_trial, t, x_hat)
                    else:
                        VPhi += Phi[j] * SL.evaluate(elem_trial, t, x_hat,
                                                     x.reshape(2, 1))

                # Compare with rhs.
                result[i] = VPhi + M0u0(t, x.reshape(2, 1))
            return result

        return residual

    def estimate_weighted_l2(self, elems, residual, use_mp=False):
        """ Returns the error estimator for given function Phi. """
        if not use_mp:
            weighted_l2 = [self.weighted_l2(elem, residual) for elem in elems]
        else:
            N = len(elems)
            globals()['__residual'] = residual
            globals()['__elems'] = elems
            globals()['__error_estimator'] = self
            cpu = mp.cpu_count()
            weighted_l2 = list(mp.Pool(cpu).map(MP_estim_l2, range(N), 10))

        weighted_l2 = np.array(weighted_l2)
        return weighted_l2

    def estimate_sobolev(self, elems, residual, use_mp=False):
        """ Returns the error estimator for given function Phi. """
        if not use_mp:
            sobolev = [self.sobolev(elem, residual) for elem in elems]
        else:
            N = len(elems)
            globals()['__residual'] = residual
            globals()['__elems'] = elems
            globals()['__error_estimator'] = self
            cpu = mp.cpu_count()
            sobolev = list(mp.Pool(cpu).map(MP_estim_sobolev, range(N), 10))

        sobolev = np.array(sobolev)
        return sobolev


if __name__ == "__main__":

    def residual(t, x_hat, x):
        return t * x_hat

    mesh = MeshParametrized(UnitSquare())
    mesh.uniform_refine()
    mesh.uniform_refine()
    error_estimator = ErrorEstimator(mesh, N_poly=11)
    for elem in mesh.leaf_elements:
        print(elem, error_estimator.Sobolev(elem, residual)[1])
