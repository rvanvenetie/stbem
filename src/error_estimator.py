import hashlib
import math
import multiprocessing as mp
from math import sqrt

import cython
import numpy as np
import numpy.typing as npt

from .norms import Slobodeckij
from .quadrature import ProductScheme2D, gauss_quadrature_scheme


def MP_estim_l2(i):
    global __elems, __error_estimator, __residual
    return __error_estimator.weighted_l2(__elems[i], __residual)


def MP_estim_sobolev_time(i):
    global __elems, __error_estimator, __residual
    return __error_estimator.sobolev_time(__elems[i],
                                          __residual,
                                          nbrs_symmetry=True)


def MP_estim_sobolev_space(i):
    global __elems, __error_estimator, __residual
    return __error_estimator.sobolev_space(__elems[i],
                                           __residual,
                                           nbrs_symmetry=True)


class ErrorEstimator:
    def __init__(self, mesh, N_poly=5, cache_dir=None, problem=None):
        assert mesh.glue_space
        if not isinstance(N_poly, tuple):
            N_poly = (N_poly, N_poly, N_poly, N_poly)
        self.N_poly = N_poly
        N_weighted_l2, N_slobo_outer, N_slobo_time, N_slobo_space = N_poly
        print(
            'N_weighted_l2={}\nN_slobo_outer={}\nN_slobo_time={}\nN_slobo_space={}'
            .format(*N_poly))

        self.bdr_mesh = mesh
        self.gamma_len = mesh.gamma_space.gamma_length
        self.gauss_2d = ProductScheme2D(gauss_quadrature_scheme(N_weighted_l2))

        self.gauss = gauss_quadrature_scheme(N_slobo_outer)
        self.slobodeckij = Slobodeckij(N_slobo_time, N_slobo_space)

        # Storing options.
        self.cache_dir = cache_dir
        if problem is None: problem = str(self.bdr_mesh.gamma_space)
        self.problem = problem

    def __integrate_h_1_2(self, residual, t_a, t_b, elem_left, elem_right):
        val = np.zeros(self.gauss.weights.shape)
        h_t = float(t_b - t_a)
        points = t_a + h_t * self.gauss.points
        for i, t in enumerate(points):

            def residual_t(x_hat: npt.ArrayLike,
                           x: npt.ArrayLike) -> npt.ArrayLike:
                return residual(np.repeat(t, len(x_hat)), x_hat, x)

            if elem_right is None:
                val[i] = self.slobodeckij.seminorm_h_1_2(
                    residual_t, *elem_left.space_interval,
                    elem_left.gamma_space)
            elif elem_left.gamma_space is elem_right.gamma_space:
                gamma = elem_left.gamma_space
                assert np.allclose(gamma(elem_left.space_interval[1]),
                                   gamma(elem_right.space_interval[0]))
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

            def slo(t: npt.ArrayLike) -> npt.ArrayLike:
                return residual(t, np.repeat(x_hat, len(t)), gamma)

            val[i] = self.slobodeckij.seminorm_h_1_4(slo, t_a, t_b)

        approx = h_x * np.dot(val, self.gauss.weights)
        return approx

    def sobolev_space(self, elem, residual, nbrs_symmetry=False):
        """ This calculates the sobolev space error estimator.

        That is, for every neighbour along a time axis, we evaluate
            |r|^2_{L(J cap J'; H^{1/2}(K cup K'))}.
        """
        time_neighbours = [elem]
        for edge in elem.edges_axis(0):
            time_neighbours += edge.neighbour_elements()

        ips = []
        for time_nbr in time_neighbours:
            # If we use neighbour symmetry, we only evaluate this comb. once!
            if nbrs_symmetry and elem.glob_idx > time_nbr.glob_idx: continue

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
                assert time_nbr is elem
                elem_left = elem
                elem_right = None

            ips.append((time_nbr.glob_idx,
                        self.__integrate_h_1_2(residual, t_a, t_b, elem_left,
                                               elem_right)))
        assert len(ips) >= 1
        return math.fsum([val for elem, val in ips]), ips

    def sobolev_time(self, elem, residual, nbrs_symmetry=False):
        """ This calculates the sobolev time error estimator.

            That is, for every neighbour along a space axis, we evaluate
            |r|^2_{H^{1/4}(J cup J'; L_2(K cap  K')}.
        """
        space_neighbours = [elem]
        for edge in elem.edges_axis(1):
            space_neighbours += edge.neighbour_elements()

        ips = []
        for space_nbr in space_neighbours:
            # If we use neighbour symmetry, we only evaluate this comb. once!
            if nbrs_symmetry and elem.glob_idx > space_nbr.glob_idx: continue

            assert elem.gamma_space == space_nbr.gamma_space
            # Intersection.
            x_a = max(space_nbr.space_interval[0], elem.space_interval[0])
            x_b = min(space_nbr.space_interval[1], elem.space_interval[1])
            assert x_a < x_b

            # Union.
            t_a = min(space_nbr.time_interval[0], elem.time_interval[0])
            t_b = max(space_nbr.time_interval[1], elem.time_interval[1])

            ips.append((space_nbr.glob_idx,
                        self.__integrate_h_1_4(residual, t_a, t_b, x_a, x_b,
                                               elem.gamma_space)))
        assert len(ips) >= 1
        return math.fsum([val for elem, val in ips]), ips

    def weighted_l2(self, elem, residual):
        """ Residual takes arguments t, x_hat, x. """
        # Evaluate squared integral.
        t_a = elem.time_interval[0]
        x_a = elem.space_interval[0]
        t = np.array(t_a + elem.h_t * self.gauss_2d.points[0])
        x_hat = np.array(x_a + elem.h_x * self.gauss_2d.points[1])

        res_sqr = np.asarray(residual(t, x_hat, elem.gamma_space))**2
        res_l2 = np.dot(res_sqr, self.gauss_2d.weights)

        #  h_t * h_x * h_t^(-1/2) in time.
        #  h_t * h_x * h_x^(-1) in space.
        return sqrt(elem.h_t) * elem.h_x * res_l2, elem.h_t * res_l2

    def residual(self, elems, Phi, SL, M0u0=None, g=None, SL_exact_eval=False):
        """ Returns the residual function. """
        SL._init_elems(elems)

        @cython.locals(VPhi=cython.double)
        def residual(t: npt.ArrayLike, x_hat: npt.ArrayLike,
                     gamma: object) -> npt.ArrayLike:
            assert len(t) == len(x_hat)
            x = gamma(x_hat)
            result = np.zeros(len(t))
            for i, (t, x_hat, x) in enumerate(zip(t, x_hat, x.T)):
                # Evaluate the SL for our trial function.
                VPhi = 0
                for j, elem_trial in enumerate(elems):
                    if t <= elem_trial.time_interval[0]: continue
                    if SL_exact_eval and elem_trial.gamma_space is gamma:
                        VPhi += Phi[j] * SL.evaluate_exact(
                            elem_trial, t, x_hat)
                    else:
                        VPhi += Phi[j] * SL.evaluate(elem_trial, t, x_hat,
                                                     x.reshape(2, 1))
                result[i] = VPhi

                # Compare with rhs.
                if M0u0:
                    result[i] += M0u0(t, x.reshape(2, 1))
                if g:
                    result[i] -= g(t, x.reshape(2, 1))

            return result

        return residual

    def estimate_weighted_l2(self, elems, residual, use_mp=False):
        """ Returns the error estimator for given function Phi. """
        N = len(elems)
        if self.cache_dir is not None:
            md5 = hashlib.md5((str(self.bdr_mesh.gamma_space) +
                               str(elems)).encode()).hexdigest()
            cache_fn = "{}/weighted_l2_{}_{}_{}_{}.npy".format(
                self.cache_dir, self.problem, N, self.N_poly[0], md5)
            try:
                weighted_l2 = np.load(cache_fn)
                print('Loaded weighted L2 from {}.'.format(cache_fn))
                return weighted_l2
            except:
                pass

        if not use_mp:
            weighted_l2 = [self.weighted_l2(elem, residual) for elem in elems]
        else:
            globals()['__residual'] = residual
            globals()['__elems'] = elems
            globals()['__error_estimator'] = self
            cpu = mp.cpu_count()
            weighted_l2 = list(
                mp.Pool(cpu).map(MP_estim_l2, range(N), N // (8 * cpu) + 1))

        weighted_l2 = np.array(weighted_l2)
        if self.cache_dir is not None:
            print('Stored weighted L2 to {}.'.format(cache_fn))
            np.save(cache_fn, weighted_l2)
        return weighted_l2

    def estimate_sobolev(self, elems, residual, use_mp=False):
        """ Returns the error estimator for given function Phi. """
        N = len(elems)
        if self.cache_dir is not None:
            md5 = hashlib.md5((str(self.bdr_mesh.gamma_space) +
                               str(elems)).encode()).hexdigest()
            cache_fn = "{}/sobolev_{}_{}_{}_{}.npy".format(
                self.cache_dir, self.problem, N,
                '_'.join(str(y) for y in self.N_poly[1:]), md5)
            try:
                sobolev = np.load(cache_fn.format(N, self.problem, md5))
                print('Loaded Sobolev from {}.'.format(cache_fn))
                return sobolev
            except:
                pass

        if not use_mp:
            sobolev_time = [
                self.sobolev_time(elem, residual, nbrs_symmetry=True)
                for elem in elems
            ]
            sobolev_space = [
                self.sobolev_space(elem, residual, nbrs_symmetry=True)
                for elem in elems
            ]
        else:
            globals()['__residual'] = residual
            globals()['__elems'] = elems
            globals()['__error_estimator'] = self
            cpu = mp.cpu_count()
            with mp.Pool(cpu) as p:
                sobolev_time = list(
                    p.map(MP_estim_sobolev_time, range(N), N // (cpu * 8) + 1))
                sobolev_space = list(
                    p.map(MP_estim_sobolev_space, range(N),
                          N // (cpu * 8) + 1))

        # Silly code to correctly sum everything up, abuses symmetry
        # for speedup of factor 2.
        glob_2_loc = {elem.glob_idx: i for i, elem in enumerate(elems)}
        sobolev = np.zeros((N, 2))
        for i, elem in zip(range(N), elems):
            sobolev[i, 0] += sobolev_time[i][0]
            for elem_nbr, val_nbr in sobolev_time[i][1]:
                if elem.glob_idx < elem_nbr:
                    sobolev[glob_2_loc[elem_nbr], 0] += val_nbr
            sobolev[i, 1] += sobolev_space[i][0]
            for elem_nbr, val_nbr in sobolev_space[i][1]:
                if elem.glob_idx < elem_nbr:
                    sobolev[glob_2_loc[elem_nbr], 1] += val_nbr

        if self.cache_dir is not None:
            print('Stored Sobolev to {}.'.format(cache_fn))
            np.save(cache_fn, sobolev)
        return sobolev
