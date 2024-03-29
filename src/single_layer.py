import hashlib
import math
import multiprocessing as mp
import time
from math import pi, sqrt

import cython
import numpy as np
import numpy.typing as npt
from scipy.special import erf, expi

from .mesh import Element
from .quadrature import (DuffyScheme2D, ProductScheme2D,
                         gauss_quadrature_scheme, log_quadrature_scheme)
from .single_layer_exact import (spacetime_evaluated_1,
                                 spacetime_integrated_kernel)

FPI_INV = cython.declare(cython.double)
FPI_INV = (4 * pi)**-1
PI_SQRT = cython.declare(cython.double)
PI_SQRT = math.sqrt(pi)


def kernel(t, x):
    assert isinstance(t, float) and isinstance(x, float)
    if (t <= 0): return 0
    else: return FPI_INV * 1. / t * np.exp(-x**2 / (4 * t))


def alpha(z):
    """ Returns lambda a_z(x) """
    return lambda x: np.sum(x**2, axis=0) / (4 * z)


def noop(x):
    return 0


def g(a, b):
    """ Returns g_z for z = a - b. """
    if a <= b:
        return noop
    z = a - b
    return lambda x: FPI_INV * expi(-np.sum(x**2, axis=0) / (4 * z))


def f(a, b):
    """ Returns f_z for z = a - b"""
    if a <= b:
        return noop
    z = a - b

    def f_z(x_sqr):
        a_z = x_sqr / (4 * z)
        return FPI_INV * (z * np.exp(-a_z) + z * (1 + a_z) * expi(-a_z))

    return f_z


def time_integrated_kernel(t, a, b):
    """ Returns heat kernel G(t-s,x) integrated over s in [a,b]. """
    assert a < b
    g_ta = g(t, a)
    g_tb = g(t, b)
    return lambda x: g_tb(x) - g_ta(x)


def double_time_integrated_kernel(a, b, c, d):
    """ Returns kernel integrated in time over [a,b] x [c, d], """
    assert a < b and c < d

    def G(x):
        x_sqr = np.sum(x**2, axis=0) / 4
        result = 0
        if b > d:
            z = b - d
            result += FPI_INV * (z * np.exp(-x_sqr / z) +
                                 (x_sqr + z) * expi(-x_sqr / z))
        if b > c:
            z = b - c
            result -= FPI_INV * (z * np.exp(-x_sqr / z) +
                                 (x_sqr + z) * expi(-x_sqr / z))
        if a > c:
            z = a - c
            result += FPI_INV * (z * np.exp(-x_sqr / z) +
                                 (x_sqr + z) * expi(-x_sqr / z))
        if a > d:
            z = a - d
            result -= FPI_INV * (z * np.exp(-x_sqr / z) +
                                 (x_sqr + z) * expi(-x_sqr / z))

        return result

    return G


def MP_SL_matrix_col(j: int) -> npt.ArrayLike:
    """ Function to evaluate SL in parallel using the multiprocessing library. """
    global __SL, __elems_test, __elems_trial
    elem_trial = __elems_trial[j]
    col = np.zeros(len(__elems_test))
    for i, elem_test in enumerate(__elems_test):
        if elem_test.time_interval[1] <= elem_trial.time_interval[0]:
            continue
        col[i] = __SL.bilform(elem_trial, elem_test)
    return col


class SingleLayerOperator:
    def __init__(self, mesh, quad_order=12, pw_exact=False, cache_dir=None):
        self.pw_exact = pw_exact
        self.gauss_scheme = gauss_quadrature_scheme(23)
        self.gauss_2d = ProductScheme2D(self.gauss_scheme)
        self.log_scheme = log_quadrature_scheme(quad_order, quad_order)
        self.log_scheme_m = self.log_scheme.mirror()
        self.log_log = ProductScheme2D(self.log_scheme, self.log_scheme)
        self.duff_log_log = DuffyScheme2D(self.log_log, symmetric=False)
        self.mesh = mesh
        self.gamma_len = self.mesh.gamma_space.gamma_length
        self.glue_space = self.mesh.glue_space
        self.cache_dir = cache_dir
        self._init_elems(self.mesh.leaf_elements)

    def _init_elems(self, elems):
        # For all elements in the mesh, register the log scheme.
        for elem in elems:
            a, b = elem.space_interval
            elem.__log_scheme_y = elem.gamma_space(a + (b - a) *
                                                   self.log_scheme.points)
            elem.__log_scheme_m_y = elem.gamma_space(a + (b - a) *
                                                     self.log_scheme_m.points)

    @cython.locals(h_x=cython.double, h_y=cython.double)
    def __integrate(self, f: object, a: float, b: float, c: float,
                    d: float) -> float:
        """ Integrates a symmetric singular f over the square [a,b]x[c,d]. """
        h_x = b - a
        h_y = d - c
        assert h_x > 1e-8 and h_y > 1e-8
        assert (a < b and c < d)
        assert (a, b) <= (c, d)

        # If are the same panel.
        if a == c and b == d:
            return self.duff_log_log.integrate(f, a, b, c, d)

        # If the panels touch in the middle, split into even parts.
        if b == c:
            if abs(h_x - h_y) < 1e-10:
                return self.duff_log_log.mirror_x().integrate(f, a, b, c, d)
            elif h_x > h_y:
                return self.duff_log_log.mirror_x().integrate(
                    f, b - h_y, b, c, d) + self.__integrate(
                        f, a, b - h_y, c, d)
            else:
                return self.duff_log_log.mirror_x().integrate(
                    f, a, b, c, c + h_x) + self.__integrate(
                        f, a, b, c + h_x, d)
        assert not math.isclose(b, c)

        # If the panels touch through in the glued boundary, split into even parts.
        if a == 0 and d == self.gamma_len and self.glue_space:
            assert b < c
            if abs(h_x - h_y) < 1e-10:
                return self.duff_log_log.mirror_y().integrate(f, a, b, c, d)
            elif h_x > h_y:
                return self.duff_log_log.mirror_y().integrate(
                    f, a, a + h_y, c, d) + self.__integrate(
                        f, a + h_y, b, c, d)
            else:
                return self.__integrate(
                    f, a, b, c,
                    d - h_x) + self.duff_log_log.mirror_y().integrate(
                        f, a, b, d - h_x, d)

        # If we are disjoint.  TODO: Do more singular stuff if close?
        # TODO: Gauss 2d for disjoint..
        if b < c:
            #return self.gauss_2d.integrate(f, a, b, c, d)
            if c - b < self.gamma_len - d + a or not self.glue_space:
                return self.log_log.mirror_x().integrate(f, a, b, c, d)
            else:
                return self.log_log.mirror_y().integrate(f, a, b, c, d)

        # If the first panel is longer than the second panel.
        if d < b:
            # TODO: Is this correct?
            return self.__integrate(
                f, a, d, c, d) + self.duff_log_log.mirror_y().integrate(
                    f, d, b, c, d)

        # First panel is contained in second one.
        if a == c:
            assert b < d
            return self.__integrate(f, a, b, c, b) + self.__integrate(
                f, a, b, b, d)
        assert not math.isclose(a, c)

        # We have overlap, split this in two parts.
        assert a < c
        return self.__integrate(f, a, c, c, d) + self.__integrate(
            f, c, b, c, d)

    @cython.locals(a=cython.double,
                   b=cython.double,
                   c=cython.double,
                   d=cython.double)
    def bilform(self, elem_trial: Element, elem_test: Element) -> float:
        """ Evaluates <V 1_trial, 1_test>. """
        # If the test element lies below the trial element, we are done.
        if elem_test.time_interval[1] <= elem_trial.time_interval[0]:
            return 0

        if self.pw_exact and elem_test.gamma_space is elem_trial.gamma_space:
            return spacetime_integrated_kernel(*elem_test.time_interval,
                                               *elem_trial.time_interval,
                                               *elem_test.space_interval,
                                               *elem_trial.space_interval)

        a, b = elem_test.time_interval
        c, d = elem_trial.time_interval

        # Calculate the time integrated kernel.
        G_time = double_time_integrated_kernel(a, b, c, d)

        gamma_test = elem_test.gamma_space
        gamma_trial = elem_trial.gamma_space

        if elem_test.space_interval <= elem_trial.space_interval:
            G_time_parametrized = lambda x: G_time(
                gamma_test(x[0]) - gamma_trial(x[1]))

            return self.__integrate(G_time_parametrized,
                                    *elem_test.space_interval,
                                    *elem_trial.space_interval)
        else:
            # Swap x,y coordinates.
            G_time_parametrized = lambda x: G_time(
                gamma_test(x[1]) - gamma_trial(x[0]))

            return self.__integrate(G_time_parametrized,
                                    *elem_trial.space_interval,
                                    *elem_test.space_interval)

    def bilform_matrix(self, elems_test=None, elems_trial=None, use_mp=False):
        """ Returns the dense matrix <V 1_trial, 1_test>. """
        if elems_test is None:
            elems_test = list(self.mesh.leaf_elements)
        if elems_trial is None:
            elems_trial = elems_test

        N = len(elems_test)
        M = len(elems_trial)

        # For small N, M, simply construct matrix inline and return.
        if N * M < 100:
            mat = np.zeros((N, M))
            for i, elem_test in enumerate(elems_test):
                for j, elem_trial in enumerate(elems_trial):
                    mat[i, j] = self.bilform(elem_trial, elem_test)
            return mat

        if self.cache_dir is not None:
            md5 = hashlib.md5((str(self.mesh.gamma_space) + str(elems_test) +
                               str(elems_trial)).encode()).hexdigest()
            cache_fn = "{}/SL_{}_{}x{}_{}.npy".format(self.cache_dir,
                                                      self.mesh.gamma_space, N,
                                                      M, md5)
            try:
                mat = np.load(cache_fn)
                print("Loaded Single Layer from file {}".format(cache_fn))
                return mat
            except:
                pass

        time_mat_begin = time.time()

        mat = np.zeros((N, M))
        if not use_mp:
            for i, elem_test in enumerate(elems_test):
                for j, elem_trial in enumerate(elems_trial):
                    mat[i, j] = self.bilform(elem_trial, elem_test)
        else:
            # Set up global variables for parallelizing.
            globals()['__elems_test'] = elems_test
            globals()['__elems_trial'] = elems_trial
            globals()['__SL'] = self
            cpu = mp.cpu_count()
            for j, col in enumerate(
                    mp.Pool(mp.cpu_count()).imap(MP_SL_matrix_col, range(M),
                                                 M // (16 * cpu) + 1)):
                mat[:, j] = col

        if self.cache_dir is not None:
            try:
                np.save(cache_fn, mat)
                print("Stored Single Layer to {}".format(cache_fn))
            except:
                pass

        print('Calculating SL matrix took {}s'.format(time.time() -
                                                      time_mat_begin))
        return mat

    def potential(self, elem_trial, t, x):
        """ Evaluates (V 1_trial)(t,x) for t,x not on the bdr. """
        assert x.shape == (2, 1)
        if t <= elem_trial.time_interval[0]: return 0

        # Calculate the time integrated kernel.
        G_time = time_integrated_kernel(t, *elem_trial.time_interval)
        G_time_parametrized = lambda y: G_time(x - elem_trial.gamma_space(y))
        return self.gauss_scheme.integrate(G_time_parametrized,
                                           *elem_trial.space_interval)

    def potential_vector(self, t, x):
        """ Returns the vector (V 1_elem)(t, x) for all elements in mesh. """
        elems = list(self.mesh.leaf_elements)
        N = len(elems)
        vec = np.zeros(shape=N)
        for j, elem_trial in enumerate(elems):
            vec[j] = self.potential(elem_trial, t, x)
        return vec

    @cython.locals(x_a=cython.double,
                   x_b=cython.double,
                   d_a=cython.double,
                   d_b=cython.double,
                   t_a=cython.double,
                   t_b=cython.double)
    def evaluate(self, elem_trial: Element, t: float, x_hat: float,
                 x: npt.ArrayLike) -> float:
        """ Evaluates (V 1_trial)(t, gamma(x_hat)) for t, x_hat in the param domain. """
        if t <= elem_trial.time_interval[0]: return 0
        #if x is None: x = self.mesh.gamma_space.eval(x_hat)
        x_a = elem_trial.space_interval[0]
        x_b = elem_trial.space_interval[1]
        t_a = elem_trial.time_interval[0]
        t_b = elem_trial.time_interval[1]

        # Check if singularity lies in this element.
        if x_a * (1 + 1e-10) <= x_hat <= x_b * (1 - 1e-10):
            # Calculate the time integrated kernel.
            def G_time_parametrized(y_hat: npt.ArrayLike):
                xy = (x - elem_trial.gamma_space(y_hat))**2
                xy = xy[0] + xy[1]
                a, b = elem_trial.time_interval
                if t <= b:
                    return -FPI_INV * expi(-xy / (4 * (t - a)))
                else:
                    return FPI_INV * (expi(-xy / (4 *
                                                  (t - b))) - expi(-xy /
                                                                   (4 *
                                                                    (t - a))))

            return self.log_scheme_m.integrate(
                G_time_parametrized, x_a, x_hat) + self.log_scheme.integrate(
                    G_time_parametrized, x_hat, x_b)

        # Calculate distance of x_hat to both endpoints.
        if self.glue_space:
            d_a = min(abs(x_hat - x_a), abs(self.gamma_len - x_hat + x_a))
            d_b = min(abs(x_hat - x_b), abs(self.gamma_len - x_b + x_hat))
        else:
            d_a = abs(x_hat - x_a)
            d_b = abs(x_hat - x_b)

        # Calculate |x - gamma(yhat)|^2 for the quadrature rule.
        if d_a <= d_b:
            xy_sqr = (x - elem_trial.__log_scheme_y)**2
        else:
            xy_sqr = (x - elem_trial.__log_scheme_m_y)**2
        xy = xy_sqr[0] + xy_sqr[1]

        # Evaluate the time integrated kernel for the above points.
        if t <= t_b:
            vec = -FPI_INV * expi(-xy / (4 * (t - t_a)))
        else:
            vec = FPI_INV * (expi(-xy / (4 * (t - t_b))) - expi(-xy /
                                                                (4 *
                                                                 (t - t_a))))
        # Return the quadrature result.
        return (x_b - x_a) * np.dot(self.log_scheme.weights, vec)

    def evaluate_exact(self, elem_trial: Element, t: float, x: float) -> float:
        """ Evaluates (V 1_trial)(t, x) for elem_trial lying on the
            same pane as x. """
        if t <= elem_trial.time_interval[0]: return 0
        a, b = elem_trial.space_interval
        if x < a or x > b:
            h = min(abs(a - x), abs(b - x))
            k = max(abs(a - x), abs(b - x))
            a, b = elem_trial.time_interval
            if t <= b:
                return -FPI_INV * (PI_SQRT * (2 * sqrt(
                    (t - a))) * (erf(h / (2 * sqrt(
                        (t - a)))) - erf(k / (2 * sqrt(
                            (t - a))))) - h * expi(-(h**2 / (4 * (t - a)))) +
                                   k * expi(-(k**2 / (4 * (t - a)))))
            else:
                return FPI_INV * (
                    2 * PI_SQRT *
                    (sqrt(t - a) *
                     (-erf(h / (2 * sqrt(t - a))) + erf(k /
                                                        (2 * sqrt(t - a)))) +
                     sqrt(t - b) *
                     (erf(h / (2 * sqrt(t - b))) - erf(k /
                                                       (2 * sqrt(t - b))))) +
                    h * expi(h**2 / (4 * (a - t))) - k * expi(k**2 /
                                                              (4 * (a - t))) -
                    h * expi(h**2 / (4 * (b - t))) + k * expi(k**2 /
                                                              (4 * (b - t))))
        elif a < x < b:
            return spacetime_evaluated_1(
                t, *elem_trial.time_interval, x - a) + spacetime_evaluated_1(
                    t, *elem_trial.time_interval, b - x)
        elif x == a or x == b:
            return spacetime_evaluated_1(t, *elem_trial.time_interval, b - a)

    def evaluate_vector(self, t, x_hat):
        """ Returns the vector (V 1_elem)(t, gamma(x_hat)) for all elements in mesh. """
        elems = list(self.mesh.leaf_elements)
        N = len(elems)
        vec = np.zeros(shape=N)
        x = self.mesh.gamma_space.eval(x_hat)
        for j, elem_trial in enumerate(elems):
            vec[j] = self.evaluate(elem_trial, t, x_hat, x)
        return vec

    def rhs_vector(self, f, gauss_order=23):
        """ Returns the vector f(1_elem) for all elements in the mesh. """
        gauss_scheme = gauss_quadrature_scheme(gauss_order)
        gauss_2d = ProductScheme2D(gauss_scheme, gauss_scheme)
        elems = list(self.mesh.leaf_elements)
        N = len(elems)
        vec = np.zeros(shape=N)
        for i, elem_test in enumerate(elems):
            f_param = lambda tx: f(tx[0], elem_test.gamma_space(tx[1]))
            vec[i] = gauss_2d.integrate(f_param, *elem_test.time_interval,
                                        *elem_test.space_interval)
        return vec
