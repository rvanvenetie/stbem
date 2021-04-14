import numpy as np
import random
from parametrization import Circle, UnitSquare, LShape
from mesh import Mesh, MeshParametrized
import itertools
from scipy.special import expi, erf, expn, erfc
from quadrature import log_quadrature_scheme, gauss_quadrature_scheme, ProductScheme2D, DuffyScheme2D
from mesh import Element

FPI_INV = (4 * np.pi)**-1
TPI_INV = (2 * np.pi)**-1


def kernel(t, x):
    assert isinstance(t, float) and isinstance(x, float)
    if (t <= 0): return 0
    else: return FPI_INV * 1. / t * np.exp(-x**2 / (4 * t))


def alpha(z):
    """ Returns lambda a_z(x) """
    return lambda x: np.sum(x**2, axis=0) / (4 * z)


def g(a, b):
    """ Returns g_z for z = a - b. """
    if a <= b:
        g_z = lambda x: 0
    else:
        z = a - b
        a_z = alpha(z)
        g_z = lambda x: FPI_INV * expi(-a_z(x))
    return g_z


def f(a, b):
    """ Returns f_z for z = a - b"""
    if a <= b:
        f_z = lambda x: 0
    else:
        z = a - b
        a_z = alpha(z)
        f_z = lambda x: FPI_INV * (z * np.exp(-a_z(x)) + z *
                                   (1 + a_z(x)) * expi(-a_z(x)))
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
    f_bd = f(b, d)
    f_bc = f(b, c)
    f_ca = f(a, c)
    f_da = f(a, d)
    return lambda x: f_bd(x) - f_bc(x) + f_ca(x) - f_da(x)


class SingleLayerOperator:
    def __init__(self, mesh, quad_order=12):
        self.gauss_scheme = gauss_quadrature_scheme(quad_order +
                                                    (quad_order + 1) % 2)
        self.log_scheme = log_quadrature_scheme(quad_order, quad_order)
        self.log_log = ProductScheme2D(self.log_scheme, self.log_scheme)
        self.duff_log_log = DuffyScheme2D(self.log_log, symmetric=False)
        self.mesh = mesh
        self.gamma_len = self.mesh.gamma_space.gamma_length

    def __integrate(self, f, a, b, c, d):
        """ Integrates a symmetric singular f over the square [a,b]x[c,d]. """
        assert (a < b and c < d)
        assert (a, b) <= (c, d)

        # If are the same panel.
        if a == c and b == d:
            return self.duff_log_log.integrate(f, a, b, c, d)

        # If the panels touch in the middle. TODO: Split into even parts?
        if b == c:
            return self.duff_log_log.mirror_x().integrate(f, a, b, c, d)
        if a == 0 and d == self.gamma_len and self.mesh.glue_space:
            assert b < c
            return self.duff_log_log.mirror_y().integrate(f, a, b, c, d)

        # If we are disjoint.  TODO: Do more singular stuff if close?
        if b < c:
            if c - b < self.gamma_len - d + a or not self.mesh.glue_space:
                return self.log_log.mirror_x().integrate(f, a, b, c, d)
            else:
                return self.log_log.mirror_y().integrate(f, a, b, c, d)

        # If the first panel is longer than the second panel.
        if d < b:
            return self.__integrate(
                f, a, d, c, d) + self.duff_log_log.mirror_y().integrate(
                    f, d, b, c, d)

        # First panel is contained in second one.
        if a == c:
            assert b < d
            return self.__integrate(f, a, b, c, b) + self.__integrate(
                f, a, b, b, d)

        # We have overlap, split this in two parts.
        assert a < c
        return self.__integrate(f, a, c, c, d) + self.__integrate(
            f, c, b, c, d)

        assert False

    def bilform(self, elem_trial, elem_test):
        """ Evaluates <V 1_trial, 1_test>. """
        # If the test element lies below the trial element, we are done.
        if elem_test.time_interval[1] <= elem_trial.time_interval[0]:
            return 0

        # Calculate the time integrated kernel.
        G_time = double_time_integrated_kernel(*elem_test.time_interval,
                                               *elem_trial.time_interval)
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

    def bilform_matrix(self):
        """ Returns the dense matrix <V 1_trial, 1_test>. """
        elems = list(self.mesh.leaf_elements)
        N = len(elems)
        mat = np.zeros(shape=(N, N))
        for i, elem_test in enumerate(elems):
            for j, elem_trial in enumerate(elems):
                mat[i, j] = self.bilform(elem_trial, elem_test)
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

    def evaluate(self, elem_trial, t, x_hat):
        """ Evaluates (V 1_trial)(t, gamma(x_hat)) for t, x_hat in the param domain. """
        if t <= elem_trial.time_interval[0]: return 0

        # Calculate the time integrated kernel.
        x = self.mesh.gamma_space.eval(x_hat)
        G_time = time_integrated_kernel(t, *elem_trial.time_interval)
        G_time_parametrized = lambda y: G_time(x - elem_trial.gamma_space(y))

        # Integrate. Check where singularity lies, i.e. for y = x_hat.
        a, b = elem_trial.space_interval
        if a < x_hat < b:
            return self.log_scheme.mirror().integrate(
                G_time_parametrized, a, x_hat) + self.log_scheme.integrate(
                    G_time_parametrized, x_hat, b)

        # Calculate distance of x_hat to both endpoints.
        if not self.mesh.glue_space:
            d_a = abs(x_hat - a)
            d_b = abs(x_hat - b)
        else:
            d_a = min(abs(x_hat - a), abs(self.gamma_len - x_hat + a))
            d_b = min(abs(x_hat - b), abs(self.gamma_len - b + x_hat))

        if d_a <= d_b:
            return self.log_scheme.integrate(G_time_parametrized, a, b)
        else:
            return self.log_scheme.mirror().integrate(G_time_parametrized, a,
                                                      b)

    def evaluate_vector(self, t, x_hat):
        """ Returns the vector (V 1_elem)(t, gamma(x_hat)) for all elements in mesh. """
        elems = list(self.mesh.leaf_elements)
        N = len(elems)
        vec = np.zeros(shape=N)
        for j, elem_trial in enumerate(elems):
            vec[j] = self.evaluate(elem_trial, t, x_hat)
        return vec

    def rhs_vector(self, f, gauss_order=23):
        """ Returns the vector f(1_elem) for all elements in the mesh. """
        gauss_scheme = gauss_quadrature_scheme(23)
        gauss_2d = ProductScheme2D(gauss_scheme, gauss_scheme)
        elems = list(self.mesh.leaf_elements)
        N = len(elems)
        vec = np.zeros(shape=N)
        for i, elem_test in enumerate(elems):
            f_param = lambda tx: f(tx[0], elem_test.gamma_space(tx[1]))
            vec[i] = gauss_2d.integrate(f_param, *elem_test.time_interval,
                                        *elem_test.space_interval)
        return vec


if __name__ == "__main__":
    mesh = MeshParametrized(Circle())
    SL = SingleLayerOperator(mesh)
    elems = list(mesh.leaf_elements)
    print(SL.evaluate(elems[0], 1, 0))

    adfa
    for gamma in [Circle(), UnitSquare(), LShape()]:
        mesh = MeshParametrized(gamma)
        dim = 4
        scheme_stroud = quadpy.cn.stroud_cn_7_1(dim)
        scheme_mcnamee = quadpy.cn.mcnamee_stenger_9b(dim)
        print(scheme_stroud.points.shape, scheme_mcnamee.points.shape)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(200):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        # We can exactly integrate the kernel if the space part coincides.
        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for i, elem_test in enumerate(elems):
            for j, elem_trial in enumerate(elems):
                if elem_test.time_interval[0] <= elem_trial.time_interval[1]:
                    continue

                val = SL.bilform(elem_trial, elem_test)

                def kernel(x):
                    #assert np.all((elem_test.time_interval[0] <= x[0])
                    #              & (x[0] <= elem_test.time_interval[1]))
                    #assert np.all((elem_test.space_interval[0] <= x[1])
                    #              & (x[1] <= elem_test.space_interval[1]))
                    #assert np.all((elem_trial.time_interval[0] <= x[2])
                    #              & (x[2] <= elem_trial.time_interval[1]))
                    #assert np.all((elem_trial.space_interval[0] <= x[3])
                    #              & (x[3] <= elem_trial.space_interval[1]))
                    ts = x[0] - x[2]
                    #assert np.all(ts > 0.5)
                    xy = elem_test.gamma_space(x[1]) - elem_trial.gamma_space(
                        x[3])
                    xysqr = np.sum(xy**2, axis=0)
                    return 1. / (4 * np.pi * ts) * np.exp(-xysqr / (4 * ts))

                cube_points = quadpy.cn.ncube_points(elem_test.time_interval,
                                                     elem_test.space_interval,
                                                     elem_trial.time_interval,
                                                     elem_trial.space_interval)
                val_stroud = scheme_stroud.integrate(kernel, cube_points)
                val_mcnamee = scheme_mcnamee.integrate(kernel, cube_points)
                print(i, j,
                      abs(val - val_stroud) / val,
                      abs(val - val_mcnamee) / val)
                assert abs(val - val_mcnamee) / val < 1e-2
                exit
