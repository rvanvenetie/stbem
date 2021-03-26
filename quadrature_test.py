from quadrature import log_quadrature_scheme, gauss_quadrature_scheme, sqrt_quadrature_scheme, gauss_sqrtinv_quadrature_scheme, ProductScheme2D, DuffyScheme2D, gauss_x_quadrature_scheme
from parametrization import circle, UnitSquare
from scipy.special import expi
import quadrature_rules
import itertools
from pytest import approx
import numpy as np


def test_quadrature():
    for N_poly in [1, 3, 5, 7, 9, 11]:
        scheme = gauss_quadrature_scheme(N_poly)

        # Check that it integrates polynomials exactly.
        for k in range(N_poly + 1):
            f = lambda x: x**k
            assert scheme.integrate(f, 0, 1) == approx(1. / (1 + k))
            assert scheme.integrate(f, 1, 5) == approx(
                (5**(k + 1) - 1.) / (1. + k))


def test_log_quadrature():
    for N_poly, N_poly_log in quadrature_rules.LOG_QUAD_RULES:
        scheme = log_quadrature_scheme(N_poly, N_poly_log)

        # First, check that it integrates polynomials exactly.
        for k in range(N_poly + 1):
            f = lambda x: x**k
            assert scheme.integrate(f, 0, 1) == approx(1. / (1 + k))
            assert scheme.integrate(f, 1, 5) == approx(
                (5**(k + 1) - 1.) / (1. + k))

        # Secondly, check that it integrates log polynomial exactly.
        for k in range(N_poly_log + 1):
            f = lambda x: x**k * np.log(x)
            assert scheme.integrate(f, 0, 1) == approx(-1. / (1 + k)**2)
            assert scheme.integrate(f, 0, 5) == approx(
                (5**(1 + k) * (-1 + np.log(5) + k * np.log(5))) / (1 + k)**2)


def test_sqrt_quadrature():
    for N_poly, N_poly_sqrt in quadrature_rules.SQRT_QUAD_RULES:
        scheme = sqrt_quadrature_scheme(N_poly, N_poly_sqrt)

        # First, check that it integrates polynomials exactly.
        for k in range(N_poly + 1):
            f = lambda x: x**k
            assert scheme.integrate(f, 0, 1) == approx(1. / (1 + k))
            assert scheme.integrate(f, 1, 5) == approx(
                (5**(k + 1) - 1.) / (1. + k))

        # Secondly, check that it integrates log polynomial exactly.
        for k in range(N_poly_sqrt + 1):
            print(N_poly, N_poly_sqrt)
            f = lambda x: x**k * np.sqrt(x)
            assert scheme.integrate(f, 0, 1) == approx(2 / (2 * k + 3))
            assert scheme.integrate(f, 0, 5) == approx(
                (2 * 5**(3 / 2 + k)) / (3 + 2 * k))


def test_gauss_sqrtinv_quadrature():
    for N_poly in range(1, 21, 2):
        scheme = gauss_sqrtinv_quadrature_scheme(N_poly)

        # Check that it integrates weighted polynomials exactly.
        for k in range(N_poly + 1):
            f = lambda x: x**k
            assert scheme.integrate(f, 0, 1) == approx(2 / (1 + 2 * k))
            assert scheme.integrate(f, 0, 5) == approx(
                (2 * 5**(1 + k)) / (1 + 2 * k))


def test_gauss_x_quadrature():
    for N_poly in range(1, 21, 2):
        scheme = gauss_x_quadrature_scheme(N_poly)

        # Check that it integrates weighted polynomials exactly.
        for k in range(N_poly + 1):
            f = lambda x: x**k
            assert scheme.integrate(f, 0, 1) == approx(1 / (2 + k))
            assert scheme.integrate(f, 0, 5) == approx(5**(1 + k) / (2 + k))


def test_product_quadrature():
    for N_poly_x, N_poly_y in itertools.product([1, 3, 5, 7, 9, 11],
                                                [1, 3, 5, 7, 9, 11]):
        scheme_x = gauss_quadrature_scheme(N_poly_x)
        scheme_y = gauss_quadrature_scheme(N_poly_y)
        scheme = ProductScheme2D(scheme_x, scheme_y)

        for i in range(N_poly_x + 1):
            for j in range(N_poly_y + 1):
                f = lambda x: x[0]**i * x[1]**j
                assert scheme.integrate(f, 0, 1, 0,
                                        1) == approx(1. / (1 + i + j + i * j))
                assert scheme.integrate(f, 2, 5, 3, 10) == approx(
                    ((2**(1 + i) - 5**(1 + i)) *
                     (3**(1 + j) - 10**(1 + j))) / ((1 + j) * (1 + i)))


def test_product_log_quadrature():
    for N_log_poly_x, N_poly_y in itertools.product([1, 3, 5, 7, 9],
                                                    [1, 3, 5, 7, 9, 11]):
        scheme_x = log_quadrature_scheme(N_log_poly_x, N_log_poly_x)
        scheme_y = gauss_quadrature_scheme(N_poly_y)
        scheme = ProductScheme2D(scheme_x, scheme_y)

        for i in range(N_log_poly_x + 1):
            for j in range(N_poly_y + 1):
                f = lambda x: x[0]**i * (1 + np.log(x[0])) * x[1]**j
                assert scheme.integrate(f, 0, 1, 0, 1) == approx(
                    i / ((1 + i)**2 * (1 + j)))


def test_duffy_quadrature():
    for symmetric in [True, False]:
        f = lambda x: (x[0] - x[1])**2 * np.log((x[0] - x[1])**2)
        scheme_x = log_quadrature_scheme(3, 3)
        scheme_y = log_quadrature_scheme(2, 2)
        scheme = ProductScheme2D(scheme_x, scheme_y)
        duff_scheme = DuffyScheme2D(scheme, symmetric=symmetric)
        assert duff_scheme.integrate(f, 0, 1, 0,
                                     1) == approx(-0.19444444444444444444,
                                                  rel=1e-15,
                                                  abs=0)


def test_singular_quadrature():
    def f(x):
        diff = (x[0] - x[1])**2
        return np.exp(-diff) + (1 + diff) * expi(-diff)

    # Integrate over [0,1] x [0, 1].
    val_exact = -1.87010542468505982755377882
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=True)
        q_f = duff_scheme.integrate(f, 0, 1, 0, 1)
        q_f_23 = duff_scheme.integrate(f, 2, 3, 2, 3)
        assert q_f == approx(q_f_23)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-15

    # Integrate over [0,1] x [1,2]
    val_exact = -0.24318315547349982560
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=True).mirror_x()
        q_f = duff_scheme.integrate(f, 0, 1, 1, 2)
        q_f_23 = duff_scheme.integrate(f, 1, 2, 2, 3)
        assert q_f == approx(q_f_23)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-14

    # Integrate over [0,2] x [2,3]
    val_exact = -0.24656642836945459944
    rel_error = 1
    for n in range(0, 13, 2):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=False).mirror_x()
        q_f = duff_scheme.integrate(f, 0, 2, 2, 3)
        q_f_24 = duff_scheme.integrate(f, 1, 3, 3, 4)
        assert q_f == approx(q_f_24)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-9

    # Integrate over [0,1] x [1,3]
    rel_error = 1
    for n in range(0, 13, 2):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=False).mirror_x()
        q_f = duff_scheme.integrate(f, 0, 1, 1, 3)
        q_f_24 = duff_scheme.integrate(f, 1, 2, 2, 4)
        assert q_f == approx(q_f_24)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-9

    # Integrate over [0,1] x [2,3]
    val_exact = -0.0033832728959547738380
    rel_error = 1
    for n in range(1, 13):
        log_scheme = log_quadrature_scheme(n, n)
        poly_scheme = gauss_quadrature_scheme(n * 2 + 1)
        prod_scheme = ProductScheme2D(log_scheme.mirror(), log_scheme)
        q_f = prod_scheme.integrate(f, 0, 1, 2, 3)
        q_f_24 = prod_scheme.integrate(f, 1, 2, 3, 4)
        assert q_f == approx(q_f_24)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error

        # Assert that some quadrature schemes work better than others.
        prod_mirror_scheme = ProductScheme2D(log_scheme, log_scheme)
        assert abs(prod_mirror_scheme.integrate(f, 0, 1, 2, 3) -
                   val_exact) > abs(q_f - val_exact)

    assert rel_error < 1e-15


def test_singular_quadrature_circle():
    def f(x):
        #return np.log(x[0] + 4 * x[1])
        diff = circle(x[0]) - circle(x[1])
        normsqr = np.sum(diff**2, axis=0)
        return np.exp(-normsqr) + (1 + normsqr) * expi(-normsqr)

    # Integrate over [0,1] x [0,1]
    val_exact = -1.87629901756723965199622242456
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=True)
        q_f = duff_scheme.integrate(f, 0, 1, 0, 1)
        q_f_23 = duff_scheme.integrate(f, 1, 2, 1, 2)
        assert q_f == approx(q_f_23)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-12

    # Integrate over [0,1] x [2pi - 1, 2pi]
    val_exact = -0.253663490380649986736253376122
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=False).mirror_y()
        q_f = duff_scheme.integrate(f, 0, 1, 2 * np.pi - 1, 2 * np.pi)
        q_f_23 = duff_scheme.integrate(f, 1, 2, 2 * np.pi, 2 * np.pi + 1)
        assert q_f == approx(q_f_23)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-12

    # Integrate over [0,1] x [1, 2]
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=False).mirror_x()
        q_f = duff_scheme.integrate(f, 0, 1, 1, 2)
        q_f_23 = duff_scheme.integrate(f, 1, 2, 2, 3)
        assert q_f == approx(q_f_23)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-12


def test_singular_quadrature_corner():
    gamma = UnitSquare()

    def f(x):
        diff = gamma.eval(x[0]) - gamma.eval(x[1])
        normsqr = np.sum(diff**2, axis=0)
        return np.exp(-normsqr) + (1 + normsqr) * expi(-normsqr)

    # Integrate over [0,1] x [0,1]
    val_exact = -1.87010542468505982755377882
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=True)
        q_f = duff_scheme.integrate(f, 0, 1, 0, 1)
        q_f_34 = duff_scheme.integrate(f, 3, 4, 3, 4)
        assert q_f == approx(q_f_34)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-15

    # Integrate over [0,1] x [1,2]
    val_exact = -0.3718093426679699430449066
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=False).mirror_x()
        q_f = duff_scheme.integrate(f, 0, 1, 1, 2)
        assert q_f == approx(duff_scheme.integrate(f, 1, 2, 2, 3))
        assert q_f == approx(duff_scheme.integrate(f, 2, 3, 3, 4))
        assert q_f == approx(duff_scheme.integrate(f, 3, 4, 0, 1))
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-12

    # Integrate over [1,2] x [0,1]
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffyScheme2D(ProductScheme2D(scheme, scheme),
                                    symmetric=False).mirror_y()
        q_f = duff_scheme.integrate(f, 1, 2, 0, 1)
        assert q_f == approx(duff_scheme.integrate(f, 2, 3, 1, 2))
        assert q_f == approx(duff_scheme.integrate(f, 3, 4, 2, 3))
        assert q_f == approx(duff_scheme.integrate(f, 0, 1, 3, 4))
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-15
        rel_error = new_rel_error
    assert rel_error < 1e-12
