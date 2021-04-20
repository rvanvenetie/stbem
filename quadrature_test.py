from quadrature import log_quadrature_scheme, gauss_quadrature_scheme, sqrt_quadrature_scheme, gauss_sqrtinv_quadrature_scheme, ProductScheme2D, DuffyScheme2D, gauss_x_quadrature_scheme, DuffySchemeIdentical3D, ProductScheme3D, DuffySchemeTouch3D
from parametrization import circle, UnitSquare
from scipy.special import expi, exp1
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
        print(N_poly, N_poly_log)
        scheme = log_quadrature_scheme(N_poly, N_poly_log)

        # First, check that it integrates polynomials exactly.
        for k in range(N_poly + 1):
            print(k)
            f = lambda x: x**k
            assert scheme.integrate(f, 0, 1) == approx(1. / (1 + k))
            assert scheme.integrate(f, 1, 5) == approx(
                (5**(k + 1) - 1.) / (1. + k))

        # Secondly, check that it integrates log polynomial exactly.
        for k in range(N_poly_log + 1):
            f = lambda x: x**k * np.log(x)
            assert scheme.integrate(f, 0, 1) == approx(-1. / (1 + k)**2)
            #assert scheme.integrate(f, 0, 3) == approx(
            #    (3**(1 + k) * (-1 + np.log(3) + k * np.log(3))) / (1 + k)**2)


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


def test_singular_duffy_3d_id():
    b = 0.25
    G_time = lambda xy: 1. / (4 * np.pi) * exp1(xy / (4 * b))

    # Test with u0 = 1.
    u0 = lambda y: np.ones(y.shape[1])

    h = 1
    f = lambda xyz: u0(xyz) * G_time(h**2 * ((xyz[0] - xyz[1])**2 + xyz[2]**2))
    val_exact = 0.075961144077555044645
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffySchemeIdentical3D(ProductScheme3D(scheme),
                                             symmetric_xy=False)
        q_f = duff_scheme.integrate(f, 0, 1, 0, 1, 0, 1)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-12
        rel_error = new_rel_error
    assert rel_error < 1e-12

    h = 0.25
    f = lambda xyz: u0(xyz) * G_time(h**2 * ((xyz[0] - xyz[1])**2 + xyz[2]**2))
    val_exact = 0.0041485131062119699490
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffySchemeIdentical3D(ProductScheme3D(scheme),
                                             symmetric_xy=True)
        q_f = h**3 * duff_scheme.integrate(f, 0, 1, 0, 1, 0, 1)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-12
        rel_error = new_rel_error

    assert rel_error < 1e-12

    # Test with u0 = sin(x) * y
    u0 = lambda xy: np.sin(xy[0]) * xy[1]

    h = 0.25
    vertices = [(1, 1), (1 - h, 1), (1, 1 - h), (1 - h, 1 - h)]
    v0, v1, v2 = [np.array(vtx).reshape(-1, 1) for vtx in vertices][0:3]

    # Make parametrization of the element Q.
    gamma_Q = lambda x, z: v0 + (v1 - v0) * x + (v2 - v0) * z
    gamma_K = lambda y: v0 + (v1 - v0) * y
    assert np.all(gamma_K(0) == v0)
    assert np.all(gamma_K(1) == v1)
    assert np.all(gamma_Q(0.5, 0) == gamma_K(0.5))
    assert not np.all(gamma_Q(0.5, 1) == gamma_K(0.5))

    #def f(xyz):
    #    x, y, z = xyz
    #    return u0(gamma_Q(x, z)) * G_time(
    #        np.sum((gamma_Q(x, z) - gamma_Q(y, 0))**2, axis=0))
    f = lambda xyz: u0(gamma_Q(xyz[0], xyz[2])) * G_time(h**2 * (
        (xyz[0] - xyz[1])**2 + xyz[2]**2))

    val_exact = 0.0028374980621858479108
    rel_error = 1
    for n in range(4, 12):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffySchemeIdentical3D(ProductScheme3D(scheme),
                                             symmetric_xy=False)
        fx = f(duff_scheme.points)
        q_f = h**3 * np.dot(fx, duff_scheme.weights)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-12
        rel_error = new_rel_error

    assert rel_error < 1e-12


def test_singular_duffy_3d_touch():
    # Test the touch quadrature rule.
    f = lambda xyz: np.log((xyz[0] + xyz[1])**2 + xyz[2]**2)
    val_exact = 0.1781673429530223041202893120098701898314
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffySchemeTouch3D(ProductScheme3D(scheme))
        q_f = duff_scheme.integrate(f, 0, 1, 0, 1, 0, 1)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-12
        rel_error = new_rel_error

    assert rel_error < 1e-12

    # Test the touch quadrature rule.
    f = lambda xyz: np.log(xyz[0]**2 + (xyz[1] + xyz[2])**2)
    val_exact = 0.1781673429530223041202893120098701898314
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffySchemeTouch3D(ProductScheme3D(scheme))
        q_f = duff_scheme.integrate(f, 0, 1, 0, 1, 0, 1)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-12
        rel_error = new_rel_error

    assert rel_error < 1e-12

    f = lambda xyz: np.log((xyz[0] + 2 * xyz[1])**2 + xyz[2]**2)
    val_exact = 0.80392693298465673176
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffySchemeTouch3D(ProductScheme3D(scheme))
        q_f = duff_scheme.integrate(f, 0, 1, 0, 1, 0, 1)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-12
        rel_error = new_rel_error

    assert rel_error < 1e-12

    f = lambda xyz: np.log((xyz[0] + 0.5 * xyz[1])**2 + xyz[2]**2)
    val_exact = -0.22999882492711279068
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffySchemeTouch3D(ProductScheme3D(scheme))
        q_f = duff_scheme.integrate(f, 0, 1, 0, 1, 0, 1)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-11
        rel_error = new_rel_error

    assert rel_error < 1e-11

    b = 0.25
    G_time = lambda xy: 1. / (4 * np.pi) * exp1(xy / (4 * b))

    v0 = np.array([1., 1.]).reshape(-1, 1)
    v1 = np.array([2, 1.]).reshape(-1, 1)
    v2 = np.array([0, 1]).reshape(-1, 1)
    v3 = np.array([1, 0]).reshape(-1, 1)
    h_elem = 1
    h = 1

    # Create parametrizations of Q and K.
    gamma_K = lambda y: v0 + (v1 - v0) * y
    gamma_Q = lambda x, z: v0 + (v2 - v0) * x + (v3 - v0) * z
    assert np.all(gamma_Q(0, 0) == gamma_K(0))

    #u0 = lambda xy: np.ones(xy.shape[1])
    #f = lambda xyz: u0(gamma_Q(xyz[0], xyz[2])) * G_time(
    #    np.sum((gamma_Q(xyz[0], xyz[2]) - gamma_K(xyz[1]))**2, axis=0))
    f = lambda xyz: 1 / (4 * np.pi) * exp1((xyz[0] + xyz[1])**2 + xyz[2]**2)

    #val_exact = 0.0201681640240317535810058111329
    #val_exact = 0.02016816651934319
    #val_exact = 0.020168166580447650
    val_exact = 0.020168166583416268
    rel_error = 1
    for n in range(2, 13):
        scheme = log_quadrature_scheme(n, n)
        duff_scheme = DuffySchemeTouch3D(ProductScheme3D(scheme))
        fx = f(duff_scheme.points)
        q_f = h_elem**2 * h * np.dot(fx, duff_scheme.weights)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(n, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-11
        rel_error = new_rel_error

    assert rel_error < 1e-11
