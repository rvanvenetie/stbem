import numpy as np
import quadrature_rules
from quadrature_rules import log_quadrature_rule, log_log_quadrature_rule, sqrt_quadrature_rule, sqrtinv_quadrature_rule, gauss_sqrtinv_quadrature_rule, gauss_x_quadrature_rule, gauss_log_quadrature_rule


def gauss_quadrature_scheme(N_poly):
    """ Returns quadrature rule that is exact on 0^1 for
    p(x) for deg(p) <= N_poly.  """
    assert (N_poly % 2 != 0)
    n = (N_poly + 1) // 2
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return QuadScheme1D(0.5 * (nodes + 1.0), 0.5 * weights)


def gauss_sqrtinv_quadrature_scheme(N_poly):
    """ Returns quadrature rule that is exact on 0^1 with weight 1/sqrt(x)
    for p(x) for deg(p) <= N_poly.  """
    assert (N_poly % 2 != 0)
    N = (N_poly + 1) // 2
    nodes, weights = gauss_sqrtinv_quadrature_rule(N)
    return QuadScheme1D(nodes, weights)


def gauss_x_quadrature_scheme(N_poly):
    """ Returns quadrature rule that is exact on 0^1 with weight x
    for p(x) for deg(p) <= N_poly.  """
    N = (N_poly + 1) // 2
    nodes, weights = gauss_x_quadrature_rule(N)
    return QuadScheme1D(nodes, weights)


def gauss_log_quadrature_scheme(N_poly):
    """ Returns quadrature rule that is exact on 0^1 with weight x
    for p(x) for deg(p) <= N_poly.  """
    N = (N_poly + 1) // 2
    nodes, weights = gauss_log_quadrature_rule(N)
    return QuadScheme1D(nodes, weights)


def log_quadrature_scheme(N_poly, N_poly_log):
    """ Returns quadrature rule that is exact on 0^1 for
    p(x) + q(x)log(x) for deg(p) <= N_poly and deg(q) <= N_log.
    """
    nodes, weights = log_quadrature_rule(N_poly, N_poly_log)
    return QuadScheme1D(nodes, weights)


def log_log_quadrature_scheme(N_poly, N_poly_log):
    """ Returns quadrature rule that is exact on 0^1 for
    p(x) + q(x)log(x) + k(x)log(1-x) for deg(p) <= N_poly, deg(k) <= deg(q) <= N_log.
    """
    nodes, weights = log_log_quadrature_rule(N_poly, N_poly_log)
    return QuadScheme1D(nodes, weights)


def sqrt_quadrature_scheme(N_poly, N_poly_log):
    """ Returns quadrature rule that is exact on 0^1 for
    p(x) + q(x)sqrt(x) for deg(p) <= N_poly and deg(q) <= N_poly_sqrt.
    """
    nodes, weights = sqrt_quadrature_rule(N_poly, N_poly_log)
    return QuadScheme1D(nodes, weights)


def sqrtinv_quadrature_scheme(N_poly, N_poly_log):
    """ Returns quadrature rule that is exact on 0^1 for
    p(x) + q(x)sqrt(x) for deg(p) <= N_poly and deg(q) <= N_poly_sqrt.
    """
    nodes, weights = sqrtinv_quadrature_rule(N_poly, N_poly_log)
    return QuadScheme1D(nodes, weights)


class QuadScheme1D:
    def __init__(self, points, weights):
        self.points = np.array(points)
        self.weights = np.array(weights)
        self._mirror = None

    def mirror(self):
        if self._mirror is None:
            self._mirror = QuadScheme1D(1 - self.points, self.weights)
        return self._mirror

    def integrate(self, f, a: float, b: float) -> float:
        if a == b: return 0
        assert b - a > 1e-5
        fx = (b - a) * np.asarray(f(a + (b - a) * self.points))
        return np.dot(fx, self.weights)


class QuadScheme2D:
    def __init__(self, points, weights):
        self.points = np.array(points)
        self.weights = np.array(weights)
        self._mirror_x = None
        self._mirror_y = None

    def mirror_x(self):
        if self._mirror_x is None:
            self._mirror_x = QuadScheme2D([1 - self.points[0], self.points[1]],
                                          self.weights)

        return self._mirror_x

    def mirror_y(self):
        if self._mirror_y is None:
            self._mirror_y = QuadScheme2D([self.points[0], 1 - self.points[1]],
                                          self.weights)
        return self._mirror_y

    def integrate(self, f, a: float, b: float, c: float, d: float) -> float:
        assert b - a > 1e-7 and d - c > 1e-7
        x = np.array(
            [a + (b - a) * self.points[0], c + (d - c) * self.points[1]])
        fx = np.asarray(f(x))
        return (d - c) * (b - a) * np.dot(fx, self.weights)


class ProductScheme2D(QuadScheme2D):
    def __init__(self, scheme_x, scheme_y=None):
        if scheme_y is None: scheme_y = scheme_x
        assert isinstance(scheme_x, QuadScheme1D) and isinstance(
            scheme_y, QuadScheme1D)
        points = np.array([
            np.repeat(scheme_x.points, len(scheme_y.points)),
            np.tile(scheme_y.points, len(scheme_x.points))
        ])
        weights = np.kron(scheme_x.weights, scheme_y.weights)
        super().__init__(points=points, weights=weights)


class QuadpyScheme2D(QuadScheme2D):
    def __init__(self, quad_scheme):
        super().__init__(points=(quad_scheme.points + 1) * 0.5,
                         weights=quad_scheme.weights)


class DuffyScheme2D(QuadScheme2D):
    def __init__(self, scheme2d, symmetric):
        assert isinstance(scheme2d, QuadScheme2D)

        x = scheme2d.points[0]
        y = 1 - scheme2d.points[1]
        xy = x * y
        weights = scheme2d.weights * x
        if symmetric:
            points = [x, xy]
            weights = weights * 2
        else:
            points = [np.hstack([x, xy]), np.hstack([xy, x])]
            weights = np.hstack([weights, weights])

        super().__init__(points=points, weights=weights)


class QuadScheme3D:
    def __init__(self, points, weights):
        self.points = np.array(points)
        self.weights = np.array(weights)
        self._mirror_x = None
        self._mirror_y = None
        self._mirror_z = None

    def integrate(self, f, a, b, c, d, k, l):
        x = np.array([
            a + (b - a) * self.points[0], c + (d - c) * self.points[1],
            k + (l - k) * self.points[2]
        ])
        fx = np.asarray(f(x))
        return (d - c) * (b - a) * (l - k) * np.dot(fx, self.weights)

    def mirror_x(self):
        if self._mirror_x is None:
            self._mirror_x = QuadScheme3D(
                [1 - self.points[0], self.points[1], self.points[2]],
                self.weights)

        return self._mirror_x

    def mirror_y(self):
        if self._mirror_y is None:
            self._mirror_y = QuadScheme3D(
                [self.points[0], 1 - self.points[1], self.points[2]],
                self.weights)
        return self._mirror_y

    def mirror_z(self):
        if self._mirror_z is None:
            self._mirror_z = QuadScheme3D(
                [self.points[0], self.points[1], 1 - self.points[2]],
                self.weights)
        return self._mirror_z


class ProductScheme3D(QuadScheme3D):
    def __init__(self, scheme_x):
        scheme_y = scheme_z = scheme_x
        assert isinstance(scheme_x, QuadScheme1D) and isinstance(
            scheme_y, QuadScheme1D)
        points_xy = np.array([
            np.repeat(scheme_x.points, len(scheme_y.points)),
            np.tile(scheme_y.points, len(scheme_x.points))
        ])
        points = np.vstack([
            np.repeat(points_xy, len(scheme_z.points), axis=1),
            np.tile(scheme_z.points, points_xy.shape[1])
        ])
        weights = np.kron(np.kron(scheme_x.weights, scheme_y.weights),
                          scheme_z.weights)
        super().__init__(points=points, weights=weights)


class DuffySchemeIdentical3D(QuadScheme3D):
    """ Duffy scheme for unit cube having singularies of the form 
        log[(x − y)^2 + z^2]. """
    def __init__(self, scheme3d, symmetric_xy):
        assert isinstance(scheme3d, QuadScheme3D)

        x = scheme3d.points[0]
        y = scheme3d.points[1]
        z = scheme3d.points[2]

        T1 = [x, x * (1 - y), x * y * z]
        T2 = [x * (1 - y + y * z), x * y * z, x]
        T3 = [x, x * (1 - y * z), x * y]

        T4 = [x * (1 - y), x, x * y * z]
        T5 = [x * y * z, x * (1 - y + y * z), x]
        T6 = [x * (1 - y * z), x, x * y]

        if symmetric_xy:
            points = np.hstack([T1, T2, T3])
            weights = 2 * np.tile(scheme3d.weights * x**2 * y, 3)
        else:
            points = np.hstack([T1, T2, T3, T4, T5, T6])
            weights = np.tile(scheme3d.weights * x**2 * y, 6)

        super().__init__(points=points, weights=weights)


class DuffySchemeTouch3D(QuadScheme3D):
    """ Duffy scheme for unit cube having singularies of the form 
        log[(x + y)^2 + z^2] or log[(x^2 + (y+z)^2. """
    def __init__(self, scheme3d):
        assert isinstance(scheme3d, QuadScheme3D)

        x = scheme3d.points[0]
        y = scheme3d.points[1]
        z = scheme3d.points[2]

        P1 = [x * y, y, z * y]
        P2 = [y, x * y, z * y]
        P3 = [x * y, z * y, y]

        points = np.hstack([P1, P2, P3])
        weights = np.tile(scheme3d.weights * y**2, 3)
        super().__init__(points=points, weights=weights)
