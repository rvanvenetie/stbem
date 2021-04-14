import numpy as np
import quadrature_rules
from parametrization import circle
from quadrature_rules import log_quadrature_rule, sqrt_quadrature_rule, sqrtinv_quadrature_rule, gauss_sqrtinv_quadrature_rule, gauss_x_quadrature_rule, gauss_log_quadrature_rule


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

    def integrate(self, f, a, b):
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

    def integrate(self, f, a, b, c, d):
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.special import expi

    for N_poly, N_poly_sqrt in quadrature_rules.LOG_QUAD_RULES:
        #for N in [7, 15, 31]:
        #scheme = gauss_log_quadrature_scheme(N * 2 + 1)
        scheme = log_quadrature_scheme(N_poly, N_poly_sqrt)
        strs = []
        for i, x in enumerate(scheme.points):
            strs.append('{{x[{}], {:.10f}}}'.format(i, x))
        for i, w in enumerate(scheme.weights):
            strs.append('{{w[{}], {:.10f}}}'.format(i, w))
        print('QuadSystem[{}, {}, {}]'.format(N_poly, N_poly_sqrt,
                                              '{' + ','.join(strs) + '}'))
        #print('QuadSystem[{}, {}]'.format(N, '{' + ','.join(strs) + '}'))
    #asdf

    def f(x):
        #return np.log(x[0] + 4 * x[1])
        diff = (x[0] - x[1])**2
        return np.exp(-diff) + (1 + diff) * expi(-diff)

    def f_circle(x):
        #return np.log(x[0] + 4 * x[1])
        diff = circle(x[0]) - circle(x[1])
        normsqr = np.sum(diff**2, axis=0)
        return np.exp(-normsqr) + (1 + normsqr) * expi(-normsqr)

    def g_circle(x):
        diff = np.array([[1], [0]]) - circle(x)
        normsqr = np.sum(diff**2, axis=0)
        return expi(-normsqr)

    for N_poly, N_poly_log in quadrature_rules.LOG_QUAD_RULES:
        log_scheme = log_quadrature_scheme(N_poly, N_poly_log)
        poly_scheme = gauss_quadrature_scheme(N_poly + N_poly_log + 1)
        log_log_scheme = ProductScheme2D(log_scheme, log_scheme)
        log_poly_scheme = ProductScheme2D(log_scheme, poly_scheme)
        poly_poly_scheme = ProductScheme2D(poly_scheme, poly_scheme)
        duff_log_log = DuffyScheme2D(log_log_scheme, symmetric=False)
        val_exact = -1.870105424685059827553778824442236893963491384152265731083999313041264853732025953955890197184038459
        val_exact = -0.02123375230858629179307243394184385025474123470002899426353062635645657161941159024150479218406435158
        val_exact = -1.870105424685059827553778824442236893963491384152265731083999313041264853732025953955890197184038459
        #val_exact = -0.24656642836945459944
        #val_exact = -0.24656642836945459944
        #val_exact = -0.22741127776021876233
        #val_exact = -0.11370563888010938117
        #val_exact = 0.27873046894330149622
        #val_exact = 1.4943409354599133715
        #val_exact = 0.75690475411678243296
        #val_exact = -1.8763828202796171356
        #val_exact = -1.8762990175672499779
        #q_f = duff_log_log.integrate(f, 0, 1, 0, 1)
        #print(q_f)
        #q_f = duff_log_log.integrate(f_circle, 0, 1, 0, 1)
        #val_exact = -0.25366349038065227355
        #q_f = duff_log_log.mirror_y().integrate(f_circle, 0, 1, 2 * np.pi - 1,
        #                                        2 * np.pi)
        #val_exact = -0.0033832728959547738380
        #q_f = log_log_scheme.integrate(f, 0, 1, 2, 3)
        #val_exact = -0.15650830639146083841
        #q_f = log_log_scheme.mirror_x().integrate(f, 0, 1, 1.1, 2)
        #q_f = log_log_scheme.mirror_x().integrate(f, 0, 1, 2, 3)
        val_exact = -0.24656642836945459944
        q_f = duff_log_log.mirror_x().integrate(f, 0, 2, 2, 3)
        #q_f = ProductScheme2D(scheme.mirror(), scheme).integrate(f, 0, 1, 0, 1)

        #val_exact = -1.7993820695380366279
        #q_f = log_scheme.integrate(g_circle, 0, np.pi / 2)

        print(N_poly, N_poly_log, abs(q_f - val_exact) / val_exact)
#        for i, x in enumerate(log_scheme.points):
#            print('{{x[{}], {:.10f}}}'.format(i, x), end=',')
#        for i, w in enumerate(log_scheme.weights):
#            print('{{w[{}], {:.10f}}}'.format(i, w), end=',')
