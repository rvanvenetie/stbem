import numpy as np
import quadrature_rules
from parametrization import circle
from scipy.special import expi, erf, expn, erfc
from quadrature import log_quadrature_scheme, sqrt_quadrature_scheme, sqrtinv_quadrature_scheme, gauss_sqrtinv_quadrature_scheme, gauss_quadrature_scheme, ProductScheme2D, DuffyScheme2D, QuadScheme1D, gauss_x_quadrature_scheme
import quadpy


class Slobodeckij:
    def __init__(self, N_poly):
        self.gauss_leg = gauss_quadrature_scheme(N_poly)
        self.gauss_x = gauss_x_quadrature_scheme(N_poly)
        self.gauss_sqrtinv = gauss_sqrtinv_quadrature_scheme(N_poly)

        gauss_x_leg_2d = ProductScheme2D(self.gauss_x, self.gauss_leg)

        self.semi_1_2_xy = gauss_x_leg_2d.points[0] * gauss_x_leg_2d.points[1]
        self.semi_1_2_weights = 2 * gauss_x_leg_2d.weights

        gauss_sqrtinv_2d = ProductScheme2D(self.gauss_sqrtinv,
                                           self.gauss_sqrtinv)
        x = gauss_sqrtinv_2d.points[0]
        y = gauss_sqrtinv_2d.points[1]
        self.semi_1_4_xy = x * (1 - y)
        self.semi_1_4_weights = 2 * gauss_sqrtinv_2d.weights / y

    def seminorm_h_1_4(self, f, a, b):
        """ Evaluates the H^{1/4}-seminorm of smooth f on the interval [a,b]. """
        h = b - a
        x = a + h * self.gauss_sqrtinv.points
        xy = a + h * self.semi_1_4_xy
        fx = np.repeat(f(x), len(x))
        fxy = np.asarray(f(xy))
        return h**(1 / 2) * np.dot((fx - fxy)**2, self.semi_1_4_weights)

    def seminorm_h_1_2(self, f, a, b, gamma=None):
        """ Evaluates the H^{1/2}-seminorm of smooth f on the interval [a,b].

        Evaluates |f(x) - f(y)|^2 / |x - y|^2.
        """
        h = b - a
        x = a + h * self.gauss_x.points
        xy = a + h * self.semi_1_2_xy
        fx = np.repeat(f(x), len(x))
        fxy = np.asarray(f(xy))
        return h**2 * np.dot(
            ((fx - fxy)**2 /
             (np.repeat(x, len(x)) - xy)**2), self.semi_1_2_weights)


if __name__ == "__main__":

    def f_slo_gauss(f, nu):
        assert 0 < nu < 1

        def integrant(x):
            f_xy = 2 * np.sum((f(x[0]) - f(x[0] * (1 - x[1])))**2, axis=0)
            xy = np.abs(x[1])
            return f_xy / xy

        return integrant

    def f_slobodeckij(f, nu):
        assert 0 < nu < 1

        def integrant(x):
            f_xy = np.sum((f(x[0]) - f(x[1]))**2, axis=0)
            xy = np.abs(x[0] - x[1])**(1 + 2 * nu)
            return f_xy / xy

        return integrant

    quad_scheme = quadpy.c2.get_good_scheme(21)

    for N_poly, N_poly_sqrt in quadrature_rules.SQRT_QUAD_RULES:
        poly_scheme = gauss_quadrature_scheme(N_poly_sqrt * 2 + 1)
        sqrt_scheme = sqrt_quadrature_scheme(N_poly, N_poly_sqrt)
        sqrtinv_scheme = sqrtinv_quadrature_scheme(N_poly, N_poly_sqrt)
        #log_scheme = log_quadrature_scheme(N_poly, N_poly)
        poly_poly = ProductScheme2D(poly_scheme, poly_scheme)
        sqrt_sqrt = ProductScheme2D(sqrt_scheme, sqrt_scheme)
        sqrtinv_sqrtinv = ProductScheme2D(sqrtinv_scheme, sqrtinv_scheme)
        #log_log = ProductScheme2D(log_scheme, log_scheme)
        duff_poly_poly = DuffyScheme2D(poly_poly, symmetric=False)
        duff_sqrt_sqrt = DuffyScheme2D(sqrt_sqrt, symmetric=False)
        #duff_log_log = DuffyScheme2D(log_log, symmetric=False)
        duff_sqrtinv_sqrtinv = DuffyScheme2D(sqrtinv_sqrtinv, symmetric=False)

        for nu in [0.25]:
            #val_exact = 1 / (3 - 5 * nu + 2 * nu**2)
            #val_quad = duff_sqrt_sqrt.integrate(f_slobodeckij(lambda x: x, nu), 0,
            #                                    1, 0, 1)
            val_exact = 0.52290517772141804605
            val_sqrt = duff_sqrt_sqrt.integrate(f_slobodeckij(circle, nu), 0,
                                                1, 0, 1)
            val_sqrtinv = duff_sqrtinv_sqrtinv.integrate(
                f_slobodeckij(circle, nu), 0, 1, 0, 1)

            scheme = gauss_sqrtinv_quadrature_scheme(N_poly_sqrt * 2 + 1)
            scheme_scheme = ProductScheme2D(scheme, scheme)
            duff_scheme_scheme = DuffyScheme2D(scheme_scheme, symmetric=True)
            val_gauss = scheme_scheme.integrate(f_slo_gauss(circle, nu), 0, 1,
                                                0, 1)

            val_exact = 2.7910628585618765042
            val_slo_1_4 = Slobodeckij(N_poly_sqrt * 2 + 1).seminorm_1_4(
                circle, -1, 1)
            print(val_slo_1_4)

            #print('orders={}\tnu={}\n\trel_sqrtinv_sqrtinv={}'.format(
            #    (N_poly, N_poly_sqrt), nu,
            #    abs(val_sqrtinv - val_exact) / val_exact))
            #print('\trel_sqrt={}'.format(
            #    abs(val_sqrt - val_exact) / val_exact))
            #print('\trel_gauss={}'.format(
            #    abs(val_gauss - val_exact) / val_exact))
            print('N={}\trel_slo_1_4={}'.format(
                2 * N_poly_sqrt + 1,
                abs(val_slo_1_4 - val_exact) / val_exact))
            print('')
