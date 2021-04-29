import numpy as np
from parametrization import line
import quadrature_rules
from parametrization import circle
from scipy.special import expi, erf, expn, erfc
from quadrature import log_quadrature_scheme, sqrt_quadrature_scheme, sqrtinv_quadrature_scheme, gauss_sqrtinv_quadrature_scheme, gauss_quadrature_scheme, ProductScheme2D, DuffyScheme2D, QuadScheme1D, gauss_x_quadrature_scheme, QuadScheme2D
import quadpy


class Slobodeckij:
    def __init__(self, N_poly):
        # Scheme for H^{1/4}.
        self.gauss_sqrtinv = gauss_sqrtinv_quadrature_scheme(N_poly)
        gauss_sqrtinv_2d = ProductScheme2D(self.gauss_sqrtinv,
                                           self.gauss_sqrtinv)
        x = gauss_sqrtinv_2d.points[0]
        y = gauss_sqrtinv_2d.points[1]
        self.semi_1_4_xy = x * (1 - y)
        self.semi_1_4_weights = 2 * gauss_sqrtinv_2d.weights / y

        # Scheme for H^{1/2}
        self.gauss_leg = gauss_quadrature_scheme(N_poly)
        self.gauss_x = gauss_x_quadrature_scheme(N_poly)
        gauss_x_leg_2d = ProductScheme2D(self.gauss_x, self.gauss_leg)

        # Identical gamma's use this.
        x = gauss_x_leg_2d.points[0]
        y = gauss_x_leg_2d.points[1]
        self.semi_1_2_xy = x * y
        self.semi_1_2_weights = gauss_x_leg_2d.weights

        # Touching elements, will use this.
        points = [np.hstack([1 - x, 1 - x * y]), np.hstack([x * y, x])]
        weights = np.hstack([gauss_x_leg_2d.weights, gauss_x_leg_2d.weights])
        self.semi_1_2_pw = QuadScheme2D(points, weights)

    def seminorm_h_1_4(self, f, a, b):
        """ Evaluates the squared H^{1/4}-seminorm of smooth f on the interval [a,b]. """
        h = b - a
        x = a + h * self.gauss_sqrtinv.points
        xy = a + h * self.semi_1_4_xy
        fx = np.repeat(f(x), len(x))
        fxy = np.asarray(f(xy))
        return h**(1 / 2) * np.dot((fx - fxy)**2, self.semi_1_4_weights)

    def seminorm_h_1_2(self, f, a, b, gamma=None):
        """ Evaluates the squared H^{1/2}-seminorm of smooth f on the interval [a,b].

        If gamma is none, evaluates: |f(x) - f(y)|^2 / |x - y|^2.
        Else: |f(x, gamma(x)) - f(y, gamma(y))|^2 / |gamma(x) - gamma(y)|^2.
        """
        h = b - a
        x_hat = a + h * self.gauss_x.points
        xy_hat = a + h * self.semi_1_2_xy
        if gamma is None:
            x = x_hat
            xy = xy_hat
            xy_sqr = (np.repeat(x, len(x)) - xy)**2
            fx = np.repeat(f(x), len(x))
            fxy = np.asarray(f(xy))
        else:
            x = gamma(x_hat)
            xy = gamma(xy_hat)
            xy_sqr = np.sum((np.repeat(x, len(x_hat), axis=1) - xy)**2, axis=0)
            fx = np.repeat(f(x_hat, x), len(x_hat))
            fxy = np.asarray(f(xy_hat, xy))

        return 2 * h**2 * np.dot((fx - fxy)**2 / xy_sqr, self.semi_1_2_weights)

    def seminorm_h_1_2_pw(self, f, a_1, b_1, gamma_1, a_2, b_2, gamma_2):
        """ Evaluates the squared H^{1/2}-seminorm of smooth f over the
            two elemens induced by gamma_1 and gamma_2.  """
        assert gamma_1 is not gamma_2
        assert np.all(gamma_1(b_1) == gamma_2(a_2))

        # Evaluate seminorm of f on [a_1, b_1].
        result = self.seminorm_h_1_2(f, a_1, b_1, gamma_1)

        # Evaluate seminorm of f on [a_2, b_2].
        result += self.seminorm_h_1_2(f, a_2, b_2, gamma_2)

        def slo(xy):
            x_hat = xy[0]
            y_hat = xy[1]
            x = gamma_1(x_hat)
            y = gamma_2(y_hat)
            xy_sqr = np.sum((x - y)**2, axis=0)
            return (f(x_hat, x) - f(y_hat, y))**2 / xy_sqr

        # Evaluate seminorm of f on [a_1, b_1] cross [a_2, b_2]
        result += 2 * self.semi_1_2_pw.integrate(slo, a_1, b_1, a_2, b_2)
        return result


if __name__ == "__main__":
    gamma_1 = line(np.array([0, 0]), np.array([1, 0]), x_start=0)[0]
    gamma_2 = line(np.array([1, 0]), np.array([1, 1]), x_start=1)[0]
    gamma = lambda x_hat: np.select([x_hat < 1, x_hat >= 1], [
        gamma_1(x_hat), gamma_2(x_hat)
    ])
    val_exact = 3.8623935284968760922
    rel_error = 1
    for N in range(1, 21, 2):

        def f(x_hat, x):
            return x[0] * np.cos(np.pi * x[1])

        q_f_1 = Slobodeckij(N).seminorm_h_1_2(f, 0, 1, gamma_1)
        q_f_2 = Slobodeckij(N).seminorm_h_1_2(f, 1, 2, gamma_2)

        #print((q_f_2 - 5.9182599061341682774) / 5.9182599061341682774)


        def slo(xy):
            x = gamma_1(xy[0])
            y = gamma_2(xy[1])
            xy_sqr = np.sum((x - y)**2, axis=0)
            return (f(xy[0], x) - f(xy[1], y))**2 / xy_sqr

        val_exact = 1.2524972308160824769
        val_approx = DuffyScheme2D(ProductScheme2D(gauss_quadrature_scheme(N)),
                                   symmetric=False).mirror_x().integrate(
                                       slo, 0, 1, 1, 2)
        print(abs(val_exact - val_approx) / val_exact)

        gauss_leg = gauss_quadrature_scheme(N)
        gauss_x = gauss_x_quadrature_scheme(N)
        gauss_x_leg_2d = ProductScheme2D(gauss_x, gauss_leg)
        x = gauss_x_leg_2d.points[0]
        xy = gauss_x_leg_2d.points[0] * gauss_x_leg_2d.points[1]
        points = [np.hstack([1 - x, 1 - xy]), np.hstack([xy, x])]
        weights = np.hstack([gauss_x_leg_2d.weights, gauss_x_leg_2d.weights])
        scheme_2d = QuadScheme2D(points, weights)
        q_f = scheme_2d.integrate(slo, 0, 1, 1, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print(N, 'split', new_rel_error)
        #val_exact = 1.6931471805597419447
        #new_rel_error = abs((q_f - val_exact) / val_exact)
        #print(N, 'split', new_rel_error)
        ###assert new_rel_error < rel_error or new_rel_error < 1e-13
        ##rel_error = new_rel_error
