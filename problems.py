import numpy as np
from scipy.special import erf, erfc


def smooth_square():
    def u_neumann(t, x_hat):
        """ evaluates the neumann trace along the lateral boundary. """
        return -np.pi * np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * (x_hat % 1))

    def u0(xy):
        return np.sin(np.pi * xy[0]) * np.sin(np.pi * xy[1])

    def M0u0(t, xy):
        x = xy[0]
        y = xy[1]
        pit = np.pi * t
        sqrtt = np.sqrt(t)
        return (((-(1 / 16)) * (erf((x - 2 * 1j * pit) / (2 * sqrtt)) + erf(
            (1 - x + 2 * 1j * pit) /
            (2 * sqrtt)) - np.exp(2 * 1j * x * np.pi) * (erf(
                (1 - x - 2 * 1j * pit) / (2 * sqrtt)) + erf(
                    (x + 2 * 1j * pit) / (2 * sqrtt)))) * (erf(
                        (y - 2 * 1j * pit) / (2 * sqrtt)) + erf(
                            (1 - y + 2 * 1j * pit) /
                            (2 * sqrtt)) - np.exp(2 * 1j * y * np.pi) * (erf(
                                (1 - y - 2 * 1j * pit) / (2 * sqrtt)) + erf(
                                    (y + 2 * 1j * pit) / (2 * sqrtt))))) /
                np.exp(1j * np.pi * (x + y - 2 * 1j * pit))).real

    return {'u-trace': u_neumann, 'u0': u0, 'M0u0': M0u0}


def smooth_pisquare():
    def u_neumann(t, x_hat):
        """ evaluates the neumann trace along the lateral boundary. """
        return -np.exp(-2 * t) * np.sin((x_hat % np.pi))

    def u0(xy):
        return np.sin(xy[0]) * np.sin(xy[1])

    def M0u0(t, xy):
        x = xy[0]
        y = xy[1]
        return ((-(1 / 16)) * np.exp((-1j) * (x + y) - 2 * t) * (-1 + erf(
            (x - 2 * 1j * t) / (2 * np.sqrt(t))) + np.exp(2 * 1j * x) * (-erf(
                (x + 2 * 1j * t) / (2 * np.sqrt(t))) + erf(
                    (x - np.pi + 2 * 1j * t) / (2 * np.sqrt(t)))) + erfc(
                        (x - np.pi - 2 * 1j * t) / (2 * np.sqrt(t)))) *
                (-1 + erf(
                    (y - 2 * 1j * t) / (2 * np.sqrt(t))) + np.exp(2 * 1j * y) *
                 (-erf((y + 2 * 1j * t) / (2 * np.sqrt(t))) + erf(
                     (y - np.pi + 2 * 1j * t) / (2 * np.sqrt(t)))) + erfc(
                         (y - np.pi - 2 * 1j * t) / (2 * np.sqrt(t))))).real

    return {'u-trace': u_neumann, 'u0': u0, 'M0u0': M0u0}


def singular_square():
    def M0u0(t, xy):
        a = xy[0]
        b = xy[1]
        return (1 / 4) * (erf(
            (1 - a) / (2 * np.sqrt(t))) + erf(a / (2 * np.sqrt(t)))) * (erf(
                (1 - b) / (2 * np.sqrt(t))) + erf(b / (2 * np.sqrt(t))))

    return {'u0': lambda xy: 1, 'M0u0': M0u0}


def singular_lshape():
    def M0u0(t, xy):
        a = xy[0]
        b = xy[1]
        return (1 / 4) * ((erf((1 - a) / (2 * np.sqrt(t))) + erf(
            (1 + a) / (2 * np.sqrt(t)))) * (erf(
                (1 - b) /
                (2 * np.sqrt(t))) + erf(b / (2 * np.sqrt(t)))) + (erf(
                    (1 - a) / (2 * np.sqrt(t))) + erf(a / (2 * np.sqrt(t)))) *
                          (-erf(b / (2 * np.sqrt(t))) + erf(
                              (1 + b) / (2 * np.sqrt(t)))))

    return {'u0': lambda xy: 1, 'M0u0': M0u0}


def problem_helper(problem, domain):
    assert problem in ['Smooth', 'Dirichlet', 'Singular']
    assert domain in ['UnitSquare', 'PiSquare', 'LShape']

    result = {}
    if problem == 'Smooth':
        if domain == 'UnitSquare':
            result.update(smooth_square())
        elif domain == 'PiSquare':
            result.update(smooth_pisquare())
        else:
            print('Invalid domain for smooth:', domain)
            assert False
    elif problem == 'Singular':
        if domain == 'UnitSquare':
            result.update(singular_square())
        elif domain == 'LShape':
            result.update(singular_lshape())
        else:
            print('Invalid domain for singular:', domain)
            assert False
    elif problem == 'Dirichlet':
        result['g-linform'] = lambda elems: np.array(
            [elem.h_t * elem.h_x for elem in elems])
        result['g'] = lambda t, xy: 1

    return result
