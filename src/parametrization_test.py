import numpy as np

from .parametrization import Circle, LShape, UnitSquare


def test_pw_param():
    for gamma in [Circle(), UnitSquare(), LShape()]:
        points = gamma.pw_start
        # Make sure that interior points map one-to-one.
        for i in range(1, len(gamma.pw_gamma)):
            assert np.allclose(gamma.pw_gamma[i](points[i]),
                               gamma.pw_gamma[i - 1](points[i]))

        # Make sure that the endpoint wraps around.
        assert np.allclose(gamma.pw_gamma[0](points[0]),
                           gamma.pw_gamma[-1](points[-1]))
