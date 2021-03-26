import numpy as np
import matplotlib.pyplot as plt


# Simple parametrizations.
def circle(x_hat):
    """ Simple circle parametrization. """
    return np.vstack([np.cos(x_hat), np.sin(x_hat)])


def circle_project(x):
    """ Finds t that minimized |circle(t) - x|_2. """
    return np.atan(x[1] / x[0])


def line(a, b, t_start=0):
    """ Returns parametrization of the segment from a to b. """
    norm = np.linalg.norm(b - a)
    direct = (b - a) / norm

    def fun(x_hat):
        x_hat = x_hat - t_start
        return np.vstack([x_hat * direct[0] + a[0], x_hat * direct[1] + a[1]])

    return fun, norm


def central_derivative(gamma, x, h=1e-5):
    return (gamma(x + h) - gamma(x - h)) / (2 * h)


class PiecewiseParametrization:
    def __init__(self, pw_start, pw_gamma, closed=True):
        self.pw_start = pw_start
        self.pw_gamma = pw_gamma
        self.closed = closed
        self.gamma_length = pw_start[-1]

        assert self.pw_start[0] == 0 and self.gamma_length > 0

        # Assert that the curve is closed.
        if self.closed:
            assert (np.allclose(self.eval(0), self.eval(self.gamma_length)))

        # Estimate derivative and ensure gamma has arc length.
        gamma_deriv = central_derivative(
            self.eval, np.linspace(1e-4, self.gamma_length - 1e-4))
        assert np.allclose(np.linalg.norm(gamma_deriv, axis=0), 1)

    def eval(self, x_hat):
        """ Evaluates this piecewise gamma. """
        # Wrap indices around.
        assert np.all((0 <= x_hat) & (x_hat <= self.gamma_length))
        #x_hat = (x_hat + self.gamma_length) % self.gamma_length
        if len(self.pw_gamma) == 1:
            return self.pw_gamma[0](x_hat)

        # Else use numpy. NOTE: Expensive.
        condlist = []
        pw_eval = []
        for i in range(len(self.pw_gamma)):
            condlist.append((self.pw_start[i] <= x_hat)
                            & (x_hat <= self.pw_start[i + 1]))
            pw_eval.append(self.pw_gamma[i](x_hat))

        return np.select(condlist, pw_eval)

    def plot(self):
        # Evaluate gamma on a set of points and plot.
        pts = self.eval(
            np.linspace(0, self.gamma_length,
                        int(self.gamma_length) * 10 + 1))
        plt.figure()
        plt.plot(pts[0, :], pts[1, :])


class PiecewisePolygon(PiecewiseParametrization):
    def __init__(self, vertices):
        for vertex in vertices:
            assert len(vertex) == 2
        assert np.all(vertices[0] == vertices[-1])

        # Store vertices.
        self.vertices = vertices

        # Create piecewise functions.
        pw_start = [0]
        pw_gamma = []
        for i in range(len(vertices) - 1):
            a, b = vertices[i], vertices[i + 1]
            gamma, length = line(a, b, t_start=pw_start[i])
            pw_start.append(length + pw_start[i])
            pw_gamma.append(gamma)

        # Invoke parent.
        super().__init__(pw_start=pw_start, pw_gamma=pw_gamma)


class PiecewisePolygon(PiecewiseParametrization):
    def __init__(self, vertices, closed=True):
        for vertex in vertices:
            assert len(vertex) == 2
        if closed:
            assert (np.all(vertices[0] == vertices[-1]))

        # Create piecewise functions.
        pw_start = [0]
        pw_gamma = []
        for i in range(len(vertices) - 1):
            a, b = vertices[i], vertices[i + 1]
            gamma, length = line(a, b, t_start=pw_start[i])
            pw_start.append(length + pw_start[i])
            pw_gamma.append(gamma)

        # Invoke parent.
        super().__init__(pw_start=pw_start, pw_gamma=pw_gamma, closed=closed)


class Circle(PiecewiseParametrization):
    def __init__(self):
        # Parametrization is simply the circle.
        pw_start = [0, 2 * np.pi]
        pw_gamma = [circle]

        # Invoke parent.
        super().__init__(pw_start=pw_start, pw_gamma=pw_gamma)

    def project(self, x):
        """ Projects the vector x onto the surface and returns the params. """


class UnitSquare(PiecewisePolygon):
    def __init__(self):
        v0 = np.array([0, 0])
        v1 = np.array([1, 0])
        v2 = np.array([1, 1])
        v3 = np.array([0, 1])
        super().__init__(vertices=[v0, v1, v2, v3, v0])


class LShape(PiecewisePolygon):
    def __init__(self):
        v0 = np.array([0, 0])
        v1 = np.array([0, -1])
        v2 = np.array([1, -1])
        v3 = np.array([1, 1])
        v4 = np.array([-1, 1])
        v5 = np.array([-1, 0])
        super().__init__(vertices=[v0, v1, v2, v3, v4, v5, v0])


class UnitInterval(PiecewisePolygon):
    def __init__(self):
        v0 = np.array([0, 0])
        v1 = np.array([1, 0])
        super().__init__(vertices=[v0, v1], closed=False)


if __name__ == "__main__":
    # Unit Interval
    gamma = UnitInterval()
    gamma.plot()

    # Circle.
    gamma = Circle()
    gamma.plot()

    # Unit square.
    gamma = UnitSquare()
    gamma.plot()

    # L-shape.
    gamma = LShape()
    gamma.plot()

    plt.show()
