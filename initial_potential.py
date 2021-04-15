from mesh import Mesh, MeshParametrized
import math
from initial_mesh import UnitSquareBoundaryRefined
import quadrature_rules
from scipy.special import exp1
from quadrature import gauss_quadrature_scheme, log_quadrature_scheme, sqrt_quadrature_scheme, ProductScheme2D, ProductScheme3D, DuffySchemeIdentical3D, DuffySchemeTouch3D
from parametrization import Circle, UnitSquare, LShape
import numpy as np


def time_integrated_kernel(a, b):
    """ Returns heat kernel G(t,x) integrated over t in [a,b]. """
    if a == 0:
        return lambda xy: 1. / (4 * np.pi) * exp1(xy / (4 * b))
    else:
        return lambda xy: 1. / (4 * np.pi) * (exp1(xy /
                                                   (4 * b)) - exp1(xy /
                                                                   (4 * a)))


class InitialPotential:
    def __init__(self, bdr_mesh, u0, initial_mesh=None, quad_order=19):
        self.bdr_mesh = bdr_mesh
        self.initial_mesh = initial_mesh
        self.space_integrator = bdr_mesh.gamma_space.integrator(quad_order)
        self.u0 = u0
        self.gauss_scheme = gauss_quadrature_scheme(quad_order)
        self.log_scheme = log_quadrature_scheme(min(quad_order, 12),
                                                min(quad_order, 12))
        self.gauss_2d = ProductScheme2D(self.gauss_scheme)
        self.duff_3d_id = DuffySchemeIdentical3D(ProductScheme3D(
            self.log_scheme),
                                                 symmetric_xy=False)
        self.duff_3d_touch = DuffySchemeTouch3D(
            ProductScheme3D(self.log_scheme))

    def linform(self, elem_trial):
        """ Evaluates <M_0 u_0, 1_trial>. """
        assert self.initial_mesh is not None

        # Integrate the heat kernel over time.
        a, b = elem_trial.time_interval
        G_time = time_integrated_kernel(a, b)

        # Calculate space bdr in Omega.
        c, d = elem_trial.space_interval

        # Create mesh of Omega adapted to the given v0, v1.
        initial_mesh = self.initial_mesh(elem_trial.gamma_space(c),
                                         elem_trial.gamma_space(d))

        # Find vertices associated to v0 and v1.
        v0 = initial_mesh.vertex_from_coords(elem_trial.gamma_space(c))
        v1 = initial_mesh.vertex_from_coords(elem_trial.gamma_space(d))
        assert v0 is not None and v1 is not None

        n0 = v0.xy_np
        n1 = v1.xy_np
        assert d - c == np.linalg.norm(n0 - n1)

        id_bdr = 0
        touch_bdr = 0
        result = 0
        ips = []
        for elem in initial_mesh.leaf_elements:
            # Check whether this element has an identical bdr,
            if v0 in elem.vertices and v1 in elem.vertices:
                id_bdr += 1
                h = d - c
                assert elem.diam == h

                # Element has an edge v0 <-> v1. Let v2 be the unique vertex
                # having an edge v0 <-> v2.
                tmp = [v for v in elem.connected_to_vertex(v0) if v is not v1]
                assert len(tmp) == 1
                n2 = tmp[0].xy_np

                # Create parametrizations of Q.
                gamma_Q = lambda x, z: n0 + (n1 - n0) * x + (n2 - n0) * z

                f = lambda xyz: self.u0(gamma_Q(xyz[0], xyz[2])) * G_time(
                    h**2 * ((xyz[0] - xyz[1])**2 + xyz[2]**2))

                fx = f(self.duff_3d_id.points)
                val = h**3 * np.dot(fx, self.duff_3d_id.weights)
                ips.append((elem, val))
                continue
            if v0 in elem.vertices:
                touch_bdr += 1

                # Find n2 and n3, vertices on edges of elem connected to v0.
                n2, n3 = [v.xy_np for v in elem.connected_to_vertex(v0)]

                # Create parametrizations of Q and K.
                gamma_K = lambda y: n0 + (n1 - n0) * y
                gamma_Q = lambda x, z: n0 + (n2 - n0) * x + (n3 - n0) * z
                assert np.all(gamma_Q(0, 0) == gamma_K(0))

            elif v1 in elem.vertices:
                touch_bdr += 1

                # Find n2 and n3 on edges of elem connected to v1.
                n2, n3 = [v.xy_np for v in elem.connected_to_vertex(v1)]

                # Create parametrizations of Q and K.
                gamma_K = lambda y: n1 + (n0 - n1) * y
                gamma_Q = lambda x, z: n1 + (n2 - n1) * x + (n3 - n1) * z
                assert np.all(gamma_Q(0, 0) == gamma_K(0))
            else:
                # Create parametrizations of Q and K.
                gamma_K = lambda y: n0 + (n1 - n0) * y
                gamma_Q = elem.gamma()

            f = lambda xyz: self.u0(gamma_Q(xyz[0], xyz[2])) * G_time(
                np.sum((gamma_Q(xyz[0], xyz[2]) - gamma_K(xyz[1]))**2, axis=0))

            fx = f(self.duff_3d_touch.points)
            val = elem.diam**2 * (d - c) * np.dot(fx,
                                                  self.duff_3d_touch.weights)
            ips.append((elem, val))

        assert id_bdr == 1
        #assert touch_bdr >= 1
        return math.fsum([val for elem, val in ips]), ips

    def linform_vector(self):
        """ Evaluates <M_0 u_0, 1_trial> for all elems in bdr mesh. """
        elems = list(self.bdr_mesh.leaf_elements)
        N = len(elems)
        vec = np.zeros(shape=N)
        for j, elem_trial in enumerate(elems):
            vec[j], _ = self.linform(elem_trial)
        return vec

    def evaluate(self, t, x):
        """ Evaluates (M_0 u_0)(t,x) for t,x not on the bdr. """
        x = np.array(x)

        def f(y):
            xy = x - y
            xy_sqr = np.sum(xy**2, axis=0)
            return 1. / (4 * np.pi * t) * np.exp(-xy_sqr /
                                                 (4 * t)) * self.u0(y)

        return self.space_integrator(f)

        #ips = []
        #for elem in self.initial_mesh.leaf_elements:
        #    val = self.gauss_2d.integrate(f, elem.vertices[0].x,
        #                                  elem.vertices[2].x,
        #                                  elem.vertices[0].y,
        #                                  elem.vertices[2].y)
        #    ips.append((elem, val))

        #result_mesh = math.fsum([val for elem, val in ips])
        #return result_mesh


if __name__ == "__main__":
    import sys
    mesh = MeshParametrized(UnitSquare())
    mesh.uniform_refine()
    elems = list(mesh.leaf_elements)
    print(elems[0])

    val_exact_sing = 0.026593400385924482551
    val_exact_touch = 0.0149249985732353144
    val_exact_bla = 0.0093647867610682937049
    val_exact_blabla = 0.0069966430198107115389
    val_exact = 0.0578798287400388022

    #    val_exact_0 = 0.024680886084096639771
    #    val_exact_1 = 0.033198942471246548041
    #
    for quad_order in range(3, 13, 2):
        IP = InitialPotential(bdr_mesh=mesh,
                              u0=lambda y: np.ones(y.shape[1]),
                              initial_mesh=UnitSquareBoundaryRefined,
                              quad_order=quad_order)
        print(quad_order)
        val_0 = IP.linform(elems[0])
        print(quad_order, abs((val_0 - val_exact)) / val_exact)
        val_1 = IP.linform_alt(elems[0])
        print(quad_order, abs((val_1 - val_exact)) / val_exact)
        val_1_aa = IP.linform_alt_alt(elems[0])
        print(quad_order, abs((val_1_aa - val_exact)) / val_exact)
        val_2, elem_ip = IP.linform(elems[0])
        print(quad_order, abs((val_2 - val_exact)) / val_exact)
        for elem, val in elem_ip:
            print('\t{}\t'.format(elem.vertices[0].xy), end='')
            if elem.vertices[0].xy == (0, 0):
                print(quad_order, abs((val - val_exact_sing)) / val_exact_sing)
            elif elem.vertices[0].xy == (0.5, 0.0):
                print(quad_order,
                      abs((val - val_exact_touch)) / val_exact_touch)
            elif elem.vertices[0].xy == (0, 0.5):
                print(quad_order, abs((val - val_exact_bla)) / val_exact_bla)
            elif elem.vertices[0].xy == (0.5, 0.5):
                print(quad_order,
                      abs((val - val_exact_blabla)) / val_exact_blabla)
        print('')

#    val_exact = 0.0084268990247339470474
#    for quad_order in range(3, 13, 2):
#        IP = InitialPotential(bdr_mesh=mesh,
#                              u0=lambda xy: np.sin(xy[0]) * xy[1],
#                              initial_mesh=UnitSquareBoundaryRefined,
#                              quad_order=quad_order)
#        print(quad_order)
#        val_0 = IP.linform(elems[0])
#        print(quad_order, abs((val_0 - val_exact)) / val_exact)
#        val_1 = IP.linform_alt(elems[0])
#        print(quad_order, abs((val_1 - val_exact)) / val_exact)
#        val_1_aa = IP.linform_alt_alt(elems[0])
#        print(quad_order, abs((val_1_aa - val_exact)) / val_exact)
#        val_2, elem_ip = IP.linform_mesh(elems[0])
#        print(quad_order, abs((val_2 - val_exact)) / val_exact)
#        print('')
#
#    val_exact = 0.010425254168044947963
#    for quad_order in range(3, 13, 2):
#        IP = InitialPotential(bdr_mesh=mesh,
#                              u0=lambda xy: np.sin(xy[0]) * xy[1],
#                              initial_mesh=UnitSquareBoundaryRefined,
#                              quad_order=quad_order)
#        print(quad_order)
#        val_0 = IP.linform(elems[1])
#        print(quad_order, abs((val_0 - val_exact)) / val_exact)
#        val_1 = IP.linform_alt(elems[1])
#        print(quad_order, abs((val_1 - val_exact)) / val_exact)
#        val_1_aa = IP.linform_alt_alt(elems[1])
#        print(quad_order, abs((val_1_aa - val_exact)) / val_exact)
#        val_2, elem_ip = IP.linform_mesh(elems[1])
#        print(quad_order, abs((val_2 - val_exact)) / val_exact)
#        print('')
#
#    mesh.uniform_refine()
#    elems = list(mesh.leaf_elements)
#    print(elems[0])
#    val_exact_0 = 0.0060025200995852428449
#    #val_exact_1 = 0.033198942471246548041
#    for quad_order in range(3, 17, 2):
#        IP = InitialPotential(mesh, lambda y: y[0], quad_order=quad_order)
#        print(quad_order)
#        val_0 = IP.linform(elems[0])
#        print(quad_order, abs((val_0 - val_exact_0)) / val_exact_0)
#        #val_1 = IP.linform_alt(elems[1])
#        #print(quad_order, abs((val_1 - val_exact_1)) / val_exact_1)

#    def f(x):
#        x = np.array(x)
#        #return x[2] + x[0] + x[1]
#        assert np.all(x >= 0)
#        assert np.all(x <= 1)
#        #return np.exp(x[2] * x[0] * x[1])
#        return np.log((x[0] - x[1])**2 + x[2]**2)
#
#    def f_duf_x(x):
#        return x[0] * (np.log((x[0] - x[2])**2 + (x[0] * x[1])**2) +
#                       np.log((x[0] * x[1] - x[2])**2 + x[0]**2))
#
#    def f_duf_z(x):
#        return 2 * x[0] * (np.log((x[0] - x[0] * x[2])**2 + x[1]**2))
#
#    def f_duf_y(x):
#        return x[1] * (np.log((x[0] * x[1] - x[2])**2 + x[1]**2) +
#                       np.log((x[1] - x[2])**2 + (x[0] * x[1])**2))
#
#    def f_duf_xy(x):
#        x, y, z = x
#        return y**2 * np.log(x**2 * y**4 + (y - z)**2) + x * np.log(x**2 + (
#            x * y - z)**2) + x * y**2 * np.log(x**2 * y**4 + (x * y - z)**2)
#
#    def f_T1(x):
#        x_hat = np.array([x[0], x[0] * (1 - x[1]), x[0] * x[1] * x[2]])
#        return f(x_hat) * x[0]**2 * x[1]
#
#    def f_T2(x):
#        #B = np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]])
#        #x_hat = np.array([x[0], x[0] * (1 - x[1]), x[0] * x[1] * x[2]])
#        #return f(B @ x_hat) * x[0]**2 * x[1]
#        x_hat = np.array([x[0], x[0] * (1 - x[1]), x[0] * x[1] * x[2]])
#        y_hat = [x_hat[1] + x_hat[2], x_hat[2], x_hat[0]]
#        return f(y_hat) * x[0]**2 * x[1]
#
#    def f_T3(x):
#        x_hat = np.array([x[0], x[0] * (1 - x[1]), x[0] * x[1] * x[2]])
#        y_hat = [x_hat[0], x_hat[0] - x_hat[2], x_hat[0] - x_hat[1]]
#        return f(y_hat) * x[0]**2 * x[1]
#
#    val_exact = -1.11017344384969642015101064286
#    #scheme_tet = quadpy.t3.get_good_scheme(5)
#    #for N in range(101):
#    for N_poly, N_log in quadrature_rules.LOG_QUAD_RULES:
#        #scheme = quadpy.c3.product(quadpy.c1.gauss_legendre(N))
#        scheme_3d = ProductScheme3D(log_quadrature_scheme(N_poly, N_log))
#        #scheme_3d = ProductScheme3D(gauss_quadrature_scheme(N * 2 + 1))
#        duff_3d = DuffySchemeIdentical3D(scheme_3d, True)
#        val = duff_3d.integrate(f, 0, 1, 0, 1, 0, 1)
#        #val = scheme_3d.integrate(f_duf_x, 0, 1, 0, 1, 0, 1)
#        print(N_poly, N_log, abs((val - val_exact) / val_exact))
#        val_2 = scheme_3d.integrate(f_duf_xy, 0, 1, 0, 1, 0, 1)
#        print(N_poly, N_log, abs((val_2 - val_exact) / val_exact))
#        val_3 = scheme_3d.integrate(f_duf_z, 0, 1, 0, 1, 0, 1)
#        print(N_poly, N_log, abs((val_3 - val_exact) / val_exact))
#        val_4 = scheme_3d.integrate(
#            lambda x: 2 * (f_T1(x) + f_T2(x) + f_T3(x)), 0, 1, 0, 1, 0, 1)
#        print(N_poly, N_log, abs((val_4 - val_exact) / val_exact))
#        print('')
#        #val_tet = scheme_tet.integrate(
#        #    lambda x: x[2] + x[0] + x[1],
#        #    #[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1]]) T1
#        #    #[[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1]]) T2
#        #    [[0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1]])
#        #print(val_tet)
#        #val_tet_duf = scheme.integrate(lambda x: f_T3(x), 0, 1, 0, 1, 0, 1)
#        #print(val_tet_duf)
#        #print(len(scheme.weights), abs((val_tet - val_tet_duf) / val_tet))
#
##    mesh = MeshParametrized(UnitSquare())
##    mesh.uniform_refine()
##    elems = list(mesh.leaf_elements)
##    print(elems[0])
##    val_exact_0 = 0.024680886084096639771
##    val_exact_1 = 0.033198942471246548041
##
##    for quad_order in range(3, 51, 2):
##        IP = InitialPotential(mesh, lambda y: y[0], quad_order=quad_order)
##        print(quad_order)
##        #val_0 = IP.linform(elems[0])
##        #print(quad_order, abs((val_0 - val_exact_0)) / val_exact_0)
##        val_1 = IP.linform(elems[1])
##        print(quad_order, abs((val_1 - val_exact_1)) / val_exact_1)
##        val_1_aa = IP.linform_alt_alt(elems[1])
##        print(quad_order, abs((val_1_aa - val_exact_1)) / val_exact_1)
##
##    mesh.uniform_refine()
##    elems = list(mesh.leaf_elements)
##    print(elems[0])
##    val_exact_0 = 0.0060025200995852428449
##    #val_exact_1 = 0.033198942471246548041
##    for quad_order in range(3, 17, 2):
##        IP = InitialPotential(mesh, lambda y: y[0], quad_order=quad_order)
##        print(quad_order)
##        val_0 = IP.linform(elems[0])
##        print(quad_order, abs((val_0 - val_exact_0)) / val_exact_0)
##        #val_1 = IP.linform_alt(elems[1])
##        #print(quad_order, abs((val_1 - val_exact_1)) / val_exact_1)
