import hashlib
import math
import multiprocessing as mp
import time

import numpy as np
from scipy.special import exp1

from .mesh import Element
from .quadrature import (DuffySchemeIdentical3D, DuffySchemeTouch3D,
                         ProductScheme2D, ProductScheme3D,
                         gauss_quadrature_scheme, log_quadrature_scheme)

FPI_INV = (4 * np.pi)**-1


def time_integrated_kernel(a, b):
    """ Returns heat kernel G(t,x) integrated over t in [a,b]. """
    if a == 0:
        return lambda xy: 1. / (4 * np.pi) * exp1(xy / (4 * b))
    else:
        return lambda xy: 1. / (4 * np.pi) * (exp1(xy /
                                                   (4 * b)) - exp1(xy /
                                                                   (4 * a)))


def MP_M0_val(j: int):
    """ Function to evaluate M0 in parallel using the multiprocessing library. """
    global __M0, __elems
    return __M0.linform(__elems[j])[0]


class InitialOperator:
    def __init__(self,
                 bdr_mesh,
                 u0,
                 initial_mesh=None,
                 quad_int=12,
                 quad_eval=19,
                 cache_dir=None,
                 problem=None):
        self.u0 = u0
        self.bdr_mesh = bdr_mesh
        self.initial_mesh = initial_mesh

        # Quadrature schemes for the evaluation.
        self.space_integrator = bdr_mesh.gamma_space.integrator(quad_eval)
        self.gauss_2d = ProductScheme2D(gauss_quadrature_scheme(quad_eval))

        # Quadrature schemes for the inner product.
        self.log_scheme = log_quadrature_scheme(quad_int, quad_int)
        self.duff_3d_id = DuffySchemeIdentical3D(ProductScheme3D(
            self.log_scheme),
                                                 symmetric_xy=False)
        self.duff_3d_touch = DuffySchemeTouch3D(
            ProductScheme3D(self.log_scheme))

        # Storing options.
        self.cache_dir = cache_dir
        if problem is None: problem = str(self.bdr_mesh.gamma_space)
        self.problem = problem

    def linform(self, elem_trial: Element):
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
        math.isclose(d - c, np.linalg.norm(n0 - n1))

        id_bdr = 0
        touch_bdr = 0
        ips = []
        for elem in initial_mesh.leaf_elements:
            # Check whether this element has an identical bdr,
            if v0 in elem.vertices and v1 in elem.vertices:
                id_bdr += 1
                h = d - c
                math.isclose(elem.diam, h)

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

            # We will use duffy 3d.
            xyz = self.duff_3d_touch.points
            xz = gamma_Q(xyz[0], xyz[2])
            y = gamma_K(xyz[1])

            # Evaluate time integrated kernel.
            xz_y = (xz - y)**2
            xz_y = xz_y[0] + xz_y[1]
            if a == 0:
                fx = self.u0(xz) * exp1(xz_y / (4 * b))
            else:
                fx = self.u0(xz) * (exp1(xz_y / (4 * b)) - exp1(xz_y /
                                                                (4 * a)))

            val = elem.diam**2 * (d - c) * FPI_INV * np.dot(
                fx, self.duff_3d_touch.weights)
            ips.append((elem, val))

        assert id_bdr == 1
        #assert touch_bdr >= 1
        return math.fsum([val for elem, val in ips]), ips

    def linform_vector(self, elems=None, use_mp=False):
        """ Evaluates <M_0 u_0, 1_trial> for all elems in bdr mesh. """
        if elems is None:
            elems = list(self.bdr_mesh.leaf_elements)
        N = len(elems)

        if self.cache_dir is not None:
            md5 = hashlib.md5((str(self.bdr_mesh.gamma_space) +
                               str(elems)).encode()).hexdigest()
            cache_fn = "{}/M0_{}_{}_{}.npy".format(self.cache_dir,
                                                   self.problem, N, md5)
            try:
                vec = np.load(cache_fn)
                print("Loaded Initial Operator from file {}".format(cache_fn))
                return vec
            except:
                pass

        time_rhs_begin = time.time()
        if not use_mp:
            vec = np.zeros(shape=N)
            for j, elem_trial in enumerate(elems):
                vec[j], _ = self.linform(elem_trial)
        else:
            # Set up global variables for parallelizing.
            globals()['__elems'] = elems
            globals()['__M0'] = self
            cpu = mp.cpu_count()
            vec = np.array(
                mp.Pool(mp.cpu_count()).map(MP_M0_val, range(N),
                                            N // (cpu * 8) + 1))

        print('Calculating initial potential took {}s'.format(time.time() -
                                                              time_rhs_begin))
        if self.cache_dir is not None:
            try:
                np.save(cache_fn, vec)
                print("Stored Initial Operator to {}".format(cache_fn))
            except:
                pass

        return vec

    def evaluate(self, t, x):
        """ Evaluates (M_0 u_0)(t,x) for t,x. """
        x = np.array(x)

        def f(y):
            xy = x - y
            xy_sqr = np.sum(xy**2, axis=0)
            return 1. / (4 * np.pi * t) * np.exp(-xy_sqr /
                                                 (4 * t)) * self.u0(y)

        #if (t < 0.01):
        #    print('this seems to become instable')

        return self.space_integrator(f)

    def evaluate_mesh(self, t, x, initial_mesh):
        """ Evaluates (M_0 u_0)(t,x) for t,x using a meshed Omega. """
        def f(y):
            xy = x - y
            xy_sqr = np.sum(xy**2, axis=0)
            return 1. / (4 * np.pi * t) * np.exp(-xy_sqr /
                                                 (4 * t)) * self.u0(y)

        ips = []
        for elem in initial_mesh.leaf_elements:
            val = self.gauss_2d.integrate(f, float(elem.vertices[0].x),
                                          float(elem.vertices[2].x),
                                          float(elem.vertices[0].y),
                                          float(elem.vertices[2].y))
            ips.append((elem, val))

        result_mesh = math.fsum([val for elem, val in ips])
        return result_mesh
