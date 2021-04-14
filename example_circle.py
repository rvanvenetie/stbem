import numpy as np
import random
import time
from scipy.special import exp1
from parametrization import Circle, UnitSquare, LShape
from mesh import Mesh, MeshParametrized
import itertools
from quadrature import log_quadrature_scheme, gauss_quadrature_scheme, ProductScheme2D, DuffyScheme2D
from single_layer import SingleLayerOperator


def u(t, x):
    return 1 / (4 * np.pi * t) * np.exp(-((x[0] - 1)**2 + (x[1] - 1)**2) /
                                        (4 * t))


def u_int(a, b):
    """ Function u integrated over the time interval [a,b]. """
    if a == 0:
        time_integrated_kernel = lambda xy: 1. / (4 * np.pi) * exp1(xy /
                                                                    (4 * b))
    else:
        time_integrated_kernel = lambda xy: 1. / (4 * np.pi) * (exp1(xy / (
            4 * b)) - exp1(xy / (4 * a)))

    return lambda x: time_integrated_kernel((x[0] - 1)**2 + (x[1] - 1)**2)


def prolongate(vec_coarse, elems_coarse, elems_fine):
    vec_fine = np.zeros(len(elems_fine))
    elem_coarse_2_idx = {}
    for i, elem_coarse in enumerate(elems_coarse):
        elem_coarse_2_idx[elem_coarse] = i

    for j, elem_fine in enumerate(elems_fine):
        elem_coarse = elem_fine
        while elem_coarse not in elem_coarse_2_idx:
            assert elem_coarse.parent
            elem_coarse = elem_coarse.parent
        assert elem_coarse in elem_coarse_2_idx
        i = elem_coarse_2_idx[elem_coarse]
        vec_fine[j] = vec_coarse[i]
    return vec_fine


mesh = MeshParametrized(Circle())
Phi_cur = None
Phi_prev = None
elems_prev = None
elems_cur = None

val_exact = u(0.5, [0, 0])
dofs = []
err_pointwise = []
err_h_h2 = []
err_h_h2_l2 = []
for _ in range(10):
    SL = SingleLayerOperator(mesh)
    time_mat_begin = time.time()
    mat = SL.bilform_matrix()
    elems_cur = list(mesh.leaf_elements)
    N = len(elems_cur)
    dofs.append(N)
    print('Loop with {} dofs'.format(N))
    print('Calculating matrix took {}s'.format(time.time() - time_mat_begin))

    # Calculate RHS
    time_rhs_begin = time.time()
    gauss_scheme = gauss_quadrature_scheme(105)
    rhs = np.zeros(shape=N)
    for i, elem_test in enumerate(elems_cur):
        f_int = u_int(*elem_test.time_interval)
        f_param = lambda x: f_int(elem_test.gamma_space(x))
        rhs[i] = gauss_scheme.integrate(f_param, *elem_test.space_interval)
    print('Calculating rhs took {}s'.format(time.time() - time_rhs_begin))

    # Solve
    time_solve_begin = time.time()
    Phi_cur = np.linalg.solve(mat, rhs)
    print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

    # Evaluate in 0.5, [0,0].
    val = np.dot(Phi_cur, SL.potential_vector(0.5, np.array([[0], [0]])))
    print('N={}\terr_pointwise={}'.format(N, (val - val_exact) / val_exact))
    err_pointwise.append(abs(val - val_exact))

    if Phi_prev is not None:
        Phi_prev_prolong = prolongate(Phi_prev, elems_prev, elems_cur)
        diff = Phi_cur - Phi_prev_prolong
        err = np.sqrt(diff.T @ mat @ diff)
        print('N={}\terr_h_h2={}'.format(len(elems_prev), err))
        err_h_h2.append(err)

        err_l2 = 0
        for j, elem in enumerate(elems_cur):
            err_l2 += (elem.space_interval[1] - elem.space_interval[0]) * (
                elem.time_interval[1] - elem.time_interval[0]) * diff[j]**2

        err_l2 = np.sqrt(err_l2)
        err_h_h2_l2.append(err_l2)
        print('N={}\terr_h_h2_l2={}'.format(len(elems_prev), err_l2))

    elems_prev = elems_cur
    Phi_prev = Phi_cur
    mesh.uniform_refine()
    print('\ndofs={}\nerr_pointwise={}\nerr_h_h2={}\nerr_h_h2_l2={}\n------'.
          format(dofs, err_pointwise, err_h_h2, err_h_h2_l2))
