import numpy as np
from fractions import Fraction
from pytest import approx
import matplotlib.pyplot as plt
import math
from mesh import Mesh, MeshParametrized
from initial_potential import InitialOperator
from initial_mesh import UnitSquareBoundaryRefined
import initial_mesh
from single_layer import SingleLayerOperator
import time
from parametrization import Circle, UnitSquare, LShape
from quadrature import gauss_quadrature_scheme, ProductScheme2D
import quadpy


def u(t, xy):
    return np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * xy[0]) * np.sin(
        np.pi * xy[1])


def u0(xy):
    return np.sin(np.pi * xy[0]) * np.sin(np.pi * xy[1])


def u_neumann(t, x_hat):
    """ evaluates the neumann trace along the lateral boundary. """
    return -np.pi * np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * (x_hat % 1))


dofs = []
errs_l2 = []
errs_estim = []
h = 2
for k in range(10):
    h = h / 2
    h_x = h
    h_t = h**(6 / 5)
    N_x = round(1 / h_x)
    N_t = round(1 / h_t)
    mesh_space = []
    for j in range(4 * N_x + 1):
        mesh_space.append(Fraction(j, N_x))
    mesh_time = []
    for j in range(N_t + 1):
        mesh_time.append(Fraction(j, N_t))

    mesh = MeshParametrized(UnitSquare(),
                            initial_space_mesh=mesh_space,
                            initial_time_mesh=mesh_time)
    print(mesh.gmsh(), file=open("./data/{}.gmsh".format(mesh.md5()), "w"))
    M0 = InitialOperator(bdr_mesh=mesh,
                         u0=u0,
                         initial_mesh=UnitSquareBoundaryRefined)

    elems_cur = list(mesh.leaf_elements)
    N = len(elems_cur)
    print('Loop with {} dofs for N_x = {} N_t = {}'.format(N, N_x, N_t))
    SL = SingleLayerOperator(mesh)
    dofs.append(N)

    cache_SL_fn = "{}/SL_graded_fast_dofs_{}_{}.npy".format(
        'data', N, mesh.md5())
    try:
        mat = np.load(cache_SL_fn)
        print("Loaded Single Layer from file {}".format(cache_SL_fn))
    except:
        calc_dict = {}
        time_mat_begin = time.time()
        mat = np.zeros((N, N))
        for i, elem_test in enumerate(elems_cur):
            for j, elem_trial in enumerate(elems_cur):
                if elem_test.time_interval[1] <= elem_trial.time_interval[0]:
                    continue
                a, b = elem_test.vertices[0].t, elem_trial.vertices[0].t
                c, d = elem_test.vertices[0].x, elem_trial.vertices[0].x
                tup = (a - b, c - math.floor(c), (d - math.floor(c)) % 4)
                if not tup in calc_dict:
                    calc_dict[tup] = SL.bilform(elem_trial, elem_test)

                mat[i, j] = calc_dict[tup]
        try:
            np.save(cache_SL_fn, mat)
            print("Stored Single Layer to {}".format(cache_SL_fn))
        except:
            pass
        print('Calculating matrix fast took {}s'.format(time.time() -
                                                        time_mat_begin))

    # Calculate initial potential.
    time_rhs_begin = time.time()
    cache_M0_fn = "{}/M0_graded_fast_dofs_{}_{}.npy".format(
        'data', N, mesh.md5())
    try:
        M0_u0 = np.load(cache_M0_fn)
        print("Loaded Initial Operator from file {}".format(cache_M0_fn))
    except:
        calc_dict = {}
        M0_u0 = np.zeros(shape=N)
        for j, elem_test in enumerate(elems_cur):
            a = elem_test.vertices[0].x - math.floor(elem_test.vertices[0].x)
            tup = (elem_test.vertices[0].t, a)
            if not tup in calc_dict:
                calc_dict[tup] = M0.linform(elem_test)
            M0_u0[j], _ = calc_dict[tup]
        np.save(cache_M0_fn, M0_u0)
        print('Calculating initial potential took {}s'.format(time.time() -
                                                              time_rhs_begin))
        print("Stored Initial Operator to {}".format(cache_M0_fn))

    # Solve.
    time_solve_begin = time.time()
    Phi_cur = np.linalg.solve(mat, -M0_u0)
    print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

    # Estimate the l2 error of the neumann trace.
    time_l2_begin = time.time()
    gauss = gauss_quadrature_scheme(11)
    gauss_2d = ProductScheme2D(gauss)
    err_l2 = []
    for i, elem in enumerate(elems_cur):
        err = lambda tx: (Phi_cur[i] - u_neumann(tx[0], tx[1]))**2
        err_l2.append(
            gauss_2d.integrate(err, *elem.time_interval, *elem.space_interval))
    err_l2 = np.sqrt(math.fsum(err_l2))
    errs_l2.append(err_l2)
    print(
        'Error estimation of \Phi - \partial_n took {}s'.format(time.time() -
                                                                time_l2_begin))
    # Calculate the weighted l2 error of the residual.
    err_estim_sqr = np.zeros(N)
    err_order = 3
    gauss_2d = ProductScheme2D(gauss_quadrature_scheme(err_order))
    #quad_scheme = quadpy.get_good_scheme(3)
    calc_dict = {}
    for i, elem in enumerate(elems_cur):
        # Evaluate the residual squared.
        def residual_squared(tx):
            result = np.zeros(tx.shape[1])
            for i, (t, x_hat) in enumerate(zip(tx[0], tx[1])):
                x = elem.gamma_space(x_hat)
                # Evaluate the SL for our trial function.
                VPhi = 0
                for j, elem_trial in enumerate(elems_cur):
                    VPhi += Phi_cur[j] * SL.evaluate(elem_trial, t, x_hat, x)

                # Compare with rhs.
                result[i] = VPhi + M0.evaluate(t, x)
            return result**2

        # Create initial mesh
        #c, d = elem.space_interval
        #initial_mesh = UnitSquareBoundaryRefined(elem.gamma_space(c),
        #                                         elem.gamma_space(d))

        t = elem.time_interval[0]
        x = elem.vertices[0].x - math.floor(elem.vertices[0].x)
        if not (t, x) in calc_dict:
            calc_dict[t, x] = (elem.h_x**(-1) +
                               elem.h_t**(-0.5)) * gauss_2d.integrate(
                                   residual_squared, *elem.time_interval, *
                                   elem.space_interval)

        err_estim_sqr[i] = calc_dict[t, x]

    errs_estim.append(np.sqrt(np.sum(err_estim_sqr)))
    print('Error estimation of weighted residual of order {} took {}s'.format(
        err_order,
        time.time() - time_l2_begin))

    if k:
        rates_estim = np.log(
            np.array(errs_estim[1:]) / np.array(errs_estim[:-1])) / np.log(
                np.array(dofs[1:]) / np.array(dofs[:-1]))
        rates_l2 = np.log(
            np.array(errs_l2[1:]) / np.array(errs_l2[:-1])) / np.log(
                np.array(dofs[1:]) / np.array(dofs[:-1]))
    else:
        rates_estim = []
        rates_l2 = []

    print(
        '\ndofs={}\nerrs_l2={}\nerr_estim={}\n\nrates_l2={}\nrates_estim={}\n------'
        .format(dofs, errs_l2, errs_estim, rates_l2, rates_estim))
