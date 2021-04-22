import numpy as np
from scipy.special import erf, erfc
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
from quadrature import log_log_quadrature_scheme, log_quadrature_scheme, gauss_quadrature_scheme, ProductScheme2D, QuadScheme2D, QuadpyScheme2D
import quadpy


def prolongate(vec_coarse, elems_coarse, elems_fine):
    elem_coarse_2_idx = {k: v for v, k in enumerate(elems_coarse)}
    vec_fine = np.zeros(len(elems_fine))

    for j, elem_fine in enumerate(elems_fine):
        elem_coarse = elem_fine
        while elem_coarse not in elem_coarse_2_idx:
            assert elem_coarse.parent
            elem_coarse = elem_coarse.parent
        assert elem_coarse in elem_coarse_2_idx
        i = elem_coarse_2_idx[elem_coarse]
        vec_fine[j] = vec_coarse[i]
    return vec_fine


def u(t, xy):
    return np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * xy[0]) * np.sin(
        np.pi * xy[1])


def u0(xy):
    return np.sin(np.pi * xy[0]) * np.sin(np.pi * xy[1])


def u_neumann(t, x_hat):
    """ evaluates the neumann trace along the lateral boundary. """
    return -np.pi * np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * (x_hat % 1))


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


mesh = MeshParametrized(UnitSquare())
M0 = InitialOperator(bdr_mesh=mesh,
                     u0=u0,
                     initial_mesh=UnitSquareBoundaryRefined)

pts_T = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
dofs = []
errs_l2 = []
errs_estim = []
errs_hierch = []
errs_pointwise = [[] for _ in pts_T]
elems_prev = None
Phi_prev = None
for k in range(7):
    elems_cur = list(mesh.leaf_elements)
    elems_cur.sort(key=lambda elem: elem.vertices[0].tx)
    N = len(elems_cur)
    print('Loop with {} dofs'.format(N))
    SL = SingleLayerOperator(mesh)
    dofs.append(N)

    print(mesh.gmsh(), file=open("./data/{}.gmsh".format(mesh.md5()), "w"))
    #time_mat_begin = time.time()
    #mat = SL.bilform_matrix(cache_dir='data')
    #print('Calculating matrix took {}s'.format(time.time() - time_mat_begin))

    cache_SL_fn = "{}/SL_dofs_{}_{}.npy".format('data', N, mesh.md5())
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
                a, _, b, _ = *elem_test.time_interval, *elem_trial.time_interval
                c, _, d, _ = *elem_test.space_interval, *elem_trial.space_interval
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
    cache_M0_fn = "{}/M0_dofs_{}_{}.npy".format('data', N, mesh.md5())
    try:
        M0_u0 = np.load(cache_M0_fn)
        print("Loaded Initial Operator from file {}".format(cache_M0_fn))
    except:
        calc_dict = {}
        M0_u0 = np.zeros(shape=N)
        for j, elem_test in enumerate(elems_cur):
            a = elem_test.space_interval[0] - math.floor(
                elem_test.space_interval[0])
            tup = (elem_test.time_interval[0], a)
            if not tup in calc_dict:
                calc_dict[tup] = M0.linform(elem_test)
            M0_u0[j], _ = calc_dict[tup]
        np.save(cache_M0_fn, M0_u0)
        print('Calculating initial potential took {}s'.format(time.time() -
                                                              time_rhs_begin))
        print("Stored Initial Operator to {}".format(cache_M0_fn))

    rhs = -M0_u0

    # Calculate the hierarchical basis estimator.
    if k:
        time_hierach_begin = time.time()
        elem_2_idx_fine = {k: v for v, k in enumerate(elems_cur)}
        Phi_prev_prolong = prolongate(Phi_prev, elems_prev, elems_cur)
        VPhi_prev = mat @ Phi_prev_prolong
        estim = np.zeros(len(elems_prev))
        for i, elem_coarse in enumerate(elems_prev):
            elems_fine = []
            for child in elem_coarse.children:
                elems_fine += child.children
            assert len(elems_fine) == 4

            for elem in elems_fine:
                j = elem_2_idx_fine[elem]
                estim[i] += abs(rhs[j] - VPhi_prev[j])**2 / mat[j, j]

        errs_hierch.append(np.sqrt(np.sum(estim)))
        print('Error estimation of hierarhical estimator took {}s'.format(
            time.time() - time_hierach_begin))

    # Solve.
    time_solve_begin = time.time()
    Phi_cur = np.linalg.solve(mat, -M0_u0)
    print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

    # Check symmetry of the solution.
    #calc_dict = {}
    #for i, elem in enumerate(elems_cur):
    #    t = elem.time_interval[0]
    #    x = elem.space_interval[0] - math.floor(elem.space_interval[0])
    #    if not (t, x) in calc_dict:
    #        calc_dict[t, x] = Phi_cur[i]
    #    else:
    #        assert Phi_cur[i] == approx(calc_dict[t, x])

    # Plot the solution.
    N_time = 2**k
    N_space = 4 * 2**k
    err = np.zeros((N_time, N_space))
    sol = np.zeros((N_time, N_space))
    for i, elem in enumerate(elems_cur):
        sol[int(elem.vertices[0].t * N_time),
            int(elem.vertices[0].x * N_time)] = Phi_cur[i]
        err[int(elem.vertices[0].t * N_time),
            int(elem.vertices[0].x *
                N_time)] = abs(Phi_cur[i] -
                               u_neumann(elem.center.t, elem.center.x))

    plt.figure()
    plt.imshow(sol, origin='lower', extent=[0, 4, 0, 1])
    plt.colorbar()
    plt.savefig('imshow_sol_{}.jpg'.format(k))
    plt.figure()
    plt.imshow(err, origin='lower', extent=[0, 4, 0, 1])
    plt.colorbar()
    plt.savefig('imshow_err_{}.jpg'.format(k))
    #plt.show()

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

    # Evaluate pointwise error in t, [0.5,0.5].
    #print('N={} pointwise error'.format(N))
    #for i, t in enumerate(pts_T):
    #    x = np.array([[0.5], [0.5]])
    #    val = np.dot(Phi_cur, SL.potential_vector(t, x)) + M0.evaluate(t, x)
    #    val_exact = u(t, [0.5, 0.5])
    #    err_rel = abs((val - val_exact) / val_exact)
    #    print('\t Relative error in ({}, {})={}'.format(
    #        t, x.flatten(), err_rel))
    #    errs_pointwise[i].append(err_rel)

    # Calculate the weighted l2 error of the residual.
    err_estim_sqr = np.zeros(N)
    mu = 1 / 2
    nu = 1 / 4
    err_order = 5
    quad_scheme = quadpy.c2.schemes['albrecht_collatz_3']()
    quad_scheme = quadpy.c2.get_good_scheme(3)
    err_order = quad_scheme.degree
    quad_2d = QuadpyScheme2D(quad_scheme)
    quad_2d = ProductScheme2D(log_log_quadrature_scheme(7, 2),
                              log_log_quadrature_scheme(7, 2))
    calc_dict = {}
    eval_dict = {}
    #for i, elem in enumerate(elems_cur):
    #    # Evaluate the residual squared.
    #    def residual_squared(tx):
    #        result = np.zeros(tx.shape[1])
    #        for i, (t, x_hat) in enumerate(zip(tx[0], tx[1])):
    #            x = elem.gamma_space(x_hat)
    #            # Evaluate the SL for our trial function.
    #            VPhi = 0
    #            for j, elem_trial in enumerate(elems_cur):
    #                if t <= elem_trial.time_interval[0]: continue
    #                tup = (elem_trial.time_interval[0] - t,
    #                       elem_trial.space_interval[0], x_hat)
    #                if not tup in eval_dict:
    #                    eval_dict[tup] = SL.evaluate(elem_trial, t, x_hat, x)
    #                VPhi += Phi_cur[j] * eval_dict[tup]

    #                #VPhi += Phi_cur[j] * SL.evaluate(elem_trial, t, x_hat, x)

    #            # Compare with rhs.
    #            #result[i] = VPhi + M0.evaluate_mesh(t, x, initial_mesh)
    #            result[i] = VPhi + M0u0(t, x)
    #        return result**2

    #    # Create initial mesh
    #    #c, d = elem.space_interval
    #    #initial_mesh = UnitSquareBoundaryRefined(elem.gamma_space(c),
    #    #                                         elem.gamma_space(d))

    #    t = elem.vertices[0].t
    #    x = elem.vertices[0].x % 1
    #    if not (t, x) in calc_dict:
    #        calc_dict[t, x] = (elem.h_x**(-2 * mu) +
    #                           elem.h_t**(-2 * nu)) * quad_2d.integrate(
    #                               residual_squared, *elem.time_interval, *
    #                               elem.space_interval)

    #    err_estim_sqr[i] = calc_dict[t, x]

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
        if k > 1:
            rates_hierch = np.log(
                np.array(errs_hierch[1:]) /
                np.array(errs_hierch[:-1])) / np.log(
                    np.array(dofs[1:-1]) / np.array(dofs[:-2]))
    else:
        rates_estim = []
        rates_l2 = []
        rates_hierch = []

    print(
        '\ndofs={}\nerrs_l2={}\nerr_estim={}\nerr_hierch={}\n\nrates_l2={}\nrates_estim={}\nrates_hierch={}\n------'
        .format(dofs, errs_l2, errs_estim, errs_hierch, rates_l2, rates_estim,
                rates_hierch))
    mesh.uniform_refine()
    Phi_prev = Phi_cur
    elems_prev = elems_cur
