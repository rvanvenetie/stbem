import numpy as np
import multiprocessing as mp
import quadpy
import random
import time
from scipy.special import exp1
from parametrization import Circle, UnitSquare, LShape
from mesh import Mesh, MeshParametrized
import itertools
from quadrature import log_quadrature_scheme, gauss_quadrature_scheme, ProductScheme2D, DuffyScheme2D
from single_layer import SingleLayerOperator


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


# RHS function.
def u(t, x):
    #if t == 0: return 0
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


# Evaluate a column of the single layer matrix using global vars.
def SL_mat_col(j):
    elem_trial = elems_cur[j]
    col = np.zeros(N)
    assert elem_trial.vertices[0].t == 0
    for i, elem_test in enumerate(elems_cur):
        if elem_test.time_interval[1] <= elem_trial.time_interval[0]:
            continue
        col[i] = SL.bilform(elem_trial, elem_test)
    return col


def error_estim_l2(i):
    elem = elems_cur[i]

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
            result[i] = u(t, x) - VPhi

        return result**2

    return (elem.h_x**(-1) + elem.h_t**(-0.5)) * gauss_2d.integrate(
        residual_squared, *elem.time_interval, *elem.space_interval)


if __name__ == "__main__":
    N_procs = mp.cpu_count()
    mp.set_start_method('fork')
    print('Running parallel with {} threads.'.format(N_procs))
    mesh = MeshParametrized(Circle())
    dofs = []
    errs_estim = []
    errs_hierch = []
    elems_prev = None
    Phi_prev = None
    for k in range(10):
        elems_cur = list(mesh.leaf_elements)
        elems_cur.sort(key=lambda elem: elem.vertices[0].tx)
        N_x = 4 * 2**k
        N_t = 2**k
        N = len(elems_cur)
        dofs.append(N)
        print(mesh.gmsh(),
              file=open("./data/circle_{}_{}.gmsh".format(N, mesh.md5()), "w"))
        print('Loop with {} dofs for N_x = {} N_t = {}'.format(N, N_x, N_t))

        SL = SingleLayerOperator(mesh)
        # Evaluate the single layer matrix parallel.
        cache_SL_fn = "{}/SL_circle_dofs_{}_{}.npy".format(
            'data', N, mesh.md5())
        try:
            mat = np.load(cache_SL_fn)
            print("Loaded Single Layer from file {}".format(cache_SL_fn))
        except:
            time_mat_begin = time.time()
            mat = np.zeros((N, N))
            for j, col in enumerate(
                    mp.Pool(N_procs).imap(SL_mat_col, range(N_x))):
                mat[:, j] = col

            for j, elem_trial in enumerate(elems_cur[N_x:]):
                assert elem_trial.vertices[0].t > 0
                start = N_x + j // N_x * N_x
                mat[start:, j + N_x] = (mat[:, j % N_x])[:-start]

            try:
                np.save(cache_SL_fn, mat)
                print("Stored Single Layer to {}".format(cache_SL_fn))
            except:
                pass
            print('Calculating matrix fast took {}s'.format(time.time() -
                                                            time_mat_begin))

        # Calculate RHS.
        #time_rhs_begin = time.time()
        #rhs = np.zeros(shape=N)
        #for i, elem_test in enumerate(elems_cur):
        #    rhs[i] = elem_test.h_t * elem_test.h_x
        #assert np.allclose(rhs, SL.rhs_vector(g))
        #print('Calculating rhs took {}s'.format(time.time() - time_rhs_begin))
        time_rhs_begin = time.time()
        gauss_scheme = gauss_quadrature_scheme(105)
        gauss_2d = ProductScheme2D(gauss_scheme, gauss_scheme)
        rhs = np.zeros(shape=N)
        rhs_2d = np.zeros(N)
        for i, elem_test in enumerate(elems_cur):
            f_int = u_int(*elem_test.time_interval)
            f_param = lambda x: f_int(elem_test.gamma_space(x))
            rhs[i] = gauss_scheme.integrate(f_param, *elem_test.space_interval)
        print('Calculating rhs took {}s'.format(time.time() - time_rhs_begin))

        # Calculate the hierarchical basis estimator.
        if k:
            assert elems_prev
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

        # Solve
        time_solve_begin = time.time()
        Phi_cur = np.linalg.solve(mat, rhs)
        print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

        # Calculate the weighted l2 error of the residual set global vars.
        time_l2_begin = time.time()
        err_order = 3
        gauss_2d = ProductScheme2D(gauss_quadrature_scheme(err_order))
        err_estim_sqr = np.array(
            mp.Pool(N_procs).map(error_estim_l2, range(N)))

        errs_estim.append(np.sqrt(np.sum(err_estim_sqr)))
        print('Error estimation of weighted residual of order {} took {}s'.
              format(err_order,
                     time.time() - time_l2_begin))

        if k:
            rates_estim = np.log(
                np.array(errs_estim[1:]) / np.array(errs_estim[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            if k > 1:
                rates_hierch = np.log(
                    np.array(errs_hierch[1:]) /
                    np.array(errs_hierch[:-1])) / np.log(
                        np.array(dofs[1:-1]) / np.array(dofs[:-2]))
        else:
            rates_estim = []
            rates_hierch = []

        print(
            '\ndofs={}\nerr_estim={}\nerr_hierch={}\n\nrates_estim={}\nrates_hierch={}\n------'
            .format(dofs, errs_estim, errs_hierch, rates_estim, rates_hierch))

        # Refine
        mesh.uniform_refine()
        Phi_prev = Phi_cur
        elems_prev = elems_cur
