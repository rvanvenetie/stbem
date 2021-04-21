import numpy as np
import multiprocessing as mp
from scipy.special import erf, erfc
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


def error_estim_l2(i):
    elem = elems_cur[i]
    if elem.vertices[0].x >= 1: return 0

    # Evaluate the residual squared.
    def residual_squared(tx):
        result = np.zeros(tx.shape[1])
        for i, (t, x_hat) in enumerate(zip(tx[0], tx[1])):
            x = elem.gamma_space(x_hat)
            #x = mesh.gamma_space.eval(x_hat)
            # Evaluate the SL for our trial function.
            VPhi = 0
            for j, elem_trial in enumerate(elems_cur):
                VPhi += Phi_cur[j] * SL.evaluate(elem_trial, t, x_hat, x)

            # Compare with rhs.
            #result[i] = VPhi + M0.evaluate_mesh(t, x, initial_mesh)
            result[i] = VPhi + M0u0(t, x)
        return result**2

    return (elem.h_x**(-1) + elem.h_t**(-0.5)) * gauss_2d.integrate(
        residual_squared, *elem.time_interval, *elem.space_interval)


def SL_mat_col(j):
    elem_trial = elems_cur[j]
    col = np.zeros(N)
    assert elem_trial.vertices[0].t == 0
    for i, elem_test in enumerate(elems_cur):
        if elem_test.time_interval[1] <= elem_trial.time_interval[0]:
            continue
        col[i] = SL.bilform(elem_trial, elem_test)
    return col


def IP_rhs(j):
    elem_test = elems_cur[j]
    if elem_test.vertices[0].x >= 1: return 0
    else: return M0.linform(elem_test)[0]


if __name__ == '__main__':
    N_procs = mp.cpu_count()
    mp.set_start_method('fork')
    print('Running parallel with {} threads.'.format(N_procs))
    dofs = []
    errs_l2 = []
    errs_estim = []
    h_x = 2
    for k in range(10):
        h_x = h_x / 2
        h_t = h_x**(6 / 5)
        N_x = 4 * round(1 / h_x)
        N_t = round(1 / h_t)
        mesh_space = [Fraction(4 * j, N_x) for j in range(N_x + 1)]
        mesh_time = [Fraction(j, N_t) for j in range(N_t + 1)]
        print(mesh_space)

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

        cache_SL_fn = "{}/SL_dofs_{}_{}.npy".format('data', N, mesh.md5())
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

        # Calculate initial potential.
        time_rhs_begin = time.time()
        cache_M0_fn = "{}/M0_dofs_{}_{}.npy".format('data', N, mesh.md5())
        try:
            M0_u0 = np.load(cache_M0_fn)
            print("Loaded Initial Operator from file {}".format(cache_M0_fn))
        except:
            M0_u0 = np.array(mp.Pool(N_procs).map(IP_rhs, range(N)))
            for j, elem_test in enumerate(elems_cur):
                if elem_test.vertices[0].x < 1: continue
                M0_u0[j] = M0_u0[j // N_x * N_x + j % (N_x // 4)]
            np.save(cache_M0_fn, M0_u0)
            print('Calculating initial potential took {}s'.format(
                time.time() - time_rhs_begin))
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
                gauss_2d.integrate(err, *elem.time_interval,
                                   *elem.space_interval))
        err_l2 = np.sqrt(math.fsum(err_l2))
        errs_l2.append(err_l2)
        print('Error estimation of \Phi - \partial_n took {}s'.format(
            time.time() - time_l2_begin))

        # Calculate the weighted l2 error of the residual set global vars.
        err_order = 3
        gauss_2d = ProductScheme2D(gauss_quadrature_scheme(err_order))
        err_estim_sqr = np.array(
            mp.Pool(N_procs).map(error_estim_l2, range(N)))
        for j, elem in enumerate(elems_cur):
            if elem.vertices[0].x < 1: continue
            err_estim_sqr[j] = err_estim_sqr[j // N_x * N_x + j % (N_x // 4)]

        errs_estim.append(np.sqrt(np.sum(err_estim_sqr)))
        print('Error estimation of weighted residual of order {} took {}s'.
              format(err_order,
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
