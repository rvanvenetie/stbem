import numpy as np
import multiprocessing as mp
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


def SL_coloc_row(i):
    row = np.zeros(N)
    elem_test = elems_cur[i]
    t, x_hat = float(elem_test.center.t), float(elem_test.center.x)
    x = elem_test.gamma_space(x_hat)
    for j, elem_trial in enumerate(elems_cur):
        row[j] = SL.evaluate(elem_trial, t, x_hat, x)
    return row


if __name__ == "__main__":
    N_procs = mp.cpu_count()
    mp.set_start_method('fork')
    mesh = MeshParametrized(UnitSquare())
    M0 = InitialOperator(bdr_mesh=mesh,
                         u0=u0,
                         initial_mesh=UnitSquareBoundaryRefined)
    dofs = []
    errs_l2 = []
    for k in range(10):
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

        cache_SL_fn = "{}/SL_coloc_dofs_{}_{}.npy".format(
            'data', N, mesh.md5())
        try:
            mat = np.load(cache_SL_fn)
            print("Loaded Single Layer from file {}".format(cache_SL_fn))
        except:
            calc_dict = {}
            time_mat_begin = time.time()
            mat = np.zeros((N, N))
            for i, row in enumerate(
                    mp.Pool(N_procs).imap(SL_coloc_row, range(N))):
                mat[i, :] = row
            try:
                np.save(cache_SL_fn, mat)
                print("Stored Single Layer to {}".format(cache_SL_fn))
            except:
                pass
            print('Calculating matrix fast took {}s'.format(time.time() -
                                                            time_mat_begin))

        # Calculate initial potential.
        time_rhs_begin = time.time()
        cache_M0_fn = "{}/M0_coloc_dofs_{}_{}.npy".format(
            'data', N, mesh.md5())
        try:
            M0_u0 = np.load(cache_M0_fn)
            print("Loaded Initial Operator from file {}".format(cache_M0_fn))
        except:
            calc_dict = {}
            M0_u0 = np.zeros(shape=N)
            for j, elem_test in enumerate(elems_cur):
                t, x_hat = float(elem_test.center.t), float(elem_test.center.x)
                M0_u0[j] = M0u0(t, elem_test.gamma_space(x_hat))
            np.save(cache_M0_fn, M0_u0)
            print('Calculating initial potential took {}s'.format(
                time.time() - time_rhs_begin))
            print("Stored Initial Operator to {}".format(cache_M0_fn))

        rhs = -M0_u0

        # Solve.
        time_solve_begin = time.time()
        Phi_cur = np.linalg.solve(mat, rhs)
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

        if k:
            rates_l2 = np.log(
                np.array(errs_l2[1:]) / np.array(errs_l2[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
        else:
            rates_l2 = []

        print('\ndofs={}\nerrs_l2={}\nrates_l2={}\n------'.format(
            dofs, errs_l2, rates_l2))
        mesh.uniform_refine()
