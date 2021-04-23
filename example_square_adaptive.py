import numpy as np
from copy import deepcopy
import multiprocessing as mp
import os
from mesh import Prolongate
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
import hashlib


def SL_mat_col(j):
    elem_trial = elems_glob[j]
    col = np.zeros(len(elems_glob))
    for i, elem_test in enumerate(elems_glob):
        if elem_test.time_interval[1] <= elem_trial.time_interval[0]:
            continue
        col[i] = SL.bilform(elem_trial, elem_test)
    return col


def IP_rhs(j):
    elem_test = elems_glob[j]
    return M0.linform(elem_test)[0]


def SL_matrix(elems):
    """ Evaluate the single layer matrix in parallel. """
    global elems_glob
    elems_glob = elems

    N = len(elems)
    md5 = hashlib.md5(str(elems).encode()).hexdigest()
    cache_SL_fn = "{}/SL_dofs_{}_{}.npy".format('data', N, md5)
    if os.path.isfile(cache_SL_fn):
        mat = np.load(cache_SL_fn)
        print("Loaded Single Layer from file {}".format(cache_SL_fn))
        return mat

    time_mat_begin = time.time()
    mat = np.zeros((N, N))
    for j, col in enumerate(mp.Pool(N_procs).imap(SL_mat_col, range(N))):
        mat[:, j] = col

    try:
        np.save(cache_SL_fn, mat)
        print("Stored Single Layer to {}".format(cache_SL_fn))
    except:
        pass
    print('Calculating SL matrix took {}s'.format(time.time() -
                                                  time_mat_begin))
    return mat


def RHS_vector(elems):
    """ Evaluate the initial potential vector in parallel. """
    global elems_glob
    elems_glob = elems

    N = len(elems)
    md5 = hashlib.md5(str(elems).encode()).hexdigest()
    cache_M0_fn = "{}/M0_dofs_{}_{}.npy".format('data', N, md5)
    if os.path.isfile(cache_M0_fn):
        print("Loaded Initial Operator from file {}".format(cache_M0_fn))
        return -np.load(cache_M0_fn)

    time_rhs_begin = time.time()
    M0_u0 = np.array(mp.Pool(N_procs).map(IP_rhs, range(N)))
    np.save(cache_M0_fn, M0_u0)
    print('Calculating initial potential took {}s'.format(time.time() -
                                                          time_rhs_begin))
    print("Stored Initial Operator to {}".format(cache_M0_fn))
    return -M0_u0


def HierarchicalErrorEstimator(Phi, mesh, SL, RHS):
    """ Returns the hierarchical basis estimator for given function. """

    # Calcualte uniform refinement of the mesh.
    leaf_elems = mesh.leaf_elements
    elems_coarse = list(leaf_elems)
    mesh.uniform_refine()
    elems_fine = list(mesh.leaf_elements)
    mesh.leaf_elements = leaf_elems

    elem_2_idx_fine = {k: v for v, k in enumerate(elems_fine)}

    # Prolongate Phi to fine mesh.
    Phi_fine = Prolongate(Phi, elems_coarse, elems_fine)

    # Evaluate SL matrix on the fine mesh.
    mat = SL(elems_fine)
    VPhi = mat @ Phi_fine

    # Evaluate the RHS on the fine mesh.
    rhs = RHS(elems_fine)

    estim = np.zeros(len(elems_coarse))
    for i, elem_coarse in enumerate(elems_coarse):
        children_fine = []
        for child in elem_coarse.children:
            children_fine += child.children
        assert len(children_fine) == 4

        for elem in children_fine:
            j = elem_2_idx_fine[elem]
            estim[i] += abs(rhs[j] - VPhi[j])**2 / mat[j, j]

    return np.sqrt(np.sum(estim)), estim


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


if __name__ == "__main__":
    N_procs = mp.cpu_count()
    mp.set_start_method('fork')
    print('Running parallel with {} threads.'.format(N_procs))

    mesh = MeshParametrized(UnitSquare())
    M0 = InitialOperator(bdr_mesh=mesh,
                         u0=u0,
                         initial_mesh=UnitSquareBoundaryRefined)

    dofs = []
    errs_l2 = []
    errs_estim = []
    errs_hierch = []
    for k in range(10):
        elems = list(mesh.leaf_elements)
        N = len(mesh.leaf_elements)
        print('Loop with {} dofs'.format(N))
        SL = SingleLayerOperator(mesh)
        dofs.append(N)

        # Calculate SL matrix.
        mat = SL_matrix(elems)

        # Calculate initial potential.
        rhs = RHS_vector(elems)

        # Solve.
        time_solve_begin = time.time()
        Phi = np.linalg.solve(mat, rhs)
        print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

        # Estimate the l2 error of the neumann trace.
        time_l2_begin = time.time()
        gauss_2d = ProductScheme2D(gauss_quadrature_scheme(11))
        err_l2 = []
        for i, elem in enumerate(elems):
            err = lambda tx: (Phi[i] - u_neumann(tx[0], tx[1]))**2
            err_l2.append(
                gauss_2d.integrate(err, *elem.time_interval,
                                   *elem.space_interval))
        errs_l2.append(np.sqrt(math.fsum(err_l2)))
        print('Error estimation of \Phi - \partial_n took {}s'.format(
            time.time() - time_l2_begin))

        # Do the hierarhical error estimator.
        time_hierarch_begin = time.time()
        err_tot, err_loc = HierarchicalErrorEstimator(Phi, mesh, SL_matrix,
                                                      RHS_vector)
        errs_hierch.append(err_tot)
        print('Hierarchical error estimator took {}s'.format(
            time.time() - time_hierarch_begin))

        if k:
            rates_l2 = np.log(
                np.array(errs_l2[1:]) / np.array(errs_l2[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            rates_hierch = np.log(
                np.array(errs_hierch[1:]) /
                np.array(errs_hierch[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
        else:
            rates_l2 = []
            rates_hierch = []

        print(
            '\ndofs={}\nerrs_l2={}\nerr_hierch={}\n\nrates_l2={}\nrates_hierch={}\n------'
            .format(dofs, errs_l2, errs_hierch, rates_l2, rates_hierch))
        mesh.uniform_refine()
