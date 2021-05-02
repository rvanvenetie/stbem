import numpy as np
from copy import deepcopy
from mesh import Vertex, Element
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
from error_estimator import ErrorEstimator
from hierarchical_error_estimator import HierarchicalErrorEstimator


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
    theta = 0.7
    M0 = InitialOperator(bdr_mesh=mesh,
                         u0=u0,
                         initial_mesh=UnitSquareBoundaryRefined)
    SL = SingleLayerOperator(mesh)

    dofs = []
    errs_l2 = []
    errs_estim = []
    errs_slo = []
    errs_hierch = []
    error_estimator = ErrorEstimator(mesh, N_poly=7)
    hierarch_error_estimator = HierarchicalErrorEstimator(SL=SL, M0=M0)

    for k in range(100):
        elems = list(mesh.leaf_elements)
        N = len(mesh.leaf_elements)
        md5 = hashlib.md5(
            (str(mesh.gamma_space) + str(elems)).encode()).hexdigest()
        print('Loop with {} dofs'.format(N))
        print(mesh.gmsh(use_gamma=True),
              file=open(
                  "./data/adaptive_{}_{}_{}.gmsh".format(
                      mesh.gamma_space, N, mesh.md5()), "w"))
        dofs.append(N)

        # Calculate SL matrix.
        mat = SL.bilform_matrix(elems, elems, cache_dir='data', use_mp=True)

        # Calculate initial potential.
        rhs = -M0.linform_vector(elems=elems, cache_dir='data', use_mp=True)

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
        hierarch = hierarch_error_estimator.estimate(elems, Phi)
        print('Hierarch\t time: {}\t space: {}\t'.format(
            np.sum(hierarch[:, 0]), np.sum(hierarch[:, 1])))
        np.save('data/hierarch_{}_{}.npy'.format(N, md5), hierarch)
        errs_hierch.append(np.sqrt(np.sum(hierarch)))
        print('Hierarchical error estimator took {}s'.format(
            time.time() - time_hierarch_begin))

        # Calculate the weighted l2 + sobolev error of the residual.
        residual = error_estimator.residual(elems, Phi, SL, M0u0)

        time_begin = time.time()
        weighted_l2 = error_estimator.estimate_weighted_l2(elems,
                                                           residual,
                                                           use_mp=True)
        print('Weighted L2\t time: {}\t space: {}\t'.format(
            np.sum(weighted_l2[:, 0]), np.sum(weighted_l2[:, 1])))
        np.save('data/weighted_l2_{}_{}.npy'.format(N, md5), weighted_l2)
        errs_estim.append(np.sqrt(np.sum(weighted_l2)))
        print('Error estimation of weighted residual took {}s'.format(
            time.time() - time_begin))

        time_begin = time.time()
        sobolev = error_estimator.estimate_sobolev(elems,
                                                   residual,
                                                   use_mp=True)
        print('Sobolev\t time: {}\t space: {}\t'.format(
            np.sum(sobolev[:, 0]), np.sum(sobolev[:, 1])))
        np.save('data/sobolev_{}_{}.npy'.format(N, md5), sobolev)
        errs_slo.append(np.sqrt(np.sum(sobolev)))
        print(
            'Error estimation of Slobodeckij normtook {}s'.format(time.time() -
                                                                  time_begin))

        if k:
            rates_estim = np.log(
                np.array(errs_estim[1:]) / np.array(errs_estim[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            rates_slo = np.log(
                np.array(errs_slo[1:]) / np.array(errs_slo[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            rates_l2 = np.log(
                np.array(errs_l2[1:]) / np.array(errs_l2[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            rates_hierch = np.log(
                np.array(errs_hierch[1:]) /
                np.array(errs_hierch[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
        else:
            rates_slo = []
            rates_l2 = []
            rates_hierch = []
            rates_estim = []

        print(
            '\ndofs={}\nerrs_l2={}\nerr_hierch={}\nerr_estim={}\nerrs_slo={}\n\nrates_l2={}\nrates_hierch={}\nrates_estim={}\nrates_slo={}\n------'
            .format(dofs, errs_l2, errs_hierch, errs_estim, errs_slo, rates_l2,
                    rates_hierch, rates_estim, rates_slo))
        #mesh.dorfler_refine_isotropic(np.sum(sobolev, axis=1), theta)
        mesh.dorfler_refine_anisotropic(sobolev, theta)
