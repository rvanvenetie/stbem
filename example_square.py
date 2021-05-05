import numpy as np
from math import sqrt
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
    cache_dir = 'data'
    problem = 'UnitSquare_Smooth'
    print('Running parallel with {} threads.'.format(N_procs))

    mesh = MeshParametrized(UnitSquare())
    theta = 0.9
    M0 = InitialOperator(bdr_mesh=mesh,
                         u0=u0,
                         initial_mesh=UnitSquareBoundaryRefined,
                         cache_dir=cache_dir,
                         problem=problem)
    SL = SingleLayerOperator(mesh, cache_dir=cache_dir)

    dofs = []
    errs_trace = []
    errs_unweighted_l2 = []
    errs_weighted_l2 = []
    errs_slo = []
    errs_hierch = []
    error_estimator = ErrorEstimator(mesh,
                                     N_poly=5,
                                     cache_dir=cache_dir,
                                     problem=problem)
    hierarch_error_estimator = HierarchicalErrorEstimator(SL=SL, M0=M0)

    for k in range(100):
        elems = list(mesh.leaf_elements)
        N = len(elems)
        md5 = hashlib.md5(
            (str(mesh.gamma_space) + str(elems)).encode()).hexdigest()
        print('Loop with {} dofs'.format(N))
        print(mesh.gmsh(use_gamma=True),
              file=open(
                  "./{}/mesh_{}_{}_{}.gmsh".format(cache_dir, problem, N, md5),
                  "w"))
        dofs.append(N)

        # Calculate SL matrix.
        mat = SL.bilform_matrix(elems, elems, use_mp=True)

        # Calculate initial potential.
        rhs = -M0.linform_vector(elems=elems, use_mp=True)

        # Solve.
        time_solve_begin = time.time()
        Phi = np.linalg.solve(mat, rhs)
        print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

        # Estimate the l2 error of the neumann trace.
        time_trace_begin = time.time()
        gauss_2d = ProductScheme2D(gauss_quadrature_scheme(11))
        err_trace = []
        for i, elem in enumerate(elems):
            err = lambda tx: (Phi[i] - u_neumann(tx[0], tx[1]))**2
            err_trace.append(
                gauss_2d.integrate(err, *elem.time_interval,
                                   *elem.space_interval))
        errs_trace.append(np.sqrt(math.fsum(err_trace)))
        print('Error estimation of \Phi - \partial_n took {}s'.format(
            time.time() - time_trace_begin))

        # Do the hierarhical error estimator.
        time_hierarch_begin = time.time()
        hierarch = hierarch_error_estimator.estimate(elems, Phi)
        print('\nHierarch\t time: {}\t space: {}\t'.format(
            np.sum(hierarch[:, 0]), np.sum(hierarch[:, 1])))
        np.save('{}/hierarch_{}_{}_{}.npy'.format(cache_dir, N, problem, md5),
                hierarch)
        errs_hierch.append(np.sqrt(np.sum(hierarch)))
        print('Hierarchical error estimator took {}s\n'.format(
            time.time() - time_hierarch_begin))

        # Calculate the weighted l2 + sobolev error of the residual.
        residual = error_estimator.residual(elems, Phi, SL, M0u0)

        time_begin = time.time()
        weighted_l2 = error_estimator.estimate_weighted_l2(elems,
                                                           residual,
                                                           use_mp=True)
        print('Weighted L2\t time: {}\t space: {}\t'.format(
            np.sum(weighted_l2[:, 0]), np.sum(weighted_l2[:, 1])))
        errs_weighted_l2.append(np.sqrt(np.sum(weighted_l2)))

        # Calculate the _unweighted_ l2 error.
        err_unweighted_l2 = 0
        for i, elem in enumerate(elems):
            err_unweighted_l2 += sqrt(elem.h_t) * weighted_l2[i, 0]
            err_unweighted_l2 += elem.h_x * weighted_l2[i, 1]
        errs_unweighted_l2.append(sqrt(err_unweighted_l2))

        print('Error estimation of weighted residual took {}s\n'.format(
            time.time() - time_begin))

        time_begin = time.time()
        sobolev = error_estimator.estimate_sobolev(elems,
                                                   residual,
                                                   use_mp=True)
        print('Sobolev\t time: {}\t space: {}\t'.format(
            np.sum(sobolev[:, 0]), np.sum(sobolev[:, 1])))
        errs_slo.append(np.sqrt(np.sum(sobolev)))
        print(
            'Error estimation of Slobodeckij normtook {}s'.format(time.time() -
                                                                  time_begin))

        if k:
            rates_unweighted_l2 = np.log(
                np.array(errs_unweighted_l2[1:]) /
                np.array(errs_unweighted_l2[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            rates_weighted_l2 = np.log(
                np.array(errs_weighted_l2[1:]) /
                np.array(errs_weighted_l2[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            rates_slo = np.log(
                np.array(errs_slo[1:]) / np.array(errs_slo[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            rates_trace = np.log(
                np.array(errs_trace[1:]) / np.array(errs_trace[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
            rates_hierch = np.log(
                np.array(errs_hierch[1:]) /
                np.array(errs_hierch[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
        else:
            rates_slo = []
            rates_trace = []
            rates_hierch = []
            rates_weighted_l2 = []
            rates_unweighted_l2 = []

        print(
            '\ndofs={}\nerrs_trace={}\nerr_hierch={}\nerr_unweighted_l2={}\nerr_weighted_l2={}\nerrs_slo={}\n\nrates_trace={}\nrates_hierch={}\nrates_unweighted_l2={}\nrates_weighted_l2={}\nrates_slo={}\n------'
            .format(dofs, errs_trace, errs_hierch, errs_unweighted_l2,
                    errs_weighted_l2, errs_slo, rates_trace, rates_hierch,
                    rates_unweighted_l2, rates_weighted_l2, rates_slo))
        #mesh.dorfler_refine_isotropic(np.sum(hierarch, axis=1), theta)
        mesh.dorfler_refine_anisotropic(sobolev, theta)
