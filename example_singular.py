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


def u0(xy):
    return 1


def M0u0(t, xy):
    a = xy[0]
    b = xy[1]
    return (1 / 4) * (erf(
        (1 - a) / (2 * np.sqrt(t))) + erf(a / (2 * np.sqrt(t)))) * (erf(
            (1 - b) / (2 * np.sqrt(t))) + erf(b / (2 * np.sqrt(t))))


if __name__ == "__main__":
    N_procs = mp.cpu_count()
    mp.set_start_method('fork')
    print('Running parallel with {} threads.'.format(N_procs))

    mesh = MeshParametrized(UnitSquare())
    theta = 0.5
    M0 = InitialOperator(bdr_mesh=mesh,
                         u0=u0,
                         initial_mesh=UnitSquareBoundaryRefined)
    SL = SingleLayerOperator(mesh, pw_exact=True)

    dofs = []
    errs_unweighted_l2 = []
    errs_weighted_l2 = []
    errs_slo = []
    errs_hierch = []
    error_estimator = ErrorEstimator(mesh, N_poly=5)
    hierarch_error_estimator = HierarchicalErrorEstimator(SL=SL, M0=M0)

    for k in range(100):
        elems = list(mesh.leaf_elements)
        N = len(mesh.leaf_elements)
        md5 = hashlib.md5(
            (str(mesh.gamma_space) + str(elems)).encode()).hexdigest()
        print('Loop with {} dofs'.format(N))
        print(mesh.gmsh(use_gamma=True),
              file=open(
                  "./data/singular_{}_{}_{}.gmsh".format(
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

        # Do the hierarhical error estimator.
        time_hierarch_begin = time.time()
        hierarch = hierarch_error_estimator.estimate(elems, Phi)
        print('\nHierarch\t time: {}\t space: {}\t'.format(
            np.sum(hierarch[:, 0]), np.sum(hierarch[:, 1])))
        np.save('data/hierarch_{}_{}.npy'.format(N, md5), hierarch)
        errs_hierch.append(np.sqrt(np.sum(hierarch)))
        print('Hierarchical error estimator took {}s'.format(
            time.time() - time_hierarch_begin))

        # Calculate the weighted l2 + sobolev error of the residual.
        residual = error_estimator.residual_pw(elems, Phi, SL, M0u0)

        time_begin = time.time()
        try:
            weighted_l2 = np.load('data/weighted_l2_{}_{}.npy'.format(N, md5))
        except:
            weighted_l2 = error_estimator.estimate_weighted_l2(elems,
                                                               residual,
                                                               use_mp=True)
            print('\nWeighted L2\t time: {}\t space: {}\t'.format(
                np.sum(weighted_l2[:, 0]), np.sum(weighted_l2[:, 1])))
            np.save('data/weighted_l2_{}_{}.npy'.format(N, md5), weighted_l2)
        errs_weighted_l2.append(np.sqrt(np.sum(weighted_l2)))

        # Calculate the _unweighted_ l2 error.
        err_unweighted_l2 = 0
        for i, elem in enumerate(elems):
            err_unweighted_l2 += sqrt(elem.h_t) * weighted_l2[i, 0]
            err_unweighted_l2 += elem.h_x * weighted_l2[i, 1]
        errs_unweighted_l2.append(sqrt(err_unweighted_l2))

        print('Error estimation of weighted residual took {}s'.format(
            time.time() - time_begin))

        time_begin = time.time()
        try:
            sobolev = np.load('data/sobolev_{}_{}.npy'.format(N, md5))
        except:
            sobolev = error_estimator.estimate_sobolev(elems,
                                                       residual,
                                                       use_mp=True)
            print('\nSobolev\t time: {}\t space: {}\t'.format(
                np.sum(sobolev[:, 0]), np.sum(sobolev[:, 1])))
            np.save('data/sobolev_{}_{}.npy'.format(N, md5), sobolev)
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
            rates_hierch = np.log(
                np.array(errs_hierch[1:]) /
                np.array(errs_hierch[:-1])) / np.log(
                    np.array(dofs[1:]) / np.array(dofs[:-1]))
        else:
            rates_slo = []
            rates_hierch = []
            rates_weighted_l2 = []
            rates_unweighted_l2 = []

        print(
            '\ndofs={}\nerr_hierch={}\nerr_unweighted_l2={}\nerr_weighted_l2={}\nerrs_slo={}\n\nrates_hierch={}\nrates_unweighted_l2={}\nrates_weighted_l2={}\nrates_slo={}\n------'
            .format(dofs, errs_hierch, errs_unweighted_l2, errs_weighted_l2,
                    errs_slo, rates_hierch, rates_unweighted_l2,
                    rates_weighted_l2, rates_slo))
        #mesh.dorfler_refine_isotropic(np.sum(hierarch, axis=1), theta)
        mesh.dorfler_refine_anisotropic(sobolev, theta)
