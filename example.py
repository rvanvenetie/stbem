import multiprocessing as mp
from quadrature import gauss_quadrature_scheme, ProductScheme2D
from single_layer import SingleLayerOperator
from initial_potential import InitialOperator
from error_estimator import ErrorEstimator
import hashlib
import time
import numpy as np
from math import sqrt, fsum
from hierarchical_error_estimator import HierarchicalErrorEstimator
from mesh import MeshParametrized
from parametrization import PiSquare, UnitSquare, LShape
from initial_mesh import UnitSquareBoundaryRefined, LShapeBoundaryRefined, PiSquareBoundaryRefined
import argparse
from problems import problem_helper
from pprint import pprint


def calc_rate(dofs, errs):
    if len(dofs) < 2: return []
    assert len(dofs) == len(errs)
    return np.log(np.array(errs[1:]) / np.array(errs[:-1])) / np.log(
        np.array(dofs[1:]) / np.array(dofs[:-1]))


if __name__ == '__main__':
    N_procs = mp.cpu_count()
    mp.set_start_method('fork')
    cache_dir = 'data_exact'
    print('Running parallel with {} threads.'.format(N_procs))

    parser = argparse.ArgumentParser(
        description='Solve parabolic equation using ngsolve.')
    parser.add_argument('--problem',
                        default='Smooth',
                        help='problem (Smooth, Singular, Dirichlet)')
    parser.add_argument('--domain',
                        default='UnitSquare',
                        help='domain (UnitSquare, PiSquare, LShape)')
    parser.add_argument('--hierarchical',
                        default=True,
                        help='Calculate the hierarchical error estim')
    parser.add_argument('--sobolev',
                        default=True,
                        help='Calculate the sobolev error estim')
    parser.add_argument('--l2',
                        default=True,
                        help='Calculate the l2 error estim')
    parser.add_argument('--refinement',
                        default='uniform',
                        help='refinement (uniform, isotropic, anisotropic)')
    parser.add_argument(
        '--estimator',
        default='sobolev',
        help='estimator for marking (hierarchical, sobolev, sobolev-l2)')
    parser.add_argument('--theta',
                        default=0.9,
                        type=float,
                        help='theta used for adaptive refinement')
    parser.add_argument('--estimator-quadrature',
                        default='5355',
                        help='Quadrature order used for the error estimator.')
    args = parser.parse_args()

    print('Arguments:')
    pprint(vars(args))

    assert args.refinement in ['uniform', 'isotropic', 'anisotropic']
    assert args.estimator in ['sobolev', 'hierarchical', 'sobolev-l2']
    assert 0 < args.theta < 1
    assert len(args.estimator_quadrature) == 4

    # Create bdr and initial mesh.
    if args.domain == 'UnitSquare':
        mesh = MeshParametrized(UnitSquare())
        initial_mesh = UnitSquareBoundaryRefined
    elif args.domain == 'PiSquare':
        mesh = MeshParametrized(PiSquare())
        initial_mesh = PiSquareBoundaryRefined
    elif args.domain == 'LShape':
        mesh = MeshParametrized(LShape())
        initial_mesh = LShapeBoundaryRefined

        # We must divide the initial mesh in space on the long sides
        # for the initial mesh to coincide.
        elems = list(mesh.leaf_elements)
        for elem in elems:
            if elem.h_x > 1: mesh.refine_space(elem)
    else:
        raise Exception('Invalid domain: {}'.format(args.domain))

    # Retrieve problem dependent data.
    data = problem_helper(args.problem, args.domain)
    problem = '{}_{}'.format(args.domain, args.problem)

    # Create SL.
    SL = SingleLayerOperator(mesh, cache_dir=cache_dir, pw_exact=True)

    # Create M0 if u0 != 0 required.
    if 'u0' in data:
        M0 = InitialOperator(bdr_mesh=mesh,
                             u0=data['u0'],
                             initial_mesh=initial_mesh,
                             cache_dir=cache_dir,
                             problem=problem)
        M0u0 = data['M0u0']
    else:
        M0 = None
        M0u0 = None

    # Set g_linform if g != 0.
    if 'g' in data:
        g = data['g']
        g_linform = data['g-linform']
    else:
        g = None
        g_linform = None

    # Create error estimators.
    error_estimator = ErrorEstimator(
        mesh,
        N_poly=tuple(int(x) for x in args.estimator_quadrature),
        cache_dir=cache_dir,
        problem=problem)
    hierarch_error_estimator = HierarchicalErrorEstimator(SL=SL,
                                                          M0=M0,
                                                          g=g_linform)

    dofs = []
    errs_trace = []

    errs_unweighted_l2 = []
    errs_weighted_l2 = []
    errs_weighted_l2_time = []
    errs_weighted_l2_space = []
    errs_slo = []
    errs_slo_time = []
    errs_slo_space = []
    errs_hierch = []

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

        # Calculate RHS.
        rhs = np.zeros(N)
        if M0:
            rhs = -M0.linform_vector(elems=elems, use_mp=True)
        if g_linform:
            rhs += g_linform(elems)

        # Solve.
        time_solve_begin = time.time()
        Phi = np.linalg.solve(mat, rhs)
        print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

        # Estimate the l2 error of the neumann trace.
        if 'u-trace' in data:
            u_neumann = data['u-trace']
            time_trace_begin = time.time()
            gauss_2d = ProductScheme2D(gauss_quadrature_scheme(11))
            err_trace = []
            for i, elem in enumerate(elems):
                err = lambda tx: (Phi[i] - u_neumann(tx[0], tx[1]))**2
                err_trace.append(
                    gauss_2d.integrate(err, *elem.time_interval,
                                       *elem.space_interval))
            errs_trace.append(sqrt(fsum(err_trace)))
            print('Error estimation of \Phi - \partial_n took {}s'.format(
                time.time() - time_trace_begin))
        else:
            errs_trace.append(0)

        # Do the hierarhical error estimator.
        if args.hierarchical:
            time_hierarch_begin = time.time()
            hierarch = hierarch_error_estimator.estimate(elems, Phi)
            print('\nHierarch\t time: {}\t space: {}\t'.format(
                np.sum(hierarch[:, 0]), np.sum(hierarch[:, 1])))
            np.save(
                '{}/hierarch_{}_{}_{}.npy'.format(cache_dir, N, problem, md5),
                hierarch)
            errs_hierch.append(np.sqrt(np.sum(hierarch)))
            print('Hierarchical error estimator took {}s\n'.format(
                time.time() - time_hierarch_begin))
        else:
            errs_hierch.append(0)

        # Calculate the weighted l2 + sobolev error of the residual.
        residual = error_estimator.residual_pw(elems, Phi, SL, M0u0, g)

        if args.l2:
            time_begin = time.time()
            weighted_l2 = error_estimator.estimate_weighted_l2(elems,
                                                               residual,
                                                               use_mp=True)
            print('Weighted L2\t time: {}\t space: {}\t'.format(
                np.sum(weighted_l2[:, 0]), np.sum(weighted_l2[:, 1])))
            errs_weighted_l2_time.append(np.sqrt(np.sum(weighted_l2[:, 0])))
            errs_weighted_l2_space.append(np.sqrt(np.sum(weighted_l2[:, 1])))
            errs_weighted_l2.append(np.sqrt(np.sum(weighted_l2)))
            print('Error estimation of weighted residual took {}s\n'.format(
                time.time() - time_begin))

            # Calculate the _unweighted_ l2 error.
            err_unweighted_l2 = 0
            for i, elem in enumerate(elems):
                err_unweighted_l2 += sqrt(elem.h_t) * weighted_l2[i, 0]
                err_unweighted_l2 += elem.h_x * weighted_l2[i, 1]
            errs_unweighted_l2.append(sqrt(err_unweighted_l2))
        else:
            errs_weighted_l2.append(0)
            errs_unweighted_l2.append(0)

        if args.sobolev:
            time_begin = time.time()
            sobolev = error_estimator.estimate_sobolev(elems,
                                                       residual,
                                                       use_mp=True)
            print('Sobolev\t time: {}\t space: {}\t'.format(
                np.sum(sobolev[:, 0]), np.sum(sobolev[:, 1])))
            errs_slo.append(np.sqrt(np.sum(sobolev)))
            errs_slo_time.append(np.sqrt(np.sum(sobolev[:, 0])))
            errs_slo_space.append(np.sqrt(np.sum(sobolev[:, 1])))
            print('Error estimation of Slobodeckij normtook {}s'.format(
                time.time() - time_begin))
        else:
            errs_slo.append(0)

        rates_unweighted_l2 = calc_rate(dofs, errs_unweighted_l2)
        rates_weighted_l2 = calc_rate(dofs, errs_weighted_l2)
        rates_slo = calc_rate(dofs, errs_slo)
        rates_trace = calc_rate(dofs, errs_trace)
        rates_hierch = calc_rate(dofs, errs_hierch)

        print(
            '\ndofs={}\nerrs_trace={}\nerr_hierch={}\nerr_unweighted_l2={}\nerr_weighted_l2={}\nerrs_slo={}\n\nerrs_weighted_l2_time={}\nerrs_weighted_l2_space={}\nerrs_slo_time={}\nerrs_slo_space={}\n\nrates_trace={}\nrates_hierch={}\nrates_unweighted_l2={}\nrates_weighted_l2={}\nrates_slo={}\n------'
            .format(dofs, errs_trace, errs_hierch, errs_unweighted_l2,
                    errs_weighted_l2, errs_slo, errs_weighted_l2_time,
                    errs_weighted_l2_space, errs_slo_time, errs_slo_space,
                    rates_trace, rates_hierch, rates_unweighted_l2,
                    rates_weighted_l2, rates_slo))

        # Find the correct estimator for marking.
        if args.estimator == 'hierarchical':
            assert args.hierarchical
            eta = hierarchical
        elif args.estimator == 'sobolev':
            assert args.sobolev
            eta = sobolev
        elif args.estimator == 'sobolev-l2':
            assert args.sobolev and args.l2
            eta = sobolev + weighted_l2
        else:
            assert False

        # Refine the mesh.
        if args.refinement == 'uniform':
            mesh.uniform_refine()
        elif args.refinement == 'isotropic':
            mesh.dorfler_refine_isotropic(np.sum(eta, axis=1), args.theta)
        elif args.refinement == 'anisotropic':
            mesh.dorfler_refine_anisotropic(eta, args.theta)
