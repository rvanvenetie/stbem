import numpy as np
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


mesh = MeshParametrized(UnitSquare())
M0 = InitialOperator(bdr_mesh=mesh,
                     u0=u0,
                     initial_mesh=UnitSquareBoundaryRefined)

pts_T = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
dofs = []
errs_l2 = []
errs_estim = []
errs_pointwise = [[] for _ in pts_T]
for k in range(10):
    elems_cur = list(mesh.leaf_elements)
    N = len(elems_cur)
    print('Loop with {} dofs'.format(N))
    SL = SingleLayerOperator(mesh)
    dofs.append(N)

    #time_mat_begin = time.time()
    #mat = SL.bilform_matrix(cache_dir='data')
    #print('Calculating matrix took {}s'.format(time.time() - time_mat_begin))

    cache_SL_fn = "{}/SL_fast_dofs_{}_{}.npy".format('data', N, mesh.md5())
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
    cache_M0_fn = "{}/M0_fast_dofs_{}_{}.npy".format('data', N, mesh.md5())
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

        err_estim_sqr[i] = (
            elem.h_t**(-2 * mu) + elem.h_x**(-2 * nu)) * gauss_2d.integrate(
                residual_squared, *elem.time_interval, *elem.space_interval)
    errs_estim.append(np.sqrt(np.sum(err_estim_sqr)))
    print('Error estimation of weighted residual of order {} took {}s'.format(
        err_order,
        time.time() - time_l2_begin))

    print('N={}\terr_estim={}'.format(N, errs_estim[-1]))
    print('N={}\terr_l2={}'.format(N, errs_l2[-1]))

    print('\ndofs={}\nerrs_l2={}\nerr_estim={}\nerrs_pointwise={}\n------'.
          format(dofs, errs_l2, errs_estim, errs_pointwise))
    mesh.uniform_refine()
