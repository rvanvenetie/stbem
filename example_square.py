import numpy as np
import matplotlib.pyplot as plt
import math
from mesh import Mesh, MeshParametrized
from initial_potential import InitialPotential
from initial_mesh import UnitSquareBoundaryRefined
import initial_mesh
from single_layer import SingleLayerOperator
import time
from parametrization import Circle, UnitSquare, LShape
from quadrature import gauss_quadrature_scheme, ProductScheme2D


def u(t, xy):
    return np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * xy[0]) * np.sin(
        np.pi * xy[1])


def u0(xy):
    return np.sin(np.pi * xy[0]) * np.sin(np.pi * xy[1])


def u_neumann(t, x_hat):
    """ evaluates the neumann trace along the lateral boundary. """
    return -np.pi * np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * (x_hat % 1))


mesh = MeshParametrized(UnitSquare())
IP = InitialPotential(bdr_mesh=mesh,
                      u0=u0,
                      initial_mesh=UnitSquareBoundaryRefined)

pts_T = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
dofs = []
errs_l2 = []
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
        np.save(cache_SL_fn, mat)
        print('Calculating matrix fast took {}s'.format(time.time() -
                                                        time_mat_begin))
        print("Stored Single Layer to {}".format(cache_SL_fn))

    # Calculate initial potential.
    time_rhs_begin = time.time()
    cache_IP_fn = "{}/IP_fast_dofs_{}_{}.npy".format('data', N, mesh.md5())
    try:
        M0_u0 = np.load(cache_IP_fn)
        print("Loaded Initial Potential from file {}".format(cache_IP_fn))
    except:
        calc_dict = {}
        M0_u0 = np.zeros(shape=N)
        for j, elem_test in enumerate(elems_cur):
            a = elem_test.space_interval[0] - math.floor(
                elem_test.space_interval[0])
            tup = (elem_test.time_interval[0], a)
            if not tup in calc_dict:
                calc_dict[tup] = IP.linform(elem_test)
            M0_u0[j], _ = calc_dict[tup]
        np.save(cache_IP_fn, mat)
        print('Calculating initial potential took {}s'.format(time.time() -
                                                              time_rhs_begin))
        print("Stored Initial Potential to {}".format(cache_IP_fn))

    # Solve.
    time_solve_begin = time.time()
    Phi_cur = np.linalg.solve(mat, -M0_u0)
    print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

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
    print('Error estimation took {}s'.format(time.time() - time_l2_begin))
    print('N={}\terr_l2={}'.format(N, err_l2))

    # Evaluate pointwise error in t, [0.5,0.5].
    print('N={} pointwise error'.format(N))
    for i, t in enumerate(pts_T):
        x = np.array([[0.5], [0.5]])
        val = np.dot(Phi_cur, SL.potential_vector(t, x)) + IP.evaluate(t, x)
        val_exact = u(t, [0.5, 0.5])
        err_rel = abs((val - val_exact) / val_exact)
        print('\t Relative error in ({}, {})={}'.format(
            t, x.flatten(), err_rel))
        errs_pointwise[i].append(err_rel)

    print('\ndofs={}\nerrs_l2={}\nerrs_pointwise={}\n------'.format(
        dofs, errs_l2, errs_pointwise))
    mesh.uniform_refine()
