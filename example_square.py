import numpy as np
import matplotlib.pyplot as plt
import math
from mesh import Mesh, MeshParametrized
from initial_potential import InitialPotential
from initial_mesh import UnitSquareBoundaryRefined
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

dofs = []
errs_l2 = []
for k in range(10):
    SL = SingleLayerOperator(mesh)
    time_mat_begin = time.time()
    mat = SL.bilform_matrix(cache_dir='data')
    elems_cur = list(mesh.leaf_elements)
    N = len(elems_cur)
    dofs.append(N)
    print('Loop with {} dofs'.format(N))
    print('Calculating matrix took {}s'.format(time.time() - time_mat_begin))

    # Calculate initial potential.
    time_rhs_begin = time.time()
    M0_u0 = IP.linform_vector()
    print('Calculating initial potential took {}s'.format(time.time() -
                                                          time_rhs_begin))

    # Solve.
    time_solve_begin = time.time()
    Phi_cur = np.linalg.solve(mat, -M0_u0)
    print('Solving matrix took {}s'.format(time.time() - time_solve_begin))

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
    print('\ndofs={}\nerrs_l2={}\n------'.format(dofs, errs_l2))
    mesh.uniform_refine()
