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


def SL_mat_col(j):
    elem_trial = __elems_trial[j]
    col = np.zeros(len(__elems_test))
    for i, elem_test in enumerate(__elems_test):
        if elem_test.time_interval[1] <= elem_trial.time_interval[0]:
            continue
        col[i] = __SL.bilform(elem_trial, elem_test)
    return col


def IP_rhs(j):
    return __M0.linform(__elems_test[j])[0]


def SL_matrix(elems_test, elems_trial):
    """ Evaluate the single layer matrix in parallel. """
    N = len(elems_test)
    M = len(elems_trial)

    # For small N, M, simply do it inline.
    if N < 5 and M < 5:
        mat = np.zeros((N, M))
        for i, elem_test in enumerate(elems_test):
            for j, elem_trial in enumerate(elems_trial):
                mat[i, j] = SL.bilform(elem_trial, elem_test)
        return mat

    md5 = hashlib.md5(
        (str(elems_test) + str(elems_trial)).encode()).hexdigest()
    cache_SL_fn = "{}/SL_{}x{}_{}.npy".format('data', N, M, md5)
    if os.path.isfile(cache_SL_fn):
        mat = np.load(cache_SL_fn)
        print("Loaded Single Layer from file {}".format(cache_SL_fn))
        return mat

    time_mat_begin = time.time()
    mat = np.zeros((N, M))

    # Set up global variables for parallelizing.
    global __elems_test, __elems_trial, __SL
    __elems_test = elems_test
    __elems_trial = elems_trial

    __SL = SL
    for j, col in enumerate(mp.Pool(N_procs).imap(SL_mat_col, range(M), 10)):
        mat[:, j] = col

    #mat_coarse = np.zeros((N, M))
    #__SL = SL_coarse
    #for j, col in enumerate(mp.Pool(N_procs).imap(SL_mat_col, range(M))):
    #    mat_coarse[:, j] = col

    #err = np.abs((mat - mat_coarse) / mat)
    #max_ij = np.unravel_index(np.nanargmax(err), err.shape)
    #print('---')
    #print('SL Max rel error', np.nanmax(err), 'for', elems_test[max_ij[0]],
    #      ' vs ', elems_trial[max_ij[1]])
    #print('SL Min rel error', np.nanmin(err))
    #print('---')

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
    N = len(elems)
    md5 = hashlib.md5(str(elems).encode()).hexdigest()
    cache_M0_fn = "{}/M0_dofs_{}_{}.npy".format('data', N, md5)
    if os.path.isfile(cache_M0_fn):
        print("Loaded Initial Operator from file {}".format(cache_M0_fn))
        return -np.load(cache_M0_fn)

    time_rhs_begin = time.time()
    global __elems_test, __M0
    __elems_test = elems
    __M0 = M0
    M0_u0 = np.array(mp.Pool(N_procs).map(IP_rhs, range(N), 10))
    #__M0 = M0_coarse
    #M0_u0_coarse = np.array(mp.Pool(N_procs).map(IP_rhs, range(N)))
    #err = np.abs((M0_u0 - M0_u0_coarse) / M0_u0)
    #print('---')
    #print('IP Max rel error', np.max(err), 'for', elems[np.argmax(err)])
    #print('IP Min rel error', np.min(err), 'for', elems[np.argmin(err)])
    #print('---')

    np.save(cache_M0_fn, M0_u0)
    print('Calculating initial potential took {}s'.format(time.time() -
                                                          time_rhs_begin))
    print("Stored Initial Operator to {}".format(cache_M0_fn))
    return -M0_u0


class DummyElement:
    """ Needed for calculation of the Hierarchical Error Estimator. """
    def __init__(self, vertices, parent):
        self.vertices = vertices
        self.parent = parent
        self.gamma_space = parent.gamma_space

        self.time_interval = float(self.vertices[0].t), float(
            self.vertices[2].t)
        self.space_interval = float(self.vertices[0].x), float(
            self.vertices[2].x)
        self.h_t = float(abs(self.vertices[2].t - self.vertices[0].t))
        self.h_x = float(abs(self.vertices[2].x - self.vertices[0].x))

    def __repr__(self):
        return "Elem(t={}, x={})".format(self.time_interval,
                                         self.space_interval)


def HierarchicalErrorEstimator(Phi, elems_coarse, SL, RHS):
    """ Returns the hierarchical basis estimator for given function. """

    # Calcualte uniform refinement of the mesh.
    elem_2_children = []
    for elem_coarse in elems_coarse:
        v0, v1, v2, v3 = elem_coarse.vertices
        v01 = Vertex(t=(v0.t + v1.t) / 2, x=(v0.x + v1.x) / 2, idx=-1)
        v12 = Vertex(t=(v1.t + v2.t) / 2, x=(v1.x + v2.x) / 2, idx=-1)
        v23 = Vertex(t=(v2.t + v3.t) / 2, x=(v2.x + v3.x) / 2, idx=-1)
        v30 = Vertex(t=(v3.t + v0.t) / 2, x=(v3.x + v0.x) / 2, idx=-1)
        vi = Vertex(t=(v0.t + v2.t) / 2, x=(v0.x + v2.x) / 2, idx=-1)

        children = [
            DummyElement(vertices=[v0, v01, vi, v30], parent=elem_coarse),
            DummyElement(vertices=[v01, v1, v12, vi], parent=elem_coarse),
            DummyElement(vertices=[v30, vi, v23, v3], parent=elem_coarse),
            DummyElement(vertices=[vi, v12, v2, v23], parent=elem_coarse),
        ]

        elem_2_children.append(children)

    # Flatten list and calculate mapping of indices.
    elems_fine = [child for children in elem_2_children for child in children]
    elem_2_idx_fine = {k: v for v, k in enumerate(elems_fine)}

    # Evaluate SL matrix tested with the fine mesh.
    mat = SL(elems_fine, elems_coarse)
    # TEST THIS MATRIX WITH SMALLER QUADRATURE.
    VPhi = mat @ Phi

    # Checking quadrature!
    #mat_coarse = SL(elems_coarse, elems_coarse)
    #print("Shape of mat_coarse is {}".format(mat_coarse.shape))
    #for i, elem_test in enumerate(elems_coarse):
    #    for j, elem_trial in enumerate(elems_coarse):
    #        val_fine = 0
    #        for child_test in elem_2_children[i]:
    #            k = elem_2_idx_fine[child_test]
    #            val_fine += mat[k, j]
    #            #print('\t', elem_trial, child_test, mat[k, j])
    #        #print(elem_trial, elem_test,
    #        #      abs((mat_coarse[i, j] - val_fine) / val_fine))
    #        assert mat_coarse[i, j] == approx(val_fine, abs=0, rel=1e-10)

    # Evaluate the RHS on the fine mesh.
    rhs = RHS(elems_fine)
    # TEST THIS VECTOR WITH SMALLER QUADRATURE.

    estim = np.zeros(len(elems_coarse))
    for i, elem_coarse in enumerate(elems_coarse):
        S = SL(elem_2_children[i], elem_2_children[i])
        children = [elem_2_idx_fine[elem] for elem in elem_2_children[i]]
        #scaling = sum(mat[j, i] for j in children)

        estim_loc = np.zeros(3)
        for k, coefs in enumerate([[1, 1, -1, -1], [1, -1, 1, -1],
                                   [1, -1, -1, 1]]):
            rhs_estim = 0
            V_estim = 0
            for j, c in zip(children, coefs):
                rhs_estim += rhs[j] * c
                V_estim += VPhi[j] * c
            coefs = np.array(coefs)
            scaling_estim = coefs @ (S @ coefs.T)
            estim_loc[k] = abs(rhs_estim - V_estim)**2 / scaling_estim
        estim[i] = np.sum(estim_loc)

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


def error_estim_l2(i):
    global elems_glob
    global error_estimator
    global residual
    return error_estimator.WeightedL2(elems_glob[i], residual)


def error_estim_slo(i):
    global elems_glob
    global error_estimator
    global residual
    return error_estimator.Sobolev(elems_glob[i], residual)


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

    for k in range(100):
        elems = list(mesh.leaf_elements)
        N = len(mesh.leaf_elements)
        print('Loop with {} dofs'.format(N))
        print(mesh.gmsh(use_gamma=True),
              file=open("./data/adaptive_N_{}_{}.gmsh".format(N, mesh.md5()),
                        "w"))
        dofs.append(N)

        # Calculate SL matrix.
        mat = SL_matrix(elems, elems)

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
        err_tot, eta_sqr = HierarchicalErrorEstimator(Phi, elems, SL_matrix,
                                                      RHS_vector)
        errs_hierch.append(err_tot)
        print('Hierarchical error estimator took {}s'.format(
            time.time() - time_hierarch_begin))

        # Calculate the weighted l2 error of the residual set global vars.
        def residual(t, x_hat, x):
            assert len(t) == len(x_hat) == x.shape[1]
            result = np.zeros(len(t))
            for i, (t, x_hat, x) in enumerate(zip(t, x_hat, x.T)):
                # Evaluate the SL for our trial function.
                VPhi = 0
                for j, elem_trial in enumerate(elems_glob):
                    VPhi += Phi[j] * SL.evaluate(elem_trial, t, x_hat,
                                                 x.reshape(2, 1))

                # Compare with rhs.
                result[i] = VPhi + M0u0(t, x.reshape(2, 1))
            return result

        time_l2_begin = time.time()
        elems_glob = elems
        err_estim_sqr = np.array(
            mp.Pool(N_procs).map(error_estim_l2, range(N), 10))
        errs_estim.append(np.sqrt(np.sum(err_estim_sqr)))
        print('Error estimation of weighted residual took {}s'.format(
            time.time() - time_l2_begin))

        time_slo_begin = time.time()
        elems_glob = elems
        err_slo_sqr = np.array(
            mp.Pool(N_procs).map(error_estim_slo, range(N), 10))
        errs_slo.append(np.sqrt(np.sum(err_slo_sqr)))
        print('Error estimation of Slobodeckij norm took {}s'.format(
            time.time() - time_slo_begin))

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

        print('Dorfler marking with theta = {}'.format(theta))
        s_idx = list(reversed(np.argsort(eta_sqr)))
        eta_tot_sqr = np.sum(eta_sqr)
        cumsum = 0.0
        marked = []
        for i in s_idx:
            marked.append(elems[i])
            cumsum += eta_sqr[i]
            if cumsum >= eta_tot_sqr * theta**2:
                break
        assert np.sqrt(cumsum) >= theta * err_tot

        print('Marked {} / {} elements'.format(len(marked), N))
        # First refine the coarse elements.
        marked.sort(key=lambda elem: elem.levels)
        for elem in marked:
            assert not elem.children
            mesh.refine(elem)
        print('After refinement {} elements'.format(len(mesh.leaf_elements)))
