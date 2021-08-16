import multiprocessing as mp
import random

import numpy as np
from pytest import approx

from .error_estimator import ErrorEstimator
from .mesh import MeshParametrized
from .parametrization import UnitSquare


def test_error_estimator_l2():
    mesh = MeshParametrized(UnitSquare())
    random.seed(5)
    for _ in range(100):
        elem = random.choice(list(mesh.leaf_elements))
        mesh.refine_axis(elem, random.random() < 0.5)

    def residual(t, x_hat, gamma):
        return np.sqrt(t) * np.sin(x_hat)

    error_estim = ErrorEstimator(mesh)
    for elem in mesh.leaf_elements:
        a, b = elem.time_interval
        c, d = elem.space_interval
        val = (1 / 4) * (a - b) * (a + b) * (
            1 / np.sqrt(-a + b) + 1 /
            (-c + d)) * (c - d - np.cos(c) * np.sin(c) + np.cos(d) * np.sin(d))

        assert np.sum(error_estim.weighted_l2(elem,
                                              residual)) == approx(val,
                                                                   abs=0,
                                                                   rel=1e-4)


def test_error_estimator_slo_unif():
    mesh = MeshParametrized(UnitSquare())
    random.seed(5)
    mesh.uniform_refine()
    elems = list(mesh.leaf_elements)

    def residual(t, x_hat, gamma):
        x = gamma(x_hat)
        return t * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])

    # Evaluate space norm around the corner.
    val_exact = 0.12136521710154857053
    rel_error = 1
    for N in range(1, 21, 2):
        error_estimator = ErrorEstimator(mesh, N_poly=N)
        _, ips = error_estimator.sobolev_space(elems[1], residual)
        assert ips[2][1] == 0
        q_f = ips[1][1]
        new_rel_error = abs((q_f - val_exact) / val_exact)
        assert new_rel_error < rel_error or new_rel_error < 1e-8
        print('tcossin', N, new_rel_error, q_f)
        rel_error = new_rel_error
    assert rel_error < 1e-10
    print('')

    # Evaluate the space norm on the x = 1 pane.
    val_exact = 0.80522075921093728426
    rel_error = 10
    for N in range(1, 21, 2):
        error_estimator = ErrorEstimator(mesh, N_poly=N)
        _, ips = error_estimator.sobolev_space(elems[6], residual)
        q_f = ips[1][1]
        new_rel_error = abs((q_f - val_exact) / val_exact)
        assert new_rel_error < rel_error or new_rel_error < 1e-8
        print('tcossin', N, new_rel_error, q_f)
        rel_error = new_rel_error
    assert rel_error < 1e-13
    print('')

    # Evaluate the time norm.
    val_exact = 0.13333333333333671732
    rel_error = 3
    for N in range(1, 21, 2):
        error_estimator = ErrorEstimator(mesh, N_poly=N)
        _, ips = error_estimator.sobolev_time(elems[4], residual)
        q_f = ips[1][1]
        new_rel_error = abs((q_f - val_exact) / val_exact)
        assert new_rel_error < rel_error or new_rel_error < 1e-8
        print('tcossin', N, new_rel_error, q_f)
        rel_error = new_rel_error
    assert rel_error < 1e-13
    print('')

    def residual(t, x_hat, gamma):
        x = gamma(x_hat)
        return np.sin(np.pi * t) * x[0] * x[1]

    val_exact = 0.047727590109465093396
    rel_error = 10
    for N in range(3, 21, 2):
        error_estimator = ErrorEstimator(mesh, N_poly=N)
        _, ips = error_estimator.sobolev_time(elems[4], residual)
        q_f = ips[1][1]
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('sinxy', N, new_rel_error, q_f)
        assert new_rel_error < rel_error or new_rel_error < 1e-8
        rel_error = new_rel_error
    assert rel_error < 1e-12


def test_error_estimator_slo():
    mesh = MeshParametrized(UnitSquare())
    random.seed(5)
    for _ in range(100):
        elem = random.choice(list(mesh.leaf_elements))
        mesh.refine_axis(elem, random.random() < 0.5)

    def residual(t, x_hat, gamma):
        return t * x_hat

    error_estimator = ErrorEstimator(mesh, N_poly=5)
    space_cache = {}
    time_cache = {}
    elems = list(mesh.leaf_elements)
    idx_2_elem = {elem.glob_idx: elem for i, elem in enumerate(elems)}
    for elem in mesh.leaf_elements:
        err, ips = error_estimator.sobolev_space(elem, residual)
        for elem_nbr, val in ips:
            # Check symmetry.
            if (elem_nbr, elem) in space_cache:
                assert val == space_cache[elem_nbr, elem]

            space_cache[elem, elem_nbr] = val
            elem_nbr = idx_2_elem[elem_nbr]
            t_a = max(elem.time_interval[0], elem_nbr.time_interval[0])
            t_b = min(elem.time_interval[1], elem_nbr.time_interval[1])
            assert t_a < t_b
            if elem is elem_nbr:
                val_exact = -(1 / 3) * (elem.h_x)**2 * (t_a**3 - t_b**3)
                assert val == approx(val_exact, abs=0, rel=1e-10)
            elif elem.gamma_space == elem_nbr.gamma_space:
                val_exact = -(1 / 3) * (elem.h_x + elem_nbr.h_x)**2 * (t_a**3 -
                                                                       t_b**3)
                assert val == approx(val_exact, abs=0, rel=1e-10)

        err, ips = error_estimator.sobolev_time(elem, residual)
        for elem_nbr, val in ips:
            # Check symmetry.
            if (elem_nbr, elem) in time_cache:
                assert val == time_cache[elem_nbr, elem]
            time_cache[elem, elem_nbr] = val
            elem_nbr = idx_2_elem[elem_nbr]

            # Intersection
            x_a = max(elem.space_interval[0], elem_nbr.space_interval[0])
            x_b = min(elem.space_interval[1], elem_nbr.space_interval[1])
            assert x_a < x_b
            if elem is elem_nbr:
                val_exact = -(8 / 45) * (x_a**3 - x_b**3) * (elem.h_t)**(5 / 2)
                assert val == approx(val_exact, abs=0, rel=1e-10)
            elif elem.gamma_space == elem_nbr.gamma_space:
                val_exact = -(8 / 45) * (x_a**3 - x_b**3) * (
                    elem.h_t + elem_nbr.h_t)**(5 / 2)
                assert val == approx(val_exact, abs=0, rel=1e-10)


def test_error_estimator_symmetry():
    mesh = MeshParametrized(UnitSquare())
    random.seed(5)
    for _ in range(300):
        elem = random.choice(list(mesh.leaf_elements))
        mesh.refine_axis(elem, random.random() < 0.5)

    elems = list(mesh.leaf_elements)

    def residual(t, x_hat, gamma):
        x = gamma(x_hat)
        return t * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])

    error_estimator = ErrorEstimator(mesh, N_poly=5)

    mp.set_start_method('fork')
    errs = error_estimator.estimate_sobolev(elems, residual, use_mp=True)
    for i, elem in enumerate(mesh.leaf_elements):
        err_time, _ = error_estimator.sobolev_time(elem, residual)
        err_space, _ = error_estimator.sobolev_space(elem, residual)
        assert err_time == approx(errs[i, 0], abs=0, rel=1e-15)
        assert err_space == approx(errs[i, 1], abs=0, rel=1e-15)
