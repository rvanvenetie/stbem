from pytest import approx
from scipy.special import exp1
from quadrature import gauss_quadrature_scheme, log_quadrature_scheme
import quadpy
import random
import numpy as np
from parametrization import Circle, UnitSquare, LShape, UnitInterval
from mesh import Mesh, MeshParametrized
from single_layer import double_time_integrated_kernel, time_integrated_kernel, SingleLayerOperator
from single_layer_exact import spacetime_integrated_kernel_1, spacetime_integrated_kernel_2, spacetime_integrated_kernel_3, spacetime_integrated_kernel_4, spacetime_evaluated_1, spacetime_evaluated_2
import itertools


def test_time_integrated():
    x_vals = [0.2, 1.0, 3]
    # Calculated using Mathematica.
    G_time = double_time_integrated_kernel(0, 1, 0, 1)
    for x, result in zip(x_vals, [0.245756, 0.0419017, 0.000602999]):
        assert G_time(x) == approx(result)

    G_time = double_time_integrated_kernel(1.5, 3, 1.5, 3)
    for x, result in zip(x_vals, [0.414955, 0.0903745, 0.00321319]):
        assert G_time(x) == approx(result, rel=1e-5)

    G_time = double_time_integrated_kernel(1.5, 3, 2, 5)
    for x, result in zip(x_vals, [0.245756, 0.0419017, 0.000602999]):
        assert G_time(x) == approx(result)

    G_time = double_time_integrated_kernel(1.5, 3, 3, 5)
    for x, result in zip(x_vals, [0., 0., 0.]):
        assert G_time(x) == approx(result)

    G_time = double_time_integrated_kernel(1.5, 3, 4, 5)
    for x, result in zip(x_vals, [0., 0., 0.]):
        assert G_time(x) == approx(result)

    G_time = double_time_integrated_kernel(1.5, 3, 1, 5)
    for x, result in zip(x_vals, [0.500433, 0.140944, 0.00872243]):
        assert G_time(x) == approx(result, rel=1e-5)

    G_time = double_time_integrated_kernel(1.5, 3, 1, 1.5)
    for x, result in zip(x_vals, [0.0854777, 0.0505691, 0.00550924]):
        assert G_time(x) == approx(result, rel=1e-5)

    G_time = double_time_integrated_kernel(1.5, 3, 1, 2)
    for x, result in zip(x_vals, [0.254677, 0.0990418, 0.0081194]):
        assert G_time(x) == approx(result, rel=1e-5)

    G_time = double_time_integrated_kernel(1.5, 3, 0, 0.5)
    for x, result in zip(x_vals, [0.0314082, 0.0274608, 0.00934553]):
        assert G_time(x) == approx(result, rel=1e-5)


def test_time_integrated_potential():
    x = 0.1
    t = 1

    # Calculated using Mathematica.
    G_time = time_integrated_kernel(t, 0, 1)
    assert G_time(x) == approx(0.43105105577457354964)
    G_time = time_integrated_kernel(t, 0, 2)
    assert G_time(x) == approx(0.43105105577457354964)
    G_time = time_integrated_kernel(t, 0, 0.5)
    assert G_time(x) == approx(0.0549603)
    G_time = time_integrated_kernel(t, 1, 5)
    assert G_time(x) == 0
    G_time = time_integrated_kernel(t, 0.5, 5)
    assert G_time(x) == approx(0.376091)


def test_single_layer_square():
    gamma = UnitSquare()
    mesh = MeshParametrized(gamma)
    SL = SingleLayerOperator(mesh)
    mat = SL.bilform_matrix()

    # This bilform_matrix should be symmetric.
    assert np.allclose(mat, mat.T)
    for i in range(4):
        assert mat[i, i] == approx(0.23680355333817647868, abs=0, rel=1e-13)
        assert mat[i, (i + 1) % 4] == approx(mat[i, (i + 3) % 4])
        assert mat[i, (i + 1) % 4] == approx(0.0838829410097953185010223929460,
                                             abs=0,
                                             rel=1e-13)
        assert mat[i, (i + 2) % 4] == approx(0.036534485699376823056,
                                             abs=0,
                                             rel=1e-13)


def test_single_layer_circle():
    gamma = Circle()
    mesh = MeshParametrized(gamma)
    SL = SingleLayerOperator(mesh)

    # Uniformly refine space twice.
    mesh.uniform_refine_space()
    mesh.uniform_refine_space()
    mesh.uniform_refine_space()
    mesh.uniform_refine_space()
    mat = SL.bilform_matrix()

    # This bilform_matrix should be symmetric.
    assert np.allclose(mat, mat.T)
    N = len(mesh.leaf_elements)

    for j in range(N):
        # Check that the row j is circular.
        for i in range(N):
            assert mat[j, (j + i) % N] == approx(mat[j, (j - i) % N])
        # Check that row j coincides with row 0.
        for i in range(N):
            assert mat[0, i] == approx(mat[j, (j + i) % N])


def test_single_layer_pw_polygon_overlap():
    for gamma in [UnitSquare(), LShape(), UnitInterval()]:
        mesh = MeshParametrized(gamma)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(200):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        # We can exactly integrate the kernel if the space part coincides.
        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for i, elem_test in enumerate(elems):
            for j, elem_trial in enumerate(elems):
                if elem_test.gamma_space != elem_trial.gamma_space: continue
                if elem_test.space_interval[0] != elem_trial.space_interval[0]:
                    continue
                val = SL.bilform(elem_trial, elem_test)

                len_space_test = elem_test.space_interval[
                    1] - elem_test.space_interval[0]
                len_space_trial = elem_trial.space_interval[
                    1] - elem_trial.space_interval[0]
                if elem_test.space_interval == elem_trial.space_interval:
                    val_exact = spacetime_integrated_kernel_1(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        len_space_test)
                else:
                    val_exact = spacetime_integrated_kernel_3(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        len_space_test, len_space_trial)

                if val_exact == 0: assert val == 0
                else: assert val == approx(val_exact, abs=0, rel=1e-9)


def test_single_layer_pw_polygon_overlap_bla():
    for gamma in [UnitSquare(), LShape(), UnitInterval()]:
        mesh = MeshParametrized(gamma)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(200):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        # We can exactly integrate the kernel if the space part coincides.
        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for i, elem_test in enumerate(elems):
            for j, elem_trial in enumerate(elems):
                if elem_test.gamma_space != elem_trial.gamma_space: continue
                if elem_test.space_interval[0] > elem_trial.space_interval[
                        0] or elem_test.space_interval[
                            1] < elem_trial.space_interval[1]:
                    continue

                assert (
                    elem_test.space_interval[0] <= elem_trial.space_interval[0]
                    and elem_test.space_interval[1] >=
                    elem_trial.space_interval[1])

                val_exact = 0
                # Evaluate the before part.
                if elem_test.space_interval[0] < elem_trial.space_interval[0]:
                    x_a, x_b = elem_test.space_interval[
                        0], elem_trial.space_interval[0]
                    y_a, y_b = elem_trial.space_interval
                    val_exact = spacetime_integrated_kernel_2(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        x_b - x_a, y_b - y_a)

                # Evaluate the overlap part.
                if (elem_trial.space_interval[0], elem_test.space_interval[1]
                    ) == elem_trial.space_interval:
                    val_exact += spacetime_integrated_kernel_1(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        elem_trial.space_interval[1] -
                        elem_trial.space_interval[0])
                else:
                    val_exact += spacetime_integrated_kernel_3(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        elem_test.space_interval[1] -
                        elem_trial.space_interval[0],
                        elem_trial.space_interval[1] -
                        elem_trial.space_interval[0])

                val = SL.bilform(elem_trial, elem_test)

                if val_exact == 0: assert val == 0
                else: assert val == approx(val_exact, abs=0, rel=1e-8)


def test_single_layer_pw_polygon_touch():
    for gamma in [UnitSquare(), LShape(), UnitInterval()]:
        mesh = MeshParametrized(gamma)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(200):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        # We can exactly integrate the kernel if the space part coincides.
        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for i, elem_test in enumerate(elems):
            for j, elem_trial in enumerate(elems):
                if elem_test.gamma_space != elem_trial.gamma_space: continue
                if elem_test.space_interval[1] == elem_trial.space_interval[0]:
                    val = SL.bilform(elem_trial, elem_test)
                    middle = elem_test.space_interval[1]
                    val_exact = spacetime_integrated_kernel_2(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        middle - elem_test.space_interval[0],
                        elem_trial.space_interval[1] - middle)
                    if val_exact == 0: assert val == 0
                    else: assert val == approx(val_exact, abs=0, rel=1e-9)
                if elem_trial.space_interval[1] == elem_test.space_interval[0]:
                    val = SL.bilform(elem_trial, elem_test)
                    middle = elem_trial.space_interval[1]
                    val_exact = spacetime_integrated_kernel_2(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        middle - elem_trial.space_interval[0],
                        elem_test.space_interval[1] - middle)
                    if val_exact == 0: assert val == 0
                    else: assert val == approx(val_exact, abs=0, rel=1e-9)


def test_single_layer_pw_polygon_disjoint():
    for gamma in [UnitSquare(), LShape(), UnitInterval()]:
        mesh = MeshParametrized(gamma)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(200):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        # We can exactly integrate the kernel if the space part coincides.
        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for i, elem_test in enumerate(elems):
            for j, elem_trial in enumerate(elems):
                if elem_test.gamma_space != elem_trial.gamma_space: continue
                if elem_test.space_interval[1] < elem_trial.space_interval[0]:
                    val = SL.bilform(elem_trial, elem_test)
                    begin = elem_test.space_interval[0]
                    val_exact = spacetime_integrated_kernel_4(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        elem_test.space_interval[1] - begin,
                        elem_trial.space_interval[0] - begin,
                        elem_trial.space_interval[1] - begin)
                    if val_exact == 0: assert val == 0
                    else: assert val == approx(val_exact)
                if elem_trial.space_interval[1] < elem_test.space_interval[0]:
                    val = SL.bilform(elem_trial, elem_test)
                    begin = elem_trial.space_interval[0]
                    val_exact = spacetime_integrated_kernel_4(
                        *elem_test.time_interval, *elem_trial.time_interval,
                        elem_trial.space_interval[1] - begin,
                        elem_test.space_interval[0] - begin,
                        elem_test.space_interval[1] - begin)
                    if val_exact == 0: assert val == 0
                    else: assert val == approx(val_exact)


def test_single_layer_far():
    for gamma in [Circle(), UnitSquare(), LShape()]:
        print(gamma)
        mesh = MeshParametrized(gamma)
        dim = 4
        scheme_stroud = quadpy.cn.stroud_cn_7_1(dim)
        scheme_mcnamee = quadpy.cn.mcnamee_stenger_9b(dim)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(200):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        # We can exactly integrate the kernel if the space part coincides.
        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for i, elem_test in enumerate(elems):
            for j, elem_trial in enumerate(elems):
                if elem_test.time_interval[
                        0] <= elem_trial.time_interval[1] + 0.5:
                    continue

                def kernel(x):
                    ts = x[0] - x[2]
                    xy = elem_test.gamma_space(x[1]) - elem_trial.gamma_space(
                        x[3])
                    xysqr = np.sum(xy**2, axis=0)
                    return 1. / (4 * np.pi * ts) * np.exp(-xysqr / (4 * ts))

                cube_points = quadpy.cn.ncube_points(elem_test.time_interval,
                                                     elem_test.space_interval,
                                                     elem_trial.time_interval,
                                                     elem_trial.space_interval)
                #val_stroud = scheme_stroud.integrate(kernel, cube_points)
                val_mcnamee = scheme_mcnamee.integrate(kernel, cube_points)
                val = SL.bilform(elem_trial, elem_test)
                assert abs(val - val_mcnamee) / val < 1e-7


def test_single_layer_potential():
    for gamma, x_ref in [(Circle(), np.array([[0.75], [0]])),
                         (UnitSquare(), np.array([[0.75], [0.75]]))]:
        print(gamma)
        mesh = MeshParametrized(gamma)
        scheme = quadpy.c2.get_good_scheme(21)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(500):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for j, elem_trial in enumerate(elems):
            for t in [0, 0.25, 0.5, 0.75, 1]:
                if elem_trial.time_interval[
                        0] <= t <= elem_trial.time_interval[1]:
                    continue

                def kernel(x):
                    ts = t - x[0]
                    xy = x_ref - elem_trial.gamma_space(x[1])
                    xysqr = np.sum(xy**2, axis=0)
                    return 1. / (4 * np.pi * ts) * np.exp(-xysqr / (4 * ts))

                rect_points = quadpy.c2.rectangle_points(
                    elem_trial.time_interval, elem_trial.space_interval)
                if t <= elem_trial.time_interval[0]: val_quadpy = 0
                else: val_quadpy = scheme.integrate(kernel, rect_points)
                val = SL.potential(elem_trial, t, x_ref)
                assert val_quadpy == approx(val)


def test_single_layer_evaluate():
    mesh = MeshParametrized(UnitSquare())
    SL = SingleLayerOperator(mesh)
    elems = list(mesh.leaf_elements)
    vecs = np.zeros(shape=(4, 4))

    # Evaluate t = 1, x = 0, 1, 2, 3, and check symmetries.
    for i in range(4):
        vecs[:, i] = SL.evaluate_vector(1, i)
    assert vecs[0, 0] == approx(0.22993167627837422544)
    assert vecs[0, 1] == approx(0.22993167627837422544)
    assert vecs[0, 2] == approx(0.067761938900930558883)
    assert vecs[0, 3] == approx(0.067761938900930558883)

    # Check that the rows are permuted.
    for j in range(4):
        for i in range(4):
            assert vecs[0, i] == approx(vecs[j, (j + i) % 4])

    # Check circle.
    mesh = MeshParametrized(Circle())
    SL = SingleLayerOperator(mesh)
    elems = list(mesh.leaf_elements)
    vecs = np.zeros(shape=(4, 4))

    # Evaluate t = 1, x = 0, 1, 2, 3, and check symmetries.
    for i in range(4):
        vecs[:, i] = SL.evaluate_vector(1, i * SL.gamma_len / 4)

    assert vecs[0, 0] == approx(0.26798886287076459268)
    assert vecs[0, 1] == approx(0.26798886287076459268)
    assert vecs[0, 2] == approx(0.039562341685844190737)
    assert vecs[0, 3] == approx(0.039562341685844190737)

    # Check that the rows are permuted.
    for j in range(4):
        for i in range(4):
            assert vecs[0, i] == approx(vecs[j, (j + i) % 4])

    # Evaluate t = 0.3, x = 0, 1, 2, 3, and check symmetries.
    for i in range(4):
        vecs[:, i] = SL.evaluate_vector(0.3, i * SL.gamma_len / 4)

    assert vecs[0, 0] == approx(0.15669367056341952730)
    assert vecs[0, 1] == approx(0.15669367056341952730)
    assert vecs[0, 2] == approx(0.0029889359574251946985)
    assert vecs[0, 3] == approx(0.0029889359574251946985)

    # Check that the rows are permuted.
    for j in range(4):
        for i in range(4):
            assert vecs[0, i] == approx(vecs[j, (j + i) % 4])


def test_single_layer_evaluate_disj():
    for gamma in [Circle(), UnitSquare(), LShape()]:
        mesh = MeshParametrized(gamma)
        scheme = quadpy.c2.get_good_scheme(21)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(300):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for j, elem_trial in enumerate(elems):
            for t in [0, 0.13, 0.25, 0.5, 0.75, 1]:
                if elem_trial.time_interval[
                        0] - 0.1 <= t <= elem_trial.time_interval[1] + 0.1:
                    continue
                for x_hat in [0, 0.5 * SL.gamma_len, SL.gamma_len]:
                    x_ref = mesh.gamma_space.eval(x_hat)

                    def kernel(x):
                        ts = t - x[0]
                        xy = x_ref - elem_trial.gamma_space(x[1])
                        xysqr = np.sum(xy**2, axis=0)
                        return 1. / (4 * np.pi * ts) * np.exp(-xysqr /
                                                              (4 * ts))

                    rect_points = quadpy.c2.rectangle_points(
                        elem_trial.time_interval, elem_trial.space_interval)
                    if t <= elem_trial.time_interval[0]: val_quadpy = 0
                    else: val_quadpy = scheme.integrate(kernel, rect_points)
                    val = SL.evaluate(elem_trial, t, x_hat)
                    assert val_quadpy == approx(val)


def test_single_layer_evaluate_pw_polygon():
    for gamma in [UnitSquare(), LShape(), UnitInterval()]:
        mesh = MeshParametrized(gamma)
        scheme = quadpy.c2.get_good_scheme(21)

        # Randomly refine this mesh.
        random.seed(5)
        for _ in range(300):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        SL = SingleLayerOperator(mesh)
        elems = list(mesh.leaf_elements)
        for j, elem_trial in enumerate(elems):
            for t in [0, 0.01, 0.13, 0.25, 0.5, 0.75, 1]:
                for x in [
                        0, 0.5 * SL.gamma_len, SL.gamma_len,
                        elem_trial.space_interval[0],
                        elem_trial.space_interval[1],
                        0.5 * (elem_trial.space_interval[0] +
                               elem_trial.space_interval[1])
                ]:
                    if not np.all(
                            mesh.gamma_space.eval(x) == elem_trial.gamma_space(
                                x)):
                        continue

                    a, b = elem_trial.space_interval
                    if x == b or x == a:
                        val_exact = spacetime_evaluated_1(
                            t, *elem_trial.time_interval, b - a)
                        val = SL.evaluate(elem_trial, t, x)
                        assert val == approx(val_exact, abs=0, rel=1e-8)
                    if a < x < b:
                        val_exact = spacetime_evaluated_1(
                            t, *elem_trial.time_interval,
                            x - a) + spacetime_evaluated_1(
                                t, *elem_trial.time_interval, b - x)
                        val = SL.evaluate(elem_trial, t, x)
                        assert val == approx(val_exact, abs=0, rel=1e-8)

                    if x < a or x > b:
                        h = min(abs(a - x), abs(b - x))
                        k = max(abs(a - x), abs(b - x))
                        val_exact = spacetime_evaluated_2(
                            t, *elem_trial.time_interval, h, k)
                        val = SL.evaluate(elem_trial, t, x)
                        assert val == approx(val_exact, rel=1e-8)


def test_rhs():
    gamma = Circle()
    mesh = MeshParametrized(gamma)
    SL = SingleLayerOperator(mesh)

    def u(t, x):
        return 1 / (4 * np.pi * t) * np.exp(-((x[0] - 1)**2 + (x[1] - 1)**2) /
                                            (4 * t))

    def u_int(a, b):
        """ Function u integrated over the time interval [a,b]. """
        if a == 0:
            time_integrated_kernel = lambda xy: 1. / (4 * np.pi) * exp1(xy / (
                4 * b))
        else:
            time_integrated_kernel = lambda xy: 1. / (4 * np.pi) * (exp1(xy / (
                4 * b)) - exp1(xy / (4 * a)))

        return lambda x: time_integrated_kernel((x[0] - 1)**2 + (x[1] - 1)**2)

    rhs = SL.rhs_vector(u)
    #assert rhs[0] == approx(0.23307837618931861567,abs=0,rel=1e-7)

    rhs = np.zeros(shape=len(mesh.leaf_elements))
    gauss_scheme = gauss_quadrature_scheme(101)
    for i, elem_test in enumerate(mesh.leaf_elements):
        f_int = u_int(*elem_test.time_interval)
        f_param = lambda x: f_int(elem_test.gamma_space(x))
        rhs[i] = gauss_scheme.integrate(f_param, *elem_test.space_interval)
    assert rhs[0] == approx(0.23307837618931861567)


def test_single_layer_refine():
    gamma = UnitSquare()
    mesh_trial = MeshParametrized(gamma)
    mesh_test = MeshParametrized(gamma)
    SL = SingleLayerOperator(mesh_trial)

    # Randomly refine the meshes
    random.seed(5)
    for _ in range(200):
        elem_trial = random.choice(list(mesh_trial.leaf_elements))
        #elem_trial = random.choice([
        #    elem for elem in mesh_trial.leaf_elements
        #    if elem.time_interval[0] == 0. or elem.space_interval[0] == 0.
        #])
        mesh_trial.refine_axis(elem_trial, random.random() < 0.5)

        elem_test = random.choice(list(mesh_test.leaf_elements))
        mesh_test.refine_axis(elem_test, random.random() < 0.5)

    elems_trial = list(mesh_trial.leaf_elements)
    elems_test = list(mesh_test.leaf_elements)

    mesh_trial.uniform_refine()
    mesh_test.uniform_refine()

    print(len(elems_trial))

    for elem_test in elems_trial:
        elems_test_children = [
            child for child_time in elem_test.children
            for child in child_time.children
        ]
        for elem_trial in elems_trial:
            elems_trial_children = [
                child for child_time in elem_trial.children
                for child in child_time.children
            ]
            val_refined = 0
            for elem_test_child in elems_test_children:
                for elem_trial_child in elems_trial_children:
                    val_refined += SL.bilform(elem_trial_child,
                                              elem_test_child)

            val = SL.bilform(elem_trial, elem_test)
            if val_refined != 0:
                err = abs((val - val_refined) / val_refined)
                if err > 1e-10 and val > 1e-35:
                    print(elem_trial, elem_test, err, val, val_refined)

            assert val == approx(val_refined, abs=1e-50, rel=1e-10)
