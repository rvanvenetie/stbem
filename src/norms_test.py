import numpy as np

from .norms import Slobodeckij
from .parametrization import line


def test_seminorm_h_1_4():
    val_exact = 3.0169889330626793416
    rel_error = 1
    for N in range(1, 9, 2):
        q_f = Slobodeckij(N).seminorm_h_1_4(lambda x: x, 0, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('id', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-13

    val_exact = 17.066666666666666667
    rel_error = 1
    for N in range(1, 9, 2):
        q_f = Slobodeckij(N).seminorm_h_1_4(lambda x: x, -2, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('id', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-13

    val_exact = 1.8386765898550357757
    rel_error = 1
    for N in range(1, 21, 2):
        q_f = Slobodeckij(N).seminorm_h_1_4(lambda x: np.cos(x), 0, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('cos', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-13

    # Non smooth function.
    val_exact = 1.0679586722084124843
    rel_error = 1
    for N in range(1, 21, 2):
        q_f = Slobodeckij(N).seminorm_h_1_4(lambda x: np.sqrt(x), 0, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('sqrt', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error > 1e-8


def test_seminorm_h_1_2():
    val_exact = 16
    rel_error = 1
    for N in range(1, 9, 2):
        q_f = Slobodeckij(N).seminorm_h_1_2(lambda x: x, -2, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('id', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-13

    val_exact = 3.1218783168509886015
    rel_error = 1
    for N in range(1, 19, 2):
        q_f = Slobodeckij(N).seminorm_h_1_2(lambda x: np.cos(x), -1, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('cos', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    #assert rel_error < 1e-13

    # Non smooth function.
    #val_exact = 1.0679586722084124843
    #rel_error = 1
    #for N in range(1, 21, 2):
    #    q_f = Slobodeckij(N).seminorm_h_1_4(lambda x: np.sqrt(x), 0, 2)
    #    new_rel_error = abs((q_f - val_exact) / val_exact)
    #    print(N, new_rel_error)
    #    assert new_rel_error < rel_error or new_rel_error < 1e-13
    #    rel_error = new_rel_error

    #assert rel_error > 1e-8


def test_seminorm_h_1_2_gamma_dummy():
    # Dummy gamma.
    for gamma in [
            line(np.array([0, 0]), np.array([1, 0]))[0],
            line(np.array([0, 0]), np.array([0, 1]))[0],
            line(np.array([1, 1]), np.array([4, 1]))[0]
    ]:
        val_exact = 16
        rel_error = 1
        for N in range(1, 9, 2):
            q_f = Slobodeckij(N).seminorm_h_1_2(lambda x_hat, x: x_hat,
                                                -2,
                                                2,
                                                gamma=gamma)
            new_rel_error = abs((q_f - val_exact) / val_exact)
            print('id', N, new_rel_error)
            assert new_rel_error < rel_error or new_rel_error < 1e-13
            rel_error = new_rel_error

        assert rel_error < 1e-13

        val_exact = 3.1218783168509886015
        rel_error = 1
        for N in range(1, 19, 2):
            q_f = Slobodeckij(N).seminorm_h_1_2(lambda x_hat, x: np.cos(x_hat),
                                                -1,
                                                2,
                                                gamma=gamma)
            new_rel_error = abs((q_f - val_exact) / val_exact)
            print('cos', N, new_rel_error)
            assert new_rel_error < rel_error or new_rel_error < 1e-13
            rel_error = new_rel_error


def test_seminorm_h_1_2_gamma_smooth():
    gamma = line(np.array([0, 0]), np.array([1, 1]))[0]

    val_exact = 16
    rel_error = 1
    for N in range(1, 9, 2):
        q_f = Slobodeckij(N).seminorm_h_1_2(lambda x_hat, x: x_hat,
                                            -2,
                                            2,
                                            gamma=gamma)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('id', N, new_rel_error, q_f)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-13

    val_exact = 1.9231117907067839377
    rel_error = 1
    for N in range(1, 18, 2):

        def f(x_hat, gamma):
            x = gamma(x_hat)
            return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])

        q_f = Slobodeckij(N).seminorm_h_1_2(f, 0, 1, gamma=gamma)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('sincos', N, new_rel_error, q_f)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-10

    val_exact = 4.8208511028016588706
    rel_error = 1
    for N in range(1, 18, 2):

        def f(x_hat, gamma):
            x = gamma(x_hat)
            return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])

        q_f = Slobodeckij(N).seminorm_h_1_2(f, 0, 2, gamma=gamma)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('sincos', N, new_rel_error, q_f)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-5

    val_exact = 3.8623935284968760922
    rel_error = 2
    for N in range(1, 18, 2):

        def f(x_hat, gamma):
            x = gamma(x_hat)
            return x[0] * np.cos(np.pi * x[1])

        q_f = Slobodeckij(N).seminorm_h_1_2(f, 0, 2, gamma=gamma)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('xcos', N, new_rel_error, q_f)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-9


def test_seminorm_h_1_2_gamma_pw():
    gamma_1 = line(np.array([0, 0]), np.array([1, 0]), x_start=0)[0]
    gamma_2 = line(np.array([1, 0]), np.array([1, 1]), x_start=1)[0]
    val_exact = 5.3862943611192362124
    rel_error = 1
    for N in range(1, 21, 2):
        q_f = Slobodeckij(N).seminorm_h_1_2_pw(lambda x_hat, x: x_hat, 0, 1,
                                               gamma_1, 1, 2, gamma_2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('id', N, new_rel_error, q_f)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error
    assert rel_error < 1e-12

    val_exact = 9.4232543677667364554
    rel_error = 3
    for N in range(1, 19, 2):

        def f(x_hat, gamma):
            x = gamma(x_hat)
            return x[0] * np.cos(np.pi * x[1])

        q_f = Slobodeckij(N).seminorm_h_1_2_pw(f, 0, 1, gamma_1, 1, 2, gamma_2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('xcos', N, new_rel_error, q_f)
        #assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error
    assert rel_error < 1e-11

    val_exact = 25.573523666405356638
    rel_error = 3
    for N in range(1, 21, 2):

        def f(x_hat, gamma):
            x = gamma(x_hat)
            return np.cos(2 * np.pi * x[0]) * np.cos(np.pi * x[1])

        q_f = Slobodeckij(N).seminorm_h_1_2_pw(f, 0, 1, gamma_1, 1, 2, gamma_2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('xcos', N, new_rel_error, q_f)
        #assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error
    assert rel_error < 1e-9
