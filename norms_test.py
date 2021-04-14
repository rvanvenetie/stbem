from slobodeckij import Slobodeckij
import numpy as np


def test_seminorm_1_4():
    val_exact = 3.0169889330626793416
    rel_error = 1
    for N in range(1, 9, 2):
        q_f = Slobodeckij(N).seminorm_1_4(lambda x: x, 0, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('id', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-13

    val_exact = 1.8386765898550357757
    rel_error = 1
    for N in range(1, 21, 2):
        q_f = Slobodeckij(N).seminorm_1_4(lambda x: np.cos(x), 0, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('cos', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-13

    # Non smooth function.
    val_exact = 1.0679586722084124843
    rel_error = 1
    for N in range(1, 21, 2):
        q_f = Slobodeckij(N).seminorm_1_4(lambda x: np.sqrt(x), 0, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('sqrt', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error > 1e-8


def test_seminorm_1_2():
    val_exact = 16
    rel_error = 1
    for N in range(1, 9, 2):
        q_f = Slobodeckij(N).seminorm_1_2(lambda x: x, -2, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('id', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    assert rel_error < 1e-13

    val_exact = 3.1218783168509886015
    rel_error = 1
    for N in range(1, 19, 2):
        q_f = Slobodeckij(N).seminorm_1_2(lambda x: np.cos(x), -1, 2)
        new_rel_error = abs((q_f - val_exact) / val_exact)
        print('cos', N, new_rel_error)
        assert new_rel_error < rel_error or new_rel_error < 1e-13
        rel_error = new_rel_error

    #assert rel_error < 1e-13

    ## Non smooth function.
    #val_exact = 1.0679586722084124843
    #rel_error = 1
    #for N in range(1, 21, 2):
    #    q_f = Slobodeckij(N).seminorm_1_4(lambda x: np.sqrt(x), 0, 2)
    #    new_rel_error = abs((q_f - val_exact) / val_exact)
    #    print(N, new_rel_error)
    #    assert new_rel_error < rel_error or new_rel_error < 1e-13
    #    rel_error = new_rel_error

    #assert rel_error > 1e-8
