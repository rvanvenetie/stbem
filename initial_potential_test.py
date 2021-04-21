from mesh import Mesh, MeshParametrized
from initial_mesh import UnitSquareBoundaryRefined
import random
from scipy.special import expi, erf, expn, erfc
from pytest import approx
from parametrization import Circle, UnitSquare, LShape
import numpy as np
from initial_potential import InitialOperator


def test_initial_potential_circle():
    mesh = MeshParametrized(Circle())
    M0 = InitialOperator(mesh, lambda y: np.ones(y.shape[1]))

    for t in [1, 0.5, 0.1, 0.05]:
        val = M0.evaluate(t, [[0], [0]])
        assert val == approx(1 / 2 * (2 - 2 * np.exp(-(1 / 4) / t)))

    assert M0.evaluate(1, [[0.5], [0]]) == approx(0.20935725027818456022)

    M0 = InitialOperator(mesh, lambda y: y[0]**2 + y[1]**2)
    for t in [1, 0.5, 0.1]:
        val = M0.evaluate(t, [[0], [0]])
        assert val == approx(
            np.exp(-(1 / 4) / t) * (-1 - 4 * t + 4 * np.exp((1 / 4) / t) * t))

    M0 = InitialOperator(mesh, lambda y: np.sin(y[0]**2) * np.cos(y[1]**2))
    for t in [1, 0.5, 0.1]:
        val = M0.evaluate(t, [[0], [0]])
        assert val == approx(
            -(-4 * np.exp(1 / 4 / t) * t + 4 * t * np.cos(1) + np.sin(1)) /
            (np.exp(1 / 4 / t) * (2 * (1 + 16 * t**2))))


def test_initial_potential_unit_square():
    mesh = MeshParametrized(UnitSquare())
    M0 = InitialOperator(mesh, lambda y: np.ones(y.shape[1]))

    for t in [1, 0.5, 0.1, 0.05]:
        val = M0.evaluate(t, [[0], [0]])
        assert val == approx(1 / 4 * erf(1 / (2 * np.sqrt(t)))**2)

    M0 = InitialOperator(mesh, lambda y: y[0] - y[1] / 4)
    for t in [1, 0.5, 0.1, 0.05]:
        val = M0.evaluate(t, [[0], [0]])
    assert val == approx(
        (3 * (-1 + np.exp(1 / 4 / t)) * np.sqrt(t) * erf(1 /
                                                         (2 * np.sqrt(t)))) /
        (np.exp(1 / 4 / t) * (8 * np.sqrt(np.pi))))


def test_initial_potential_square():
    mesh = MeshParametrized(UnitSquare())
    mesh.uniform_refine()
    M0 = InitialOperator(mesh,
                         u0=lambda y: np.ones(y.shape[1]),
                         initial_mesh=UnitSquareBoundaryRefined)
    elems = list(mesh.leaf_elements)
    vec = M0.linform_vector()
    for i, elem in enumerate(elems):
        if elem.time_interval[0] == 0:
            assert vec[i] == approx(0.0578798287400388022, abs=0, rel=1e-8)
        else:
            assert vec[i] == approx(0.023234887895444152863, abs=0, rel=1e-12)

    _, elem_ip = M0.linform(elems[0])
    for elem, val in elem_ip:
        if elem.vertices[0].xy == (0, 0):
            assert val == approx(0.026593400385924482551, abs=0, rel=1e-8)
        elif elem.vertices[0].xy == (0.5, 0.0):
            assert val == approx(0.0149249985732353144, abs=0, rel=1e-8)
        elif elem.vertices[0].xy == (0, 0.5):
            assert val == approx(0.0093647867610682937049, abs=0, rel=1e-8)
        elif elem.vertices[0].xy == (0.5, 0.5):
            assert val == approx(0.0069966430198107115389, abs=0, rel=1e-8)

    M0 = InitialOperator(mesh,
                         u0=lambda y: y[0],
                         initial_mesh=UnitSquareBoundaryRefined)
    vec = M0.linform_vector()
    for i, elem in enumerate(elems):
        if elem.time_interval[0] == 0.5 and elem.space_interval == (0., 0.5):
            assert vec[i] == approx(0.011284335403468743609, abs=0, rel=1e-12)
        if elem.time_interval[0] == 0.5 and elem.space_interval == (0.5, 1.):
            assert vec[i] == approx(0.011950552491979953909, abs=0, rel=1e-12)
        if elem.time_interval[0] == 0. and elem.space_interval == (0.5, 1.):
            assert vec[i] == approx(0.033198942373537576959, abs=0, rel=1e-8)
        if elem.time_interval[0] == 0. and elem.space_interval == (3.5, 4.):
            assert vec[i] == approx(0.020441546057317420098, abs=0, rel=1e-8)

    # Test all values for u0(x,y) = sin(x) * y.
    M0 = InitialOperator(mesh,
                         u0=lambda y: np.sin(y[0]) * y[1],
                         initial_mesh=UnitSquareBoundaryRefined)
    vec = M0.linform_vector()
    for i, elem in enumerate(elems):
        if elem.time_interval[0] == 0. and elem.space_interval == (0, 0.5):
            assert vec[i] == approx(0.0084268990247339470474, abs=0, rel=1e-8)
        if elem.time_interval[0] == 0. and elem.space_interval == (0.5, 1):
            assert vec[i] == approx(0.010425254168044947963, abs=0, rel=1e-8)
        if elem.time_interval[0] == 0. and elem.space_interval == (1, 1.5):
            assert vec[i] == approx(0.014124726413635962236, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (1.5, 2):
            assert vec[i] == approx(0.019638534150369279880, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (2, 2.5):
            assert vec[i] == approx(0.019930334687473911117, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (2.5, 3):
            assert vec[i] == approx(0.014620516060188132933, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (3, 3.5):
            assert vec[i] == approx(0.010720508899757249512, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (3.5, 4):
            assert vec[i] == approx(0.0085020395721043026763, abs=0, rel=1e-7)

        if elem.time_interval[0] == 0.5 and elem.space_interval == (0, 0.5):
            assert vec[i] == approx(0.0049009043334054711203, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0.5 and elem.space_interval == (0.5, 1):
            assert vec[i] == approx(0.0051688745546247351823, abs=0, rel=1e-6)
        if elem.time_interval[0] == 0.5 and elem.space_interval == (1, 1.5):
            assert vec[i] == approx(0.0054720697055367215018, abs=0, rel=1e-6)
        if elem.time_interval[0] == 0.5 and elem.space_interval == (1.5, 2):
            assert vec[i] == approx(0.0057957778795656491943, abs=0, rel=1e-5)

    # Test some values for u0(x,y) = sin(x) * y.
    mesh.uniform_refine()
    M0 = InitialOperator(mesh,
                         u0=lambda y: np.sin(y[0]) * y[1],
                         initial_mesh=UnitSquareBoundaryRefined)
    vec = M0.linform_vector()
    elems = list(mesh.leaf_elements)
    for i, elem in enumerate(elems):
        if elem.time_interval[0] == 0. and elem.space_interval == (0, 0.25):
            assert vec[i] == approx(0.0018754911097772964009, abs=0, rel=1e-8)
        if elem.time_interval[0] == 0. and elem.space_interval == (0.25, 0.5):
            assert vec[i] == approx(0.0026894443401318872309, abs=0, rel=1e-8)
        if elem.time_interval[0] == 0. and elem.space_interval == (0.5, 0.75):
            assert vec[i] == approx(0.0031728320183010962938, abs=0, rel=1e-6)
        if elem.time_interval[0] == 0. and elem.space_interval == (0.75, 1):
            assert vec[i] == approx(0.0029778525721086943537, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (1, 1.25):
            assert vec[i] == approx(0.0036112935512706994032, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (1.25, 1.5):
            assert vec[i] == approx(0.0057342478146120195920, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (1.5, 1.75):
            assert vec[i] == approx(0.0072787492490388637853, abs=0, rel=1e-7)
        if elem.time_interval[0] == 0. and elem.space_interval == (1.75, 2):
            assert vec[i] == approx(0.0070246978081986050930, abs=0, rel=1e-6)


def test_initial_potential_evaluate():
    mesh = MeshParametrized(UnitSquare())
    M0 = InitialOperator(bdr_mesh=mesh,
                         u0=lambda y: np.sin(y[0]) * y[1],
                         quad_eval=55)
    # Choice X in the interior.
    x = np.array([[0.5], [0.5]])
    assert M0.evaluate(1, x) == approx(0.0175628125343357995877740924697,
                                       abs=0,
                                       rel=1e-15)
    assert M0.evaluate(0.5, x) == approx(0.0337504573274485763918544294309,
                                         abs=0,
                                         rel=1e-15)
    assert M0.evaluate(0.1, x) == approx(0.1254904823141527244648782241110,
                                         abs=0,
                                         rel=1e-12)
    assert M0.evaluate(0.01, x) == approx(0.237147179846976417036758053345,
                                          abs=0,
                                          rel=1e-12)
    # TODO: INSTABLE.
    assert M0.evaluate(0.001, x) == approx(0.239473176349241907505021205474,
                                           abs=0,
                                           rel=1e-4)

    # Choice X on the boundary.
    x = np.array([[0.5], [0]])
    assert M0.evaluate(1, x) == approx(0.0158639139909734737898716357674,
                                       abs=0,
                                       rel=1e-15)
    assert M0.evaluate(0.5, x) == approx(0.0276704987660549098138357290730,
                                         abs=0,
                                         rel=1e-15)
    assert M0.evaluate(0.1, x) == approx(0.0558118617425246607110298404885,
                                         abs=0,
                                         rel=1e-15)
    assert M0.evaluate(0.01, x) == approx(0.0267700878683716800648993551890,
                                          abs=0,
                                          rel=1e-15)
    assert M0.evaluate(0.001, x) == approx(0.00854499738192775818970279763857,
                                           abs=0,
                                           rel=1e-5)

    x = np.array([[1], [1]])
    assert M0.evaluate(0.5, x) == approx(0.0311242955274897776912896738701,
                                         abs=0,
                                         rel=1e-15)
    assert M0.evaluate(0.1, x) == approx(0.0946123480667496304577916434116,
                                         abs=0,
                                         rel=1e-15)
    assert M0.evaluate(0.01, x) == approx(0.171341260445764059011160158183,
                                          abs=0,
                                          rel=1e-15)
    assert M0.evaluate(0.001, x) == approx(0.198013791933202148280141026357,
                                           abs=0,
                                           rel=1e-8)
