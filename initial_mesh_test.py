import numpy as np
from initial_mesh import UnitSquare, LShape
import random
from collections import defaultdict


def test_uniform_refine():
    mesh = UnitSquare()
    for k in range(4):
        mesh.uniform_refine()
        for elem in mesh.leaf_elements:
            assert elem.level == k + 1
        assert len(mesh.leaf_elements) == 4**(k + 1)

    mesh = LShape()
    for k in range(4):
        mesh.uniform_refine()
        for elem in mesh.leaf_elements:
            assert elem.level == k + 1
        assert len(mesh.leaf_elements) == 3 * 4**(k + 1)


def test_local_refine_square():
    mesh = UnitSquare()
    for k in range(10):
        for elem in list(mesh.leaf_elements):
            if elem.vertices[0].xy == (0, 0):
                mesh.refine(elem)
                break
        assert len(mesh.leaf_elements) == 4 + k * 3


def test_local_refine_lshape():
    mesh = LShape()
    for k in range(10):
        for elem in list(mesh.leaf_elements):
            if elem.vertices[0].xy == (0, 0):
                mesh.refine(elem)
                break
        assert len(mesh.leaf_elements) == 6 + k * 9


def test_local_refinement_square():
    mesh = UnitSquare()
    mesh.uniform_refine()
    assert len(mesh.leaf_elements) == 4
    for elem in mesh.leaf_elements:
        if elem.vertices[0].xy == (0.5, 0.5):
            mesh.refine(elem)
            break
    assert len(mesh.leaf_elements) == 7
    for elem in mesh.leaf_elements:
        if elem.vertices[0].xy == (0.5, 0.5):
            mesh.refine(elem)
            break
    assert len(mesh.leaf_elements) == 16
    for elem in mesh.leaf_elements:
        if elem.vertices[0].xy == (0.5, 0.5):
            mesh.refine(elem)
            break
    assert len(mesh.leaf_elements) == 28


def test_bdr_refine():
    for bdr in [((1, 0.53125), (1, 0.5)), ((1, 0.5), (1, 0.53125))]:
        mesh = UnitSquare()
        elem = mesh.refine_msh_bdr(*bdr)
        assert len(mesh.leaf_elements) == 25
        assert elem.contains((1, 0.53125))
        assert elem.contains((1, 0.5))
        assert elem.contains((1, 0.5))
        assert elem.contains((1, 0.53125))
        assert not elem.contains((1, 0.54))


def test_gamma():
    mesh = UnitSquare()
    mesh.refine_msh_bdr((1, 0.53125), (1, 0.5))
    assert len(mesh.leaf_elements) == 25
    for elem in mesh.leaf_elements:
        gamma = elem.gamma()
        assert np.all(gamma(0, 0) == elem.vertices[0].xy_np)
        assert np.all(gamma(1, 0) == elem.vertices[1].xy_np)
        assert np.all(gamma(1, 1) == elem.vertices[2].xy_np)
        assert np.all(gamma(0, 1) == elem.vertices[3].xy_np)
        assert elem.contains(gamma(0.5, 0.5))
        assert elem.contains(gamma(0.5, 0))
        assert elem.contains(gamma(0.5, 1))
        assert elem.contains(gamma(0, 0.5))
        assert elem.contains(gamma(1, 0.5))
        assert elem.contains(gamma(1, 0.5))
