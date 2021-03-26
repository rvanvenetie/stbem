from mesh import Mesh, MeshParametrized
import numpy as np
from itertools import product
import random
from parametrization import Circle, UnitSquare, LShape


def test_initial_mesh():
    for glue_space in [False, True]:
        glue_space = True
        mesh = Mesh(glue_space=glue_space,
                    initial_space_mesh=[0., 0.3, 0.8, 1.])
        assert len(mesh.roots) == 3
        assert mesh.roots[0].edges[3].on_boundary
        assert mesh.roots[-1].edges[1].on_boundary
        for elem in mesh.roots:
            assert elem.edges[0].on_boundary
            assert elem.edges[2].on_boundary
            assert not elem.edges[0].neighbour_elements()
            assert not elem.edges[2].neighbour_elements()

        for i in range(len(mesh.roots) - 1):
            assert mesh.roots[i].edges[1].neighbour_elements() == [
                mesh.roots[i + 1]
            ]
            assert mesh.roots[i + 1].edges[3].neighbour_elements() == [
                mesh.roots[i]
            ]

        if glue_space:
            assert mesh.roots[0].edges[3].glued
            assert mesh.roots[-1].edges[1].glued
            assert mesh.roots[0].edges[3].nbr_edge == mesh.roots[-1].edges[1]
            assert mesh.roots[-1].edges[1].nbr_edge == mesh.roots[0].edges[3]
            assert mesh.roots[0].edges[3].neighbour_elements() == [
                mesh.roots[-1]
            ]
            assert mesh.roots[-1].edges[1].neighbour_elements() == [
                mesh.roots[0]
            ]


def test_refine_time():
    for glue_space in [False, True]:
        mesh = Mesh(glue_space=glue_space)
        e1, e2 = mesh.roots[0].edges_axis(0)
        assert e1.vertices[0].x == e1.vertices[1].x
        assert e1.vertices[0].t == 1 - e1.vertices[1].t
        assert e2.vertices[0].x == e2.vertices[1].x
        assert e2.vertices[0].t == 1 - e2.vertices[1].t
        children = mesh.refine_time(mesh.roots[0])
        assert len(children) == 2
        c1, c2 = children
        assert c1.levels == (1, 0)
        assert c2.levels == (1, 0)

        assert c1.vertices[0].tx == (0, 0)
        assert c1.vertices[1].tx == (0, 1)
        assert c1.vertices[2].tx == (0.5, 1)
        assert c1.vertices[3].tx == (0.5, 0)

        assert c2.vertices[0].tx == (0.5, 0)
        assert c2.vertices[1].tx == (0.5, 1)
        assert c2.vertices[2].tx == (1, 1)
        assert c2.vertices[3].tx == (1, 0)

        for k in range(3):
            leaves = list(mesh.leaf_elements)
            for elem in leaves:
                mesh.refine_time(elem)
            assert len(mesh.leaf_elements) == 2**(k + 2)


def test_refine_space():
    for glue_space in [False, True]:
        mesh = Mesh(glue_space=glue_space)
        e1, e2 = mesh.roots[0].edges_axis(1)
        assert e1.vertices[0].t == e1.vertices[1].t
        assert e1.vertices[0].x == 1 - e1.vertices[1].x
        assert e2.vertices[0].t == e2.vertices[1].t
        assert e2.vertices[0].x == 1 - e2.vertices[1].x
        children = mesh.refine_space(mesh.roots[0])
        assert len(children) == 2
        c1, c2 = children
        assert c1.levels == (0, 1)
        assert c2.levels == (0, 1)

        assert c1.vertices[0].tx == (0, 0)
        assert c1.vertices[1].tx == (0, 0.5)
        assert c1.vertices[2].tx == (1, 0.5)
        assert c1.vertices[3].tx == (1, 0)

        assert c2.vertices[0].tx == (0, 0.5)
        assert c2.vertices[1].tx == (0, 1)
        assert c2.vertices[2].tx == (1, 1)
        assert c2.vertices[3].tx == (1, 0.5)

        for k in range(3):
            leaves = list(mesh.leaf_elements)
            for elem in leaves:
                mesh.refine_space(elem)
            assert len(mesh.leaf_elements) == 2**(k + 2)


def test_uniform_refine():
    for glue_space, initial_mesh in product([True, False],
                                            [[0., 1.], [0., 0.3, 0.8, 1.]]):
        mesh = Mesh(glue_space=glue_space, initial_space_mesh=initial_mesh)
        for k in range(3):
            mesh.uniform_refine()
            for elem in mesh.leaf_elements:
                assert elem.levels == (k + 1, k + 1)
            assert len(
                mesh.leaf_elements) == (len(initial_mesh) - 1) * 4**(k + 1)


def test_local_refine():
    mesh = Mesh(glue_space=False)
    for k in range(5):
        for elem in list(mesh.leaf_elements):
            if elem.vertices[0].tx == (0, 0):
                mesh.refine(elem)
                break
        assert len(mesh.leaf_elements) == 4 + k * 3


def test_locally_uniform():
    for glue_space, initial_mesh in product([True, False],
                                            [[0., 1.], [0., 0.3, 0.8, 1.]]):
        mesh = Mesh(glue_space=glue_space, initial_space_mesh=initial_mesh)
        random.seed(5)
        for _ in range(100):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        leaves = list(mesh.leaf_elements)
        for elem in leaves:
            for edge in elem.edges:
                assert len(edge.neighbour_elements()) <= 2
                for nbr in edge.neighbour_elements():
                    assert abs(nbr.level_time - elem.level_time) <= 1
                    assert abs(nbr.level_space - elem.level_space) <= 1


def test_anisotropic():
    mesh = Mesh(glue_space=False)
    # Refine towards t = 0.
    for k in range(8):
        for elem in list(mesh.leaf_elements):
            if elem.vertices[0].tx == (0, 0):
                mesh.refine_time(elem)
                break
        assert len(mesh.leaf_elements) == k + 2

    # Refine towards t = 0.5, from below.
    mesh = Mesh(glue_space=False)
    c, _ = mesh.refine_time(mesh.roots[0])
    for k in range(5):
        _, c = mesh.refine_time(c)
        assert len(mesh.leaf_elements) == 3 + 2 * k


def test_boundaries():
    for glue_space, initial_mesh in product([True, False],
                                            [[0., 1.], [0., 0.3, 0.8, 1.]]):
        mesh = Mesh(glue_space=glue_space, initial_space_mesh=initial_mesh)
        random.seed(5)
        for _ in range(100):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)
        leaves = list(mesh.leaf_elements)

        for elem in leaves:
            for edge in elem.edges:
                assert edge.space_edge != edge.time_edge
                if edge.time_edge and edge.vertices[0].x in [0., 1.]:
                    # If this time edge lies on the space boundary.
                    if glue_space:
                        assert edge.glued and edge.on_boundary
                        assert len(edge.neighbour_elements()) > 0
                    else:
                        assert not edge.glued and edge.on_boundary
                        assert len(edge.neighbour_elements()) == 0
                elif edge.space_edge and edge.vertices[0].t in [0., 1.]:
                    # If this space edge lies on the time boundary.
                    assert not edge.glued and edge.on_boundary
                    assert len(edge.neighbour_elements()) == 0
                else:
                    assert not edge.on_boundary and not edge.glued
                    assert len(edge.neighbour_elements()) > 0


def test_parametrized():
    for gamma in [Circle(), UnitSquare(), LShape()]:
        mesh = MeshParametrized(gamma)
        assert len(mesh.roots) == len(gamma.pw_gamma)
        assert len(mesh.leaf_elements) >= 3
        random.seed(5)
        for _ in range(100):
            elem = random.choice(list(mesh.leaf_elements))
            mesh.refine_axis(elem, random.random() < 0.5)

        leaves = list(mesh.leaf_elements)
        for elem in leaves:
            assert elem.gamma_space is not None
            assert np.all(
                gamma.eval(elem.vertices[0].x) == elem.gamma_space(
                    elem.vertices[0].x))
