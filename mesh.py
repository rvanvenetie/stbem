import random
import numpy as np
from collections import OrderedDict
from parametrization import Circle, UnitInterval, UnitSquare, LShape, PiecewiseParametrization
import hashlib


class Vertex:
    def __init__(self, t, x, idx):
        self.t = t
        self.x = x
        self.idx = idx

    @property
    def tx(self):
        return (self.t, self.x)

    def __repr__(self):
        return "({},{})".format(self.t, self.x)


class Edge:
    """ Represents a single edge of some element. """
    def __init__(self, vertices, parent=None):
        self.vertices = vertices
        self.parent = parent

        # These are to be filled.
        self.elem = None
        self.nbr_edge = None
        self.children = []

        if parent:
            self.on_boundary = parent.on_boundary
            self.glued = parent.glued
        else:
            self.on_boundary = False
            self.glued = False

    @property
    def space_edge(self):
        return self.vertices[0].t == self.vertices[1].t

    @property
    def time_edge(self):
        return self.vertices[0].x == self.vertices[1].x

    def bisect(self, child_vertex):
        """ Bisects this edge given the child_vertex. """
        if not self.children:
            a, b = self.vertices
            self.children = (Edge((a, child_vertex),
                                  self), Edge((child_vertex, b), self))

            # Update neighbouring relations between edges.
            if self.nbr_edge and self.nbr_edge.children:
                if not self.glued:
                    assert self.vertices[0] == self.nbr_edge.vertices[1]
                    assert self.vertices[1] == self.nbr_edge.vertices[0]
                assert not self.nbr_edge.children[0].nbr_edge
                assert not self.nbr_edge.children[1].nbr_edge

                self.children[0].nbr_edge = self.nbr_edge.children[1]
                self.children[1].nbr_edge = self.nbr_edge.children[0]
                self.nbr_edge.children[0].nbr_edge = self.children[1]
                self.nbr_edge.children[1].nbr_edge = self.children[0]

        return self.children

    def neighbour_elements(self):
        """ Find the neighbouring element of this edge. """
        # If we have a neighbour edge that is not refined, return the nbr elem.
        if self.nbr_edge and not self.nbr_edge.children:
            return [self.nbr_edge.elem]

        # If we have a neighbour edge that is refined, return its children.
        if self.nbr_edge and self.nbr_edge.children:
            return [child.elem for child in self.nbr_edge.children]

        # If we have have no neighbouring edge, but our parent does, return this.
        if not self.nbr_edge and self.parent and self.parent.nbr_edge:
            return self.parent.neighbour_elements()

        # Else we do not have neighbours, must be on the boundary.
        assert self.on_boundary and not self.glued
        return []

    def __repr__(self):
        return "Edge({}, {})".format(self.vertices[0], self.vertices[1])


class Element:
    def __init__(self, edges, levels, parent=None):
        """ 
        Edges are in order (v0 v1), (v1, v2), (v2, v3), (v3, v0) 
        Vertices in order of (0, 0), (0, 1), (1, 1), (1, 0).  """
        self.edges = edges
        self.levels = levels
        self.vertices = [edge.vertices[0] for edge in edges]
        self.parent = parent
        self.children = []

        if parent:
            self.gamma_space = parent.gamma_space
        else:
            assert levels == (0, 0)
            self.gamma_space = None
        #print('Create elem with vertices {}'.format(self.vertices))

        # Register ourselves in the edges.
        for edge in edges:
            assert not edge.elem
            edge.elem = self

        # Sanity check.
        for i in range(4):
            assert edges[i - 1].vertices[1] == edges[i].vertices[0]
        assert len(edges) == 4 and len(levels) == 2
        assert self.vertices[0].t == self.vertices[1].t
        assert self.vertices[1].x == self.vertices[2].x
        assert self.vertices[2].t == self.vertices[3].t
        assert self.vertices[3].x == self.vertices[0].x
        assert self.vertices[0].t < self.vertices[2].t
        assert self.vertices[0].x < self.vertices[1].x

        self.center = Vertex(t=(self.vertices[0].t + self.vertices[2].t) / 2,
                             x=(self.vertices[0].x + self.vertices[1].x) / 2,
                             idx=-1)

        self.time_interval = self.vertices[0].t, self.vertices[2].t
        self.space_interval = self.vertices[0].x, self.vertices[2].x
        self.h_t = abs(self.vertices[2].t - self.vertices[0].t)
        self.h_x = abs(self.vertices[2].x - self.vertices[0].x)

    def dist(self, other):
        """ Calculates the distance in the embedded space. """
        assert self.gamma_space and other.gamma_space
        return np.linalg.norm(
            self.gamma_space(self.center) - other.gamma_space(other.center))

    @property
    def level_time(self):
        return self.levels[0]

    @property
    def level_space(self):
        return self.levels[1]

    def edges_axis(self, ax):
        assert 0 <= ax <= 1
        return (self.edges[1 - ax], self.edges[3 - ax])

    def __repr__(self):
        return "Elem(t={}, x={})".format(self.time_interval,
                                         self.space_interval)


class Mesh:
    def __init__(self,
                 glue_space=False,
                 initial_space_mesh=[0, 1],
                 initial_time_mesh=[0, 1]):
        self.glue_space = glue_space

        # Generate all vertices on both time boundaries.
        vertices = []
        for j, t in enumerate(initial_time_mesh):
            for i, x in enumerate(initial_space_mesh):
                vertices.append(Vertex(t=t, x=x, idx=len(vertices)))

        # Generate all the necessary elements + edges.
        roots = []
        N_t = len(initial_time_mesh) - 1
        N_x = len(initial_space_mesh) - 1
        for j in range(N_t):
            for i in range(N_x):
                v0 = vertices[j * len(initial_space_mesh) + i]
                v1 = vertices[j * len(initial_space_mesh) + i + 1]
                v2 = vertices[(j + 1) * len(initial_space_mesh) + i + 1]
                v3 = vertices[(j + 1) * len(initial_space_mesh) + i]

                # Create four edges and the element.
                e1 = Edge(vertices=(v0, v1))
                e2 = Edge(vertices=(v1, v2))
                e3 = Edge(vertices=(v2, v3))
                e4 = Edge(vertices=(v3, v0))
                roots.append(Element(edges=[e1, e2, e3, e4], levels=(0, 0)))
                roots[-1].glob_idx = len(roots) - 1

                # Set boundary edges correctly.
                if j == 0: e1.on_boundary = True
                if i + 1 == N_x: e2.on_boundary = True
                if i == 0: e4.on_boundary = True
                if j + 1 == N_t: e3.on_boundary = True

                # Set the space edge nbrs correctly.
                if i > 0:
                    roots[-2].edges[1].nbr_edge = roots[-1].edges[3]
                    roots[-1].edges[3].nbr_edge = roots[-2].edges[1]

                # Set time edge nbrs correctly.
                if j > 0:
                    roots[(j - 1) * N_x +
                          i].edges[2].nbr_edge = roots[-1].edges[0]
                    roots[-1].edges[0].nbr_edge = roots[(j - 1) * N_x +
                                                        i].edges[2]

            # Glue the outside time edges in case we have a closed manifold.
            if glue_space:
                roots[j * N_x].edges[3].glued = True
                roots[-1].edges[1].glued = True

                roots[j * N_x].edges[3].nbr_edge = roots[-1].edges[1]
                roots[-1].edges[1].nbr_edge = roots[j * N_x].edges[3]

        self.vertices = vertices
        self.roots = roots
        self.leaf_elements = OrderedDict.fromkeys(roots)
        self.N_elements = len(roots)

    def __bisect_edge(self, edge):
        """ Bisects edge and returns the vertex in the middle of edge. """
        assert not edge.children

        # Check if the vertex in the middle already exists.
        if not edge.glued and edge.nbr_edge and edge.nbr_edge.children:
            child_vertex = edge.nbr_edge.children[0].vertices[1]
        else:
            a, b = edge.vertices
            child_vertex = Vertex(t=(a.t + b.t) / 2,
                                  x=(a.x + b.x) / 2,
                                  idx=len(self.vertices))
            self.vertices.append(child_vertex)
            assert (a.t == b.t == child_vertex.t) ^ (a.x == b.x ==
                                                     child_vertex.x)

        edge.bisect(child_vertex)
        return child_vertex

    def __create_edges(self, vertices):
        """ Creates both edges between vertices. """
        e1 = Edge(vertices=(vertices[0], vertices[1]), parent=None)
        e2 = Edge(vertices=(vertices[1], vertices[0]), parent=None)
        e1.nbr_edge = e2
        e2.nbr_edge = e1
        return e1, e2

    def refine_axis(self, elem, ax):
        assert 0 <= ax <= 1

        # Ensure conformity in the current axis.
        for edge in elem.edges:
            for nbr_elem in edge.neighbour_elements():
                if nbr_elem.levels[ax] < elem.levels[ax]:
                    self.refine_axis(nbr_elem, ax)

        # Remove current elem from the currente dges.
        assert not elem.children
        for edge in elem.edges:
            assert edge.elem == elem
            edge.elem = None

        # Lets bisect the edges in the given axis and store new vertices.
        new_vertices = []
        for edge in elem.edges_axis(ax):
            new_vertices.append(self.__bisect_edge(edge))
        assert new_vertices[0].tx[ax] == new_vertices[1].tx[ax]
        assert new_vertices[0].tx[1 - ax] != new_vertices[1].tx[1 - ax]

        # Create the edges between new vertices.
        e1, e2 = self.__create_edges(new_vertices)

        # Create the two new elements
        edges = elem.edges
        if ax == 0:
            # Refining in time
            child1 = Element(edges=(edges[0], edges[1].children[0], e1,
                                    edges[3].children[1]),
                             levels=(elem.level_time + 1, elem.level_space),
                             parent=elem)
            child2 = Element(edges=(e2, edges[1].children[1], edges[2],
                                    edges[3].children[0]),
                             levels=(elem.level_time + 1, elem.level_space),
                             parent=elem)
        else:
            # Refining in space
            child1 = Element(edges=(edges[0].children[0], e1,
                                    edges[2].children[1], edges[3]),
                             levels=(elem.level_time, elem.level_space + 1),
                             parent=elem)
            child2 = Element(edges=(edges[0].children[1], edges[1],
                                    edges[2].children[0], e2),
                             levels=(elem.level_time, elem.level_space + 1),
                             parent=elem)

        child1.glob_idx = self.N_elements
        child2.glob_idx = self.N_elements + 1
        self.N_elements += 2

        # Update datastructures with new elements.
        self.leaf_elements.pop(elem)
        self.leaf_elements.setdefault(child1)
        self.leaf_elements.setdefault(child2)

        elem.children = (child1, child2)
        return elem.children

    def refine_time(self, elem):
        return self.refine_axis(elem, 0)

    def refine_space(self, elem):
        return self.refine_axis(elem, 1)

    def refine(self, elem):
        result = []
        for child in self.refine_time(elem):
            result.extend(self.refine_space(child))
        return result

    def uniform_refine(self):
        leaves = sorted(list(self.leaf_elements),
                        key=lambda elem: elem.level_time)
        for elem in leaves:
            self.refine_time(elem)

        leaves = sorted(list(self.leaf_elements),
                        key=lambda elem: elem.level_space)
        for elem in leaves:
            self.refine_space(elem)

    def uniform_refine_space(self):
        leaves = list(self.leaf_elements)
        for elem in leaves:
            self.refine_space(elem)

    def dorfler_refine_isotropic(self, eta_sqr, theta):
        print('Dorfler marking with theta = {}'.format(theta))
        elems = list(self.leaf_elements)
        N = len(elems)
        assert len(eta_sqr) == N
        s_idx = list(reversed(np.argsort(eta_sqr)))
        eta_tot_sqr = np.sum(eta_sqr)
        cumsum = 0.0
        marked = []
        for i in s_idx:
            marked.append(elems[i])
            cumsum += eta_sqr[i]
            if cumsum >= eta_tot_sqr * theta**2:
                break
        assert np.sqrt(cumsum) >= theta * np.sqrt(eta_tot_sqr)
        print('Marked {} / {} elements'.format(len(marked), N))

        # First refine in time.
        marked.sort(key=lambda elem: elem.level_time)
        children_time = []
        for elem in marked:
            assert not elem.children
            children_time.extend(self.refine_time(elem))

        # Then refine in space.
        children_time.sort(key=lambda elem: elem.level_space)
        for elem in children_time:
            assert not elem.children
            self.refine_space(elem)
        print(
            'Refinement added {} elements'.format(len(self.leaf_elements) - N))

    def dorfler_refine_anisotropic(self, eta_sqr, theta):
        print('Dorfler marking with theta = {}'.format(theta))
        elems = list(self.leaf_elements)
        N = len(elems)
        assert eta_sqr.shape == (N, 2)
        # Concatenate both lists.
        errs = [(val, elem, 0) for val, elem in zip(eta_sqr[:, 0], elems)]
        errs += [(val, elem, 1) for val, elem in zip(eta_sqr[:, 1], elems)]
        errs.sort(reverse=True, key=lambda tup: tup[0])
        eta_tot_sqr = np.sum(eta_sqr)
        cumsum = 0.0
        marked = [[], []]
        for val, elem, refine_axis in errs:
            marked[refine_axis].append(elem)
            cumsum += val
            if cumsum >= eta_tot_sqr * theta**2:
                break
        assert np.sqrt(cumsum) >= theta * np.sqrt(eta_tot_sqr)
        print('Marked {} elements for time refinemenent.'.format(
            len(marked[0]), N))
        print('Marked {} elements for space refinemenent.'.format(
            len(marked[1]), N))

        # First refine in time.
        marked[0].sort(key=lambda elem: elem.level_time)
        for elem in marked[0]:
            assert not elem.children
            self.refine_time(elem)

        # Replace elements marked for space refinement that have been refined
        # by the time refinemenent.
        marked_space = []
        for elem in marked[1]:
            if elem.children:
                marked_space.extend(elem.children)
            else:
                marked_space.append(elem)

        marked_space.sort(key=lambda elem: elem.level_space)
        for elem in marked_space:
            assert not elem.children
            self.refine_space(elem)
        print(
            'Refinement added {} elements'.format(len(self.leaf_elements) - N))

    def refine_grading(self, sigma=2, K=4):
        """ Refines the mesh such that h_t \eqsim h_x**sigma. """
        N = len(self.leaf_elements)
        marked_space = True
        marked_time = True
        while marked_space or marked_time:
            marked_space = []
            marked_time = []
            elems = list(self.leaf_elements)
            for elem in elems:
                if elem.h_t / K >= elem.h_x**sigma:
                    marked_time.append(elem)
                elif elem.h_x**sigma >= K * elem.h_t:
                    marked_space.append(elem)
                else:
                    assert elem.h_t / K < elem.h_x**sigma < K * elem.h_t

            marked_time.sort(key=lambda elem: elem.level_time)
            for elem in marked_time:
                self.refine_time(elem)

            marked_space.sort(key=lambda elem: elem.level_space)
            for elem in marked_space:
                assert not elem.children
                self.refine_space(elem)
        print('Grading added {} elements'.format(len(self.leaf_elements) - N))

    def md5(self):
        return hashlib.md5(self.gmsh().encode()).hexdigest()

    def gmsh(self, use_gamma=False):
        """Returns the (leaf) grid in gmsh format."""
        result = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{}\n".format(
            len(self.vertices))
        if not use_gamma:
            for vertex in self.vertices:
                result += "{} {} {} 0\n".format(vertex.idx + 1, vertex.t,
                                                vertex.x)
        else:
            for vertex in self.vertices:
                x, y = self.gamma_space.eval(vertex.x)[:, 0]
                result += "{} {} {} {}\n".format(vertex.idx + 1, vertex.t, x,
                                                 y)
        result += "$EndNodes\n$Elements\n{}\n".format(len(self.leaf_elements))
        for idx, element in enumerate(self.leaf_elements):
            result += "{} 3 2 0 0 {} {} {} {}\n".format(
                idx + 1, element.vertices[0].idx + 1,
                element.vertices[1].idx + 1, element.vertices[2].idx + 1,
                element.vertices[3].idx + 1)
        result += "$EndElements\n"
        return result


class MeshParametrized(Mesh):
    def __init__(self,
                 gamma_space,
                 initial_space_mesh=None,
                 initial_time_mesh=[0, 1]):
        assert isinstance(gamma_space, PiecewiseParametrization)
        if initial_space_mesh is None:
            initial_space_mesh = gamma_space.pw_start

        assert initial_space_mesh[0] == 0
        assert initial_space_mesh[-1] == gamma_space.pw_start[-1]

        super().__init__(glue_space=gamma_space.closed,
                         initial_space_mesh=initial_space_mesh,
                         initial_time_mesh=initial_time_mesh)
        self.gamma_space = gamma_space
        for elem in self.roots:
            for i in range(len(gamma_space.pw_start)):
                if gamma_space.pw_start[i] <= elem.vertices[
                        0].x < gamma_space.pw_start[i + 1]:
                    elem.gamma_space = gamma_space.pw_gamma[i]
                    break
            assert elem.gamma_space

        # Count vertices.
        assert len([vtx for vtx in self.vertices
                    if vtx.x == 0]) == len(initial_time_mesh)
        assert len([vtx for vtx in self.vertices
                    if vtx.t == 0]) == len(initial_space_mesh)
        assert len([
            vtx for vtx in self.vertices
            if vtx.x == self.gamma_space.gamma_length
        ]) == len(initial_time_mesh)
        assert len([
            vtx for vtx in self.vertices if vtx.t == initial_time_mesh[-1]
        ]) == len(initial_space_mesh)

        # Ensure that the initial space consists at least of three elements.
        if self.glue_space and len(self.roots) < 3:
            for elem in self.roots:
                self.refine_space(elem)
            leaves = list(self.leaf_elements)
            for elem in leaves:
                self.refine_space(elem)


def Prolongate(vec_coarse, elems_coarse, elems_fine):
    """ Helper function to prolongate a vector. """
    elem_coarse_2_idx = {k: v for v, k in enumerate(elems_coarse)}
    vec_fine = np.zeros(len(elems_fine))

    for j, elem_fine in enumerate(elems_fine):
        elem_coarse = elem_fine
        while elem_coarse not in elem_coarse_2_idx:
            assert elem_coarse.parent
            elem_coarse = elem_coarse.parent
        assert elem_coarse in elem_coarse_2_idx
        i = elem_coarse_2_idx[elem_coarse]
        vec_fine[j] = vec_coarse[i]
    return vec_fine


if __name__ == "__main__":
    gamma = UnitSquare()
    mesh = MeshParametrized(gamma)
    np.random.seed(5)
    for k in range(5):
        eta = np.random.rand(len(mesh.leaf_elements), 2)
        mesh.dorfler_refine_anisotropic(eta, 0.6)

#    mesh = Mesh()
#    mesh.uniform_refine_space()
#    print(mesh.leaf_elements)
#    asdf
#    #random.seed(5)
#    #for _ in range(20):
#    #    elem = random.choice(list(mesh.leaf_elements))
#    #    mesh.refine_axis(elem, random.random() < 0.5)
#    mesh = MeshParametrized(Circle())
#    #mesh.uniform_refine()
#    #mesh.uniform_refine()
#    #mesh.uniform_refine()
#    #random.seed(5)
#    #for _ in range(20):
#    #    elem = random.choice(list(mesh.leaf_elements))
#    #    mesh.refine_axis(elem, random.random() < 0.5)
#    print(mesh.gmsh())
