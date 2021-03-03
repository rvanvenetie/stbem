import random


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
                #assert self.vertices[0] == self.nbr_edge.vertices[1]
                #assert self.vertices[1] == self.nbr_edge.vertices[0]
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


class Element:
    def __init__(self, edges, levels, parent=None):
        """ 
        Edges are in order (v0 v1), (v1, v2), (v2, v3), (v3, v0) 
        Vertices in order of (0, 0), (0, 1), (1, 1), (1, 0).  """
        self.edges = edges
        self.levels = levels
        self.vertices = [edge.vertices[0] for edge in edges]
        self.parent = None
        self.children = []

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
        return "Elem{}".format(self.vertices)


class Mesh:
    def __init__(self, glue_space=False):
        self.vertices = [
            Vertex(t=0., x=0., idx=0),
            Vertex(t=0., x=1., idx=1),
            Vertex(t=1., x=1., idx=2),
            Vertex(t=1., x=0., idx=3),
        ]

        root_edges = [
            Edge(vertices=(self.vertices[i], self.vertices[(i + 1) % 4]),
                 parent=None) for i in range(4)
        ]

        for edge in root_edges:
            edge.on_boundary = True

        # Glue edges 1 and 3 together.
        if glue_space:
            root_edges[1].glued = True
            root_edges[3].glued = True

            root_edges[1].nbr_edge = root_edges[3]
            root_edges[3].nbr_edge = root_edges[1]

        self.root = Element(root_edges, levels=(0, 0))
        self.leaf_elements = set([self.root])

    def __bisect_edge(self, edge):
        """ Bisects edge and returns the vertex in the middle of edge. """
        assert not edge.children

        # Check if the vertex in the middle already exists.
        if not edge.glued and edge.nbr_edge and edge.nbr_edge.children:
            child_vertex = edge.nbr_edge.children[0].vertices[1]
        else:
            a, b = edge.vertices
            child_vertex = Vertex(t=0.5 * (a.t + b.t),
                                  x=0.5 * (a.x + b.x),
                                  idx=len(self.vertices))
            self.vertices.append(child_vertex)

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

        # Create the edges between new vertices.
        e1, e2 = self.__create_edges(new_vertices)

        # Create the two new elements
        edges = elem.edges
        if ax == 0:
            # Refining in time
            child1 = Element(edges=(edges[0], edges[1].children[0], e1,
                                    edges[3].children[1]),
                             levels=(elem.level_time + 1, elem.level_space))
            child2 = Element(edges=(e2, edges[1].children[1], edges[2],
                                    edges[3].children[0]),
                             levels=(elem.level_time + 1, elem.level_space))
        else:
            # Refining in space
            child1 = Element(edges=(edges[0].children[0], e1,
                                    edges[2].children[1], edges[3]),
                             levels=(elem.level_time, elem.level_space + 1))
            child2 = Element(edges=(edges[0].children[1], edges[1],
                                    edges[2].children[0], e2),
                             levels=(elem.level_time, elem.level_space + 1))

        # Update datastructures with new elements.
        self.leaf_elements.remove(elem)
        self.leaf_elements.add(child1)
        self.leaf_elements.add(child2)

        child1.parent = elem
        child2.parent = elem
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
        leaves = list(self.leaf_elements)
        for elem in leaves:
            self.refine(elem)

    def gmsh(self):
        """Returns the (leaf) grid in gmsh format."""
        result = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{}\n".format(
            len(self.vertices))
        for vertex in self.vertices:
            result += "{} {} {} 0\n".format(vertex.idx + 1, vertex.x, vertex.t)
        result += "$EndNodes\n$Elements\n{}\n".format(len(self.leaf_elements))
        for idx, element in enumerate(self.leaf_elements):
            result += "{} 3 2 0 0 {} {} {} {}\n".format(
                idx + 1, element.vertices[0].idx + 1,
                element.vertices[1].idx + 1, element.vertices[2].idx + 1,
                element.vertices[3].idx + 1)
        result += "$EndElements\n"
        return result


if __name__ == "__main__":
    mesh = Mesh(glue_space=True)
    random.seed(5)
    for _ in range(20):
        elem = random.choice(list(mesh.leaf_elements))
        mesh.refine_axis(elem, random.random() < 0.5)
    print(mesh.gmsh())
