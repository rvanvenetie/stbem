import numpy as np


class Vertex:
    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.xy = (x, y)
        self.xy_np = np.array([[x], [y]])
        self.idx = idx

    def __repr__(self):
        return "({},{})".format(self.x, self.y)


class Element:
    def __init__(self, vertices, parent=None):
        self.vertices = vertices
        self.parent = parent

        if parent:
            self.level = parent.level + 1
        else:
            self.level = 0

        assert vertices[0].y == vertices[1].y
        assert vertices[1].x == vertices[2].x
        assert vertices[2].y == vertices[3].y
        assert vertices[3].x == vertices[0].x

        # Check that we are square.
        assert self.vertices[1].x - self.vertices[0].x == self.vertices[
            3].y - self.vertices[0].y

    @property
    def edges(self):
        return [(self.vertices[0], self.vertices[1]),
                (self.vertices[1], self.vertices[2]),
                (self.vertices[2], self.vertices[3]),
                (self.vertices[3], self.vertices[0])]

    @property
    def diam(self):
        return self.vertices[1].x - self.vertices[0].x

    # Check whether this element contains a given point.
    def contains(self, pt):
        return self.vertices[0].x <= pt[0] <= self.vertices[
            2].x and self.vertices[0].y <= pt[1] <= self.vertices[2].y

    # Returns parametrization of [0,1]^2 to this elem.
    def gamma(self):
        n0 = self.vertices[0].xy_np
        n1 = self.vertices[1].xy_np
        n3 = self.vertices[3].xy_np
        return lambda x, z: n0 + (n1 - n0) * x + (n3 - n0) * z

    # Returns the two vertices sharing a boundary with the given vertex.
    def connected_to_vertex(self, vertex):
        assert vertex in self.vertices
        result = []
        for vtx in self.vertices:
            if (vtx.x == vertex.x) ^ (vtx.y == vertex.y):
                result.append(vtx)
        assert len(result) == 2
        return result

    def __repr__(self):
        return "Elem({}, {})".format(self.vertices[0], self.vertices[2])


class InitialMesh:
    def __init__(self, vertices, elements):
        self.vertices = []
        for i, xy in enumerate(vertices):
            self.vertices.append(Vertex(x=xy[0], y=xy[1], idx=i))
        self.__bisect_edge = {}
        self.parent_edge = {}
        self.elements = []
        self.leaf_elements = set()
        self.nbrs = {}
        for v0, v1, v2, v3 in elements:
            self.elements.append(
                Element(vertices=(self.vertices[v0], self.vertices[v1],
                                  self.vertices[v2], self.vertices[v3])))
            self.leaf_elements.add(self.elements[-1])

        for elem in self.elements:
            for edge in elem.edges:
                self.nbrs[edge] = elem

    def vertex_from_coords(self, xy):
        for vtx in self.vertices:
            if vtx.x == xy[0] and vtx.y == xy[1]: return vtx
        return None

    def bisect_edge(self, a, b):
        assert not (a, b) in self.__bisect_edge

        if (b, a) in self.__bisect_edge:
            new_vtx = self.__bisect_edge[(b, a)]
        else:
            new_vtx = Vertex(x=(a.x + b.x) / 2,
                             y=(a.y + b.y) / 2,
                             idx=len(self.vertices))
            self.vertices.append(new_vtx)

        self.__bisect_edge[(a, b)] = new_vtx
        self.parent_edge[(a, new_vtx)] = (a, b)
        self.parent_edge[(new_vtx, b)] = (a, b)

        return new_vtx

    def refine(self, element):
        # Check neighbours.
        for a, b in element.edges:
            # If we have no neighbours along this edge, first refine the nbr.
            if not (b, a) in self.nbrs:
                # Root edge does not need to be refined.
                if not (a, b) in self.parent_edge: continue

                # Find the parent edge.
                pa, pb = self.parent_edge[(a, b)]

                # Check if we have a nbr along this edge, if not it is on bdr.
                if (pb, pa) in self.nbrs:
                    assert self.nbrs[(pb, pa)].level == element.level - 1
                    self.refine(self.nbrs[(pb, pa)])

        # Unpack vertices.
        v0, v1, v2, v3 = element.vertices

        # Bisect all edges.
        v01 = self.bisect_edge(v0, v1)
        v12 = self.bisect_edge(v1, v2)
        v23 = self.bisect_edge(v2, v3)
        v30 = self.bisect_edge(v3, v0)

        # Create interior vertex.
        vi = Vertex(x=(v0.x + v2.x) / 2,
                    y=(v0.y + v2.y) / 2,
                    idx=len(self.vertices))
        self.vertices.append(vi)

        # Create the four children.
        children = [
            Element(vertices=[v0, v01, vi, v30], parent=element),
            Element(vertices=[v01, v1, v12, vi], parent=element),
            Element(vertices=[vi, v12, v2, v23], parent=element),
            Element(vertices=[v30, vi, v23, v3], parent=element)
        ]

        # Register ourselves in the nbrs list.
        for child in children:
            for edge in child.edges:
                self.nbrs[edge] = child

        # Update leaf elements
        self.elements.extend(children)
        self.leaf_elements.remove(element)
        self.leaf_elements.update(children)
        return children

    def uniform_refine(self):
        leaves = list(self.leaf_elements)
        for elem in leaves:
            self.refine(elem)

    def refine_msh_bdr(self, v0, v1):
        """ Locally refines the mesh until it contains an element (touching the boundary)
            that has an edge that coincides with the given edge v0 <--> v1. """
        # Cast the input to a vector.
        v0 = np.array(v0).reshape(-1, 1)
        v1 = np.array(v1).reshape(-1, 1)

        # Sort the input
        v0, v1 = sorted([v0, v1], key=lambda v: abs(v[0]) + abs(v[1]))
        axis = None
        for i in range(2):
            if v0[i] == v1[i]:
                axis = i
        assert axis is not None
        n_axis = int(not axis)

        # Start with all elements.
        children = self.leaf_elements
        while True:
            parent = None

            # Find the element that contains edge v0 -- v1.
            for elem in children:
                for a, b in elem.edges:
                    # Find corresponding x,y coords, and sort them.
                    va, vb = sorted([a.xy, b.xy])

                    # Check that we lie on correct edge.
                    if not (v0[axis] == va[axis] == vb[axis]):
                        continue

                    # Check whether v0 v1 is contained in other edge.
                    if va[n_axis] <= v0[n_axis] <= v1[n_axis] <= vb[n_axis]:
                        # If this elements edge coincides with v0, v1, return!
                        if va[n_axis] == v0[n_axis] and v1[n_axis] == vb[
                                n_axis]:
                            return elem
                        parent = elem

            assert parent
            children = self.refine(parent)

    def gmsh(self):
        """Returns the (leaf) grid in gmsh format."""
        result = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{}\n".format(
            len(self.vertices))
        for i, vtx in enumerate(self.vertices):
            result += "{} {} {} 0\n".format(i + 1, vtx.x, vtx.y)

        result += "$EndNodes\n$Elements\n{}\n".format(len(self.leaf_elements))
        for idx, element in enumerate(self.leaf_elements):
            result += "{} 3 2 0 0 {} {} {} {}\n".format(
                idx + 1, element.vertices[0].idx + 1,
                element.vertices[1].idx + 1, element.vertices[2].idx + 1,
                element.vertices[3].idx + 1)
        result += "$EndElements\n"
        return result


def UnitSquare():
    return InitialMesh(vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
                       elements=[(0, 1, 2, 3)])


def UnitSquareBoundaryRefined(v0, v1):
    mesh = UnitSquare()
    mesh.refine_msh_bdr(v0, v1)
    return mesh


if __name__ == "__main__":
    mesh = UnitSquare()
    mesh.uniform_refine()
    mesh.refine_msh_bdr((1, 0.53125), (1, 0.5))
