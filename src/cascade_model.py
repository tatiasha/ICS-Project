import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def generate_random_value(probabilities):
    return np.random.choice(range(len(probabilities)), p=probabilities)


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    """
    If there is a cycle that is reachable from root, then this will see infinite recursion.
    G: the graph
    root: the root node of current branch
    width: horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    pos: a dict saying where all nodes go if they have been assigned
    parent: parent of this branch.

    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = list(G.adj[root].keys())
    if parent in neighbors:  # this should be removed for directed graphs.
        neighbors.remove(parent)  # if directed, then parent not in neighbors.
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                parent=root)
    return pos


class MyDiGraph(nx.DiGraph):
    def compute_degree_theta_sum(self, theta_power):
        return sum(map(lambda n: self.degree(n) ** theta_power, self.nodes))

    def _calculate_probability(self, node, theta_power, degree_sum=None):
        if self.degree(node) == 0:
            return 0

        if degree_sum is None:
            degree_sum = self.compute_degree_theta_sum(theta_power)

        return self.degree(node) ** theta_power / degree_sum

    def calculate_probabilities(self, theta_power):
        degree_sum = self.compute_degree_theta_sum(theta_power)
        return [self._calculate_probability(node, theta_power, degree_sum) for node in self.nodes]

    @classmethod
    def generate_cascade(cls, size, theta_power):
        graph = cls()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1)

        for new_node in range(2, size):
            probabilities = graph.calculate_probabilities(theta_power)
            graph_node = generate_random_value(probabilities)
            graph.add_node(new_node)
            graph.add_edge(graph_node, new_node)

        return graph


theta = 1
N = 100


if __name__ == '__main__':
    cascade = MyDiGraph.generate_cascade(N, theta)
    pos = hierarchy_pos(cascade, 0)
    nx.draw(cascade, pos=pos, with_labels=True)
    plt.show()
