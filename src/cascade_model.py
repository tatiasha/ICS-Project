import networkx as nx
import matplotlib.pyplot as plt
import random


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                  pos=None, parent=None):
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


theta = 0.1
N = 100


def compute_degree_theta_sum(graph, theta_power):
    return sum(map(lambda node: graph.degree(node) ** theta_power, graph.nodes))


G = nx.DiGraph()
G.add_node(0)
G.add_node(1)
G.add_edge(0, 1)

for i in range(2, N):

    random_value = random.uniform(0, 1)
    new = [x for x in G.nodes if len(G.succ[x].keys()) < 3]
    random_value1 = random.uniform(0, 1)
    if random_value1 < 0.5:
        for it in range(len(list(new))):
            node_i = random.choice(list(new))
            temp = compute_degree_theta_sum(G, theta)
            probability = G.degree(node_i) ** theta / compute_degree_theta_sum(G, theta)
            # print(probability)
            if (probability > random_value) and i not in list(G.nodes):
                G.add_node(i)
                G.add_edge(node_i, i)
    else:
        for node_i in list(set(G.nodes) - set(new)):
            temp = compute_degree_theta_sum(G, theta)
            probability = G.degree(node_i) ** theta / compute_degree_theta_sum(G, theta)
            # print(probability)
            if (probability > random_value) and i not in list(G.nodes):
                G.add_node(i)
                G.add_edge(node_i, i)
print(set(G.nodes) - set(new), new, G.nodes)
pos = hierarchy_pos(G, 0)
nx.draw(G, pos=pos, with_labels=True)
plt.show()
print(nx.average_shortest_path_length(G))
