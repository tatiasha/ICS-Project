import networkx as nx
import math
import matplotlib.pyplot as plt
import random
import csv


def get_model(time, probability_p, probability_q, probability_d, per):
    probability_r = 1.0 - probability_p - probability_q
    G = nx.DiGraph()
    G.add_node(0)
    per = time * (per / 100.0)
    """
    At each time step
    with probability p, a new node is created attaching to a directed node,
    with probability q, a new node is created attached by a directed link 
    with probability r, a directed link is created between the old nodes.
    """
    for step in range(time):

        array_node = list(G.nodes)
        new_in = []
        new_out = []
        D = []

        for i in array_node:
            if G.in_degree(i) > 0:
                new_in.append(i)
            if G.out_degree(i) > per:
                new_out.append(i)
            if G.degree(i) < per:
                D.append(i)

        random_value_a = random.random()
        random_value_b = random.random()
        random_value_c = random.random()
        random_value_d = random.random()

        if random_value_a < probability_p:
            node_i = random.choice(new_in) if new_in else random.choice(array_node)
            G.add_node(step)
            G.add_edge(node_i, step)

        if random_value_b < probability_q:
            node_i = random.choice(array_node) if new_out else random.choice(array_node)
            G.add_node(step)
            G.add_edge(step, node_i)

        if random_value_c < probability_r:
            node1 = random.choice(array_node)
            node2 = random.choice(array_node)
            G.add_edge(node1, node2)

        if D and random_value_d < probability_d:
            node_i = random.choice(D)
            G.remove_node(node_i)

    # nx.draw(G, with_labels=True, node_color = "blue", node_shape='o', alpha = 0.9, linewidths = 6)
    # plt.show()
    return G


def get_lambda(degrees):
    exp = sum(map(lambda degree: math.log10(degree / degrees[0]), degrees))
    exp = 1 + len(degrees) / exp
    return exp


def plot_degree(exp, degrees, degrees_list):
    degrees_list_length = len(degrees_list)
    degrees_length = len(degrees)

    probability = [degrees.count(degrees_list[i]) / float(degrees_length) for i in range(degrees_list_length)]
    y_lambda = [j ** -exp for j in degrees_list]

    plt.scatter(degrees_list, probability, c='b', marker='x', label="Users activity")
    plt.plot(degrees_list, y_lambda, label='Lambda = -' + str(exp), color="orange")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log)')
    plt.ylabel('Probability (log)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def save_network(file_path, network):
    with open(file_path, 'w') as f:
        f.write('\n')  # We write the blank line at the top
        writer_n = csv.writer(f, delimiter=',')
        writer_n.writerow(['Source', 'Target'])
        writer_n.writerows(network.edges())


if __name__ == "__main__":
    probability_p = 0.067
    probability_q = 0.042  # q<p
    probability_d = 0.002
    time = 100
    per = 0.1
    network = get_model(time, probability_p, probability_q, probability_d, per)

    network_degrees = [network.degree(x) for x in network.nodes]
    network_degrees.sort()
    network_degrees_list = list(set(network_degrees))
    network_degrees_list.sort()

    lmd = get_lambda(network_degrees_list)
    plot_degree(lmd, network_degrees, network_degrees_list)

    # path = ""
    # saveNetwork(path, network)
