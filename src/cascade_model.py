import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import json

import config

from multiprocessing import Pool
from collections import defaultdict

from src.utils import generate_random_value, split_list


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


class Cascader:
    def __init__(self, data_folder, metrics_file_path=None, statistics_file_path=None, cpu_count=None):
        """

        :param data_folder: folder containing the list of cascade files. File extension must be .edgelist.

        :param metrics_file_path: file that will be used to save list of metrics. File extension must be .json.
        Format: {
            "$cascade_size1": [($average_path1, $degree_variance1), ($average_path2, $degree_variance2), ...],
        }

        :param statistics_file_path: file that will be used to save list of average statistics. File extension must be .json.
        Format: {
            "$cascade_size1": {"average_path": $value1, "$degree_variance": $value2},
        }
        :param cpu_count: number of cpu that will be used.
        """
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        self.data_folder = data_folder
        self.metrics_file_path = metrics_file_path or os.path.join(data_folder, 'metrics.json')
        self.statistics_file_path = statistics_file_path or os.path.join(data_folder, 'statistics.json')
        self.cpu_count = cpu_count or os.cpu_count()

    @staticmethod
    def list_of_degrees(graph):
        return list(map(lambda node: node[1], graph.degree()))

    def generate_cascades(self, sizes, count_per_size, theta_power=0):
        for size in sizes:
            for i in range(count_per_size):
                graph = MyDiGraph.generate_cascade(size, theta_power)
                filename = '{size}_{i}_of_{count_per_size}.edgelist'.format(size=size, i=i + 1,
                                                                            count_per_size=count_per_size)
                nx.write_edgelist(graph, os.path.join(self.data_folder, filename))

    @staticmethod
    def _analyze_cascades(list_of_filenames):
        statistics = defaultdict(list)
        for filename in list_of_filenames:
            cascade = nx.read_edgelist(filename)
            statistics[nx.number_of_nodes(cascade)].append((nx.average_shortest_path_length(cascade),
                                                            np.var(Cascader.list_of_degrees(cascade))))
        return statistics

    def analyze(self):
        overall_statistics = defaultdict(list)
        filenames = [os.path.join(self.data_folder, filename)
                     for filename in os.listdir(self.data_folder) if filename.endswith('edgelist')]

        filenames = split_list(filenames, self.cpu_count)

        with Pool() as pool:
            statistics_list = pool.map(self._analyze_cascades, filenames)

        for statistics in statistics_list:
            for size in statistics:
                overall_statistics[size] += statistics[size]

        with open(self.metrics_file_path, 'w') as fp:
            json.dump(overall_statistics, fp)

        average_statistics = defaultdict(defaultdict)

        for size in overall_statistics:

            list_of_average_pathes = list(map(lambda metrics: metrics[0], overall_statistics[size]))
            list_of_degree_variance = list(map(lambda metrics: metrics[1], overall_statistics[size]))

            average_statistics[size]['average_path'] = sum(list_of_average_pathes) / len(list_of_average_pathes)
            average_statistics[size]['degree_variance'] = sum(list_of_degree_variance) / len(list_of_degree_variance)
            average_statistics[size]['average_path_std'] = np.std(list_of_average_pathes)
            average_statistics[size]['degree_variance_std'] = np.std(list_of_degree_variance)

        with open(self.statistics_file_path, 'w') as fp:
            json.dump(average_statistics, fp)

    def show_cascades_plots(self):
        with open(os.path.join(self.statistics_file_path)) as fp:
            data = json.load(fp)

        x = sorted(map(int, data.keys()))
        average_path = list(map(lambda size: data[str(size)]['average_path'], x))
        plt.title('Average path length')
        plt.plot(x, average_path)
        plt.show()


def make_list_of_dirs(theta_list):
    result_list = []

    for theta in theta_list:
        folder_for_theta = 'theta_{}'.format('_'.join(str(theta).split('.')))
        data_folder = os.path.join(config.DATA_DIR, folder_for_theta)
        result_list.append(data_folder)

    return result_list


def prepare_data_for_plot(list_of_dirs, vk_folder, field):
    total_data = []
    list_of_dirs += [vk_folder]
    for folder in list_of_dirs:
        statistics_file = os.path.join(folder, 'statistics.json')
        with open(statistics_file) as fp:
            local_data = json.load(fp)

        x = sorted(map(int, local_data.keys()))
        average_path = list(map(lambda size: local_data[str(size)][field], x))
        total_data += [x, average_path]

    with open(os.path.join(vk_folder, 'statistics.json')) as fp:
        vk_data = json.load(fp)
        e = [vk_data[str(size)][field + '_std'] for size in sorted(list(map(int, vk_data.keys())))]

    total_data += [e]
    return total_data


def generate_cascades_by_theta_list(theta_list, folder_list):
    for i, theta in enumerate(theta_list):
        data_folder = folder_list[i]
        data_metrics = os.path.join(data_folder, 'list_of_metrics.json')
        data_statistics = os.path.join(data_folder, 'statistics.json')

        analyzer = Cascader(data_folder, data_metrics, data_statistics)
        analyzer.generate_cascades([10, 20, 30, 40, 50], 100, theta_power=theta)
        analyzer.analyze()
        print(theta, 'ready')


def analyze_vk_data():
    data_folder = os.path.join(config.DATA_DIR, 'cascades')
    data_metrics = os.path.join(data_folder, 'list_of_metrics.json')
    data_statistics = os.path.join(data_folder, 'statistics.json')
    analyzer = Cascader(data_folder, data_metrics, data_statistics)
    analyzer.analyze()
    # analyzer.show_cascades_plots()


VK_DATA_FOLDER = os.path.join(config.DATA_DIR, 'cascades')
THETA = [1.6, 1.8, 2, 2.2]


def plot_all_data(data, title):
    plt.title(title)

    for i, theta in enumerate(THETA):
        plt.plot(data[2 * i], data[2 * i + 1])

    plt.errorbar(data[-3], data[-2], data[-1], fmt='bs--')

    plt.legend(['theta = {}'.format(str(theta)) for theta in THETA] + ['vk'])
    plt.grid()

    plt.xlabel('Size of cascade')
    plt.ylabel(title.lower())

    plt.show()


degree_field = 'degree_variance'
degree_title = 'Degree variance'

path_field = 'average_path'
path_title = 'Shortest average path'

if __name__ == '__main__':
    field = degree_field
    title = degree_title
    dirs_list = make_list_of_dirs(THETA)
    # generate_cascades_by_theta_list(THETA, dirs_list)
    prepared_data = prepare_data_for_plot(dirs_list, VK_DATA_FOLDER, field)
    plot_all_data(prepared_data, title)
    # analyze_vk_data()
