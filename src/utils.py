import numpy as np


def generate_random_value(probabilities):
    return np.random.choice(range(len(probabilities)), p=probabilities)


def split_list(a, number_of_parts):
    k, m = divmod(len(a), number_of_parts)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(number_of_parts)]