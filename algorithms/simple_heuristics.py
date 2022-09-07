import numpy as np
from random import sample
from itertools import chain
from copy import deepcopy

from core import MNPAlgorithm


class SimpleHeuristics(MNPAlgorithm):
    def __init__(self, method: str = "greedy", **kwargs):
        super(SimpleHeuristics, self).__init__(**kwargs)
        self.method = method

    def greedy(self, data: list):
        _sums = np.zeros(self.number_of_sets)
        sets = [[] for _ in _sums]

        for elem in sorted(data, reverse=True):
            idx = np.argmin(_sums)
            sets[idx].append(elem)
            _sums[idx] += elem

        _of = self.objective_function(sets, self.get_perfect_split(data))

        return _of, sets

    def random_split(self, data: list):
        n = len(data)
        m = self.number_of_sets
        jumbled_data = sample(data, n)

        split = [0] + sorted(sample(range(1, n), m - 1)) + [n]
        sets = [jumbled_data[split[i]: split[i + 1]] for i in range(m)]

        _of = self.objective_function(sets, self.get_perfect_split(data))

        return _of, sets

    def execute_algorithm(self, data: list):
        if self.method == "greedy":
            return self.greedy(data)

        elif self.method == "random_split":
            return self.random_split(data)

        else:
            return "You can only choose: greedy, random_split"


class SmallChanges:
    def __init__(self, state):
        self.state = state

    def change_elem(self, elem, idx_to_add, idx_to_remove):
        state_copy = deepcopy(self.state)
        state_copy[idx_to_add].append(elem)
        state_copy[idx_to_remove].remove(elem)

        return state_copy

    def neighborhood_generation(self) -> list:
        idx_max_set = np.argmax(list(map(sum, self.state)))
        sets = [
            [
                self.change_elem(elem, idx_to_add, idx_max_set)
                for elem in self.state[idx_max_set]
            ]
            for idx_to_add in range(len(self.state)) if idx_to_add != idx_max_set
        ]

        return list(chain.from_iterable(sets))

    def swap(self, method: str = 'max-min'):
        state_copy = deepcopy(self.state)
        sum_list = list(map(sum, state_copy))
        if method == 'max-min':
            i1, i2 = np.argmax(sum_list), np.argmin(sum_list)
        elif method == 'max-random':
            i1, i2 = np.argmax(sum_list), np.random.randint(len(state_copy))
        else:
            i1, i2 = np.random.randint(len(state_copy), size=2)
        j1, j2 = list(map(lambda x: np.random.randint(len(state_copy[x])), [i1, i2]))

        state_copy[i1][j1], state_copy[i2][j2] = state_copy[i2][j2], state_copy[i1][j1]

        return state_copy
