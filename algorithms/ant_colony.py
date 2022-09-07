import numpy as np

from copy import deepcopy
from random import choices
from collections import defaultdict

from core import MNPAlgorithm
from algorithms.simple_heuristics import SimpleHeuristics


class AntColonyOptimization(MNPAlgorithm):

    def __init__(self, epochs: int = 100, ant_soldiers: int = 50, ant_queens: int = 10, alpha: float = 0.5,
                 forgetfulness: float = 0.05, q: float = 0.1, **kwargs):
        super(AntColonyOptimization, self).__init__(**kwargs)
        self.epochs = epochs
        self.ant_soldiers = ant_soldiers
        self.ant_queens = ant_queens
        self.alpha = alpha
        self.forgetfulness = forgetfulness
        self.q = q
        self.default_p = 1
        self.tau = defaultdict(lambda: self.default_p)
        self.object_SH = SimpleHeuristics('greedy', **kwargs)

    def execute_algorithm(self, data: list):
        data = sorted(data, reverse=True)
        best_split = self.get_perfect_split(data)
        _of_best, _best_state = self.object_SH.execute_algorithm(data)
        self.update_tau([_best_state], best_split)

        epoch = 1
        while epoch <= self.epochs:
            all_ways = [self.find_path(data, 0) for _ in range(self.ant_soldiers)] + \
                       [self.find_path(data, 1) for _ in range(self.ant_queens)]

            state = self.update_tau(all_ways, best_split)
            if state:
                return 0, state

            _of = [self.objective_function(state, best_split) for state in all_ways]
            idx = np.argmin(_of)
            if _of[idx] < _of_best:
                _best_state = all_ways[idx]
                _of_best = _of[idx]

            epoch += 1

        return _of_best, _best_state

    # ###############################################################################################################
    @staticmethod
    def add_elem(state, el, idx_to_add):
        state_copy = deepcopy(state)
        state_copy[idx_to_add].append(el)

        return state_copy

    def get_possible_ways(self, state, el):
        return [
            self.add_elem(state, el, m)
            for m in range(self.number_of_sets)
        ]

    def find_path(self, data: list, ant: int = 0):
        current_subsets = [[] for _ in range(self.number_of_sets)]
        for el in data:
            possible_ways = self.get_possible_ways(current_subsets, el)
            sum_ = list(map(lambda sets: [sum(set_) for set_ in sets], possible_ways))
            n = np.array(list(map(
                lambda set_: 999999 if max(set_) - min(set_) == 0 else 1 / (max(set_) - min(set_)),
                sum_
            )))

            tau_lst = np.array([
                self.tau.get((m, el)) if self.tau.get((m, el)) else self.default_p
                for m in range(self.number_of_sets)
            ])
            _p = self.alpha * tau_lst + (1 - self.alpha) * n

            if ant == 0:
                chosen_path = choices(range(self.number_of_sets), weights=_p / sum(_p))[0]
            else:
                chosen_path = np.argmax(_p)

            current_subsets = possible_ways[chosen_path]

        return current_subsets

    def update_tau(self, states, best_split):
        for key in self.tau:
                self.tau[key] *= (1 - self.forgetfulness)

        for state in states:
            _of = self.objective_function(state, best_split)
            if _of == 0:
                return state
            for m in range(self.number_of_sets):
                for el in state[m]:
                    self.tau[(m, el)] += self.q / _of
