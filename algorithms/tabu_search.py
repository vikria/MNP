from core import MNPAlgorithm
from algorithms.simple_heuristics import SimpleHeuristics, SmallChanges
from random import sample


class TabuSearch(MNPAlgorithm):

    def __init__(self, max_tabu_size: int = 10, max_iter: int = 300, start_method: str = 'greedy',
                 method_changes: str = 'max-min', **kwargs):
        super(TabuSearch, self).__init__(**kwargs)
        self.max_tabu_size = max_tabu_size
        self.max_iter = max_iter
        self.start_method = start_method
        self.method_changes = method_changes
        self.object_SH = SimpleHeuristics(self.start_method, **kwargs)

    def execute_algorithm(self, data: list):
        best_split = self.get_perfect_split(data)
        _of, best_state = self.object_SH.execute_algorithm(data)
        idx_add, idx_remove = sample(range(self.number_of_sets), 2)
        if self.method_changes == 'one-move':
            best_candidate = SmallChanges(best_state).change_elem(sample(best_state[idx_remove], 1)[0],
                                                                  idx_add, idx_remove)
        else:
            best_candidate = SmallChanges(best_state).swap(method=self.method_changes)

        tabu_list = [best_state]
        it = 1
        while it < self.max_iter:
            neighborhood = SmallChanges(best_candidate).neighborhood_generation()
            for candidate in neighborhood:
                if candidate not in tabu_list and \
                        self.objective_function(candidate, best_split) <= self.objective_function(best_candidate,
                                                                                                  best_split):
                    best_candidate = candidate

            if best_candidate in tabu_list:
                if self.method_changes == 'one-move':
                    best_candidate = SmallChanges(best_candidate).change_elem(sample(best_candidate[idx_remove], 1)[0],
                                                                              idx_add, idx_remove)
                else:
                    best_candidate = SmallChanges(best_candidate).swap(method=self.method_changes)

                it += 1
                continue

            if self.objective_function(best_candidate, best_split) <= self.objective_function(best_state, best_split):
                best_state = best_candidate

            tabu_list.append(best_candidate)

            if len(tabu_list) > self.max_tabu_size:
                tabu_list = tabu_list[1:]

            it += 1

        _of = self.objective_function(best_state, best_split)

        return _of, best_state
