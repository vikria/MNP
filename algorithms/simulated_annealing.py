import numpy as np

from core import MNPAlgorithm
from algorithms.simple_heuristics import SimpleHeuristics, SmallChanges


class SimulatedAnnealing(MNPAlgorithm):

    def __init__(self, t_min: int = 10, t_max: int = 1000, max_iter: int = 300, start_method: str = 'greedy',
                 method_changes: str = 'max-min', **kwargs):
        super(SimulatedAnnealing, self).__init__(**kwargs)
        self.t_min = t_min
        self.t_max = t_max
        self.max_iter = max_iter
        self.start_method = start_method
        self.method_changes = method_changes
        self.object_SH = SimpleHeuristics(self.start_method, **kwargs)

    def execute_algorithm(self, data: list):
        best_solution = self.get_perfect_split(data)
        _of, best_state = self.object_SH.execute_algorithm(data)
        of_best_state = self.objective_function(best_state, best_solution)
        cur_state = best_state
        t_cur = self.t_max
        it = 1
        while t_cur >= self.t_min and it <= self.max_iter:
            temp_state = SmallChanges(cur_state).swap(method=self.method_changes)
            of_temp_state = self.objective_function(temp_state, best_solution)

            if of_best_state > of_temp_state:
                best_state = cur_state = temp_state
                of_best_state = of_temp_state
            elif np.random.random() <= np.exp(
                                            max([
                                                -200,
                                                (self.objective_function(cur_state, best_solution) - of_temp_state)
                                                / t_cur
                                                ])
                                            ):
                cur_state = temp_state

            t_cur = self.temperature_change(self.t_max, it)
            it += 1

        _of = self.objective_function(best_state, best_solution)

        return _of, best_state

    # ###############################################################################################################
    @staticmethod
    def temperature_change(temperature: float, k: int, method: str = 'Koshy') -> float:
        if method == 'Boltzmann':
            return temperature / np.log(1 + k)
        return temperature / k
