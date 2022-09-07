from abc import ABCMeta, abstractmethod


class MNPAlgorithm(metaclass=ABCMeta):
    """ Base class for Multiway Number Partitioning."""

    def __init__(self, number_of_sets: int = 1, **kwargs):
        super(MNPAlgorithm, self).__init__()
        self.number_of_sets = number_of_sets
        self.settings = dict(**kwargs)

    def get_perfect_split(self, data: list):
        return sum(data) / self.number_of_sets

    @staticmethod
    def objective_function(state=None, perfect=None):
        return sum((sum(s) - perfect) ** 2 for s in state)

    def run(self, data: list):
        _of, _sets = self.execute_algorithm(data)
        return _of, _sets

    @abstractmethod
    def execute_algorithm(self, data: list):
        raise NotImplementedError()
