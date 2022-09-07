from abc import ABC, abstractmethod
from .data_source import DataSource
from .algorithm import MNPAlgorithm


class Solver(ABC):
    """ Base solver class. Puts together DataSource and Algorithm objects. """

    def __init__(self, source: DataSource = None, algorithm: MNPAlgorithm = None):
        self.source = source
        self.algorithm = algorithm

    def set_params(self, source: DataSource = None, algorithm: MNPAlgorithm = None):
        self.source = source or self.source
        self.algorithm = algorithm or self.algorithm

    @abstractmethod
    def solve_problem(self):
        raise NotImplementedError()
