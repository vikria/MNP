from core.solver import Solver


class MNP(Solver):
    """ General class for solving Multiway Number Partitioning. """

    def __init__(self, **kwargs):
        super(MNP, self).__init__(**kwargs)
        self.of = None
        self.sets = None

    def solve_problem(self):
        self.of, self.sets = self.algorithm.run(self.source.get_data())
        return self.of, self.sets

    def __str__(self):
        if self.of is None:
            _str = 'NOT SOLVED'
        else:
            _str = f'objective function: {self.of}' \
                   + f'\nset sums: {list(map(sum, self.sets))}' \
                   + '\nsets:\n' \
                   + '\n'.join(f'{s}' for s in self.sets) \
                   + '\n'
        return _str

    def __repr__(self):
        return self.__str__()
