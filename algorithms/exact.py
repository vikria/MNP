import gurobipy as grb
from core import MNPAlgorithm
from typing import List


class ExactAlgorithm(MNPAlgorithm):
    """
    Uses Gurobi optimizer to solve MNP
    """

    def execute_algorithm(self, data: List):
        n = len(data)
        m = self.number_of_sets
        _best = self.get_perfect_split(data)

        model = grb.Model()

        x_vars = model.addVars(
            (
                (i, j)
                for i in range(n)
                for j in range(m)
            ),
            vtype=grb.GRB.BINARY
        )

        model.setObjective(
            sum(
                (sum(x_vars[i, j] * elem for i, elem in enumerate(data)) - _best) ** 2
                for j in range(m)
            ),
            grb.GRB.MINIMIZE
        )

        model.addConstrs(
            x_vars.sum(i, '*') == 1
            for i in range(n)
        )

        if self.settings.get('TimeLimit'):
            model.setParam('TimeLimit', self.settings.get('TimeLimit'))

        model.optimize()

        sets = [
            [elem for i, elem in enumerate(data) if x_vars[i, j].x == 1]
            for j in range(m)
        ]

        return model.objVal, sets
