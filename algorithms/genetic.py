import numpy as np
import random

from collections import defaultdict
from itertools import chain
from math import ceil
from core import MNPAlgorithm
from algorithms.simple_heuristics import SimpleHeuristics, SmallChanges

from typing import List

Population = List[object]
Individual = List[list]


class Genetic(MNPAlgorithm):

    def __init__(self, size_population: int = 10, number_of_epoch: int = 100, selection: float = 0.2,
                 start_method: str = 'greedy', method_changes: str = 'random', max_num_split: int = 1,
                 num_population: int = 1, num_migration: int = 1,
                 method_choose_parent: str = 'random', method_choose_next_population: str = 'best',
                 method_migration: str = 'best', frequency_migration: int = 5, **kwargs):

        super(Genetic, self).__init__(**kwargs)
        self.start_method = start_method
        self.method_changes = method_changes
        self.size_population = size_population
        self.number_of_epoch = number_of_epoch
        self.max_num_split = max_num_split
        self.object_SH = SimpleHeuristics(self.start_method, **kwargs)
        self.selection = selection
        self.num_population = num_population
        self.num_migration = num_migration
        self.method_migration = method_migration
        self.frequency_migration = frequency_migration
        self.method_choose_parent = method_choose_parent
        self.method_choose_next_population = method_choose_next_population

    def execute_algorithm(self, data: list):
        best_split, populations, _ofs, _of_best, best_individual = self.preparatory_steps(data)
        data = np.array(data)

        for _epoch in range(self.number_of_epoch):

            children = np.array([
                self.get_new_chromosome(populations[i], _ofs[i], data, best_split)
                for i in range(self.num_population)
            ])

            populations, _ofs, _of_best, best_individual = self.create_new_population(
                children, data, best_split, _of_best, best_individual, _epoch
            )

        return _of_best, self.decode_individual(data, best_individual)

    # ###############################################################################################################
    '''Preparatory steps: create population, get _of, find best'''

    def preparatory_steps(self, data: list):
        best_split = self.get_perfect_split(data)

        data = np.array(data)
        populations = [self.first_population(data) for _ in range(self.num_population)]

        _of = np.array([
            list(map(
                lambda individual: self.objective_function(self.decode_individual(data, individual), best_split),
                pop_i
            ))
            for pop_i in populations
        ])

        _of_best = np.min(_of)
        i, j = np.argwhere(_of == np.min(_of))[0]
        best_individual = populations[i][j]
        return best_split, populations, _of, _of_best, best_individual

    # ###############################################################################################################
    '''Create first population'''

    def first_population(self, data: np.array):
        _of, individual = self.object_SH.execute_algorithm(data)
        population = [SmallChanges(individual).swap(method=self.method_changes) for _ in range(self.size_population)]

        return np.array(list(map(
            lambda individual: self.encode_individual(data, individual),
            population
        )))

    # ###############################################################################################################
    '''Encode/Decode chromosome'''

    def encode_individual(self, data: list, individual: Individual):
        n = self.number_of_sets
        m = len(data)

        individual_enc = np.zeros((n, m))
        for i in range(n):
            for el in individual[i]:
                for j in self.indexes(data, el):
                    if 1 not in individual_enc[:i, j] and 1 != individual_enc[i, j]:
                        individual_enc[i][j] = 1
                        break

        return individual_enc.tolist()

    def decode_chromosome(self, data: np.array, individual_enc: Individual):
        return list(data[self.indexes(individual_enc, 1)])

    def decode_individual(self, data: np.array, individual_enc: Individual):
        return [self.decode_chromosome(data, chromosome_enc) for chromosome_enc in individual_enc]

    # ###############################################################################################################
    '''Сhange with one population'''

    def get_new_chromosome(self, population, _of, data, best_split):
        children_cros = list(chain.from_iterable(
            self.crossover(population, _of, data, best_split, self.max_num_split)
            for _ in range(self.size_population)
        ))

        children_mut = [
            self.mutation(data, random.choice(population), random.choice(opt_mut))
            for _ in range(self.size_population)
            for opt_mut in ['max-min', 'max-random', 'random']
        ]

        return children_cros + children_mut
    # ###############################################################################################################
    '''Crossover'''

    def crossover(self, population: Population, _of: np.array, data: np.array, best_split: float, max_num_split: int):
        return [self.crossover_best_subset(population, _of, data, best_split)] + \
               [self.crossover_split(population, _of, data, i + 1) for i in range(max_num_split)]

    def crossover_split(self, population: Population, _of: np.array, data: np.array, number_split: int):
        parent1, parent2 = random.choices(population, weights=(1 - _of / sum(_of)), k=2)

        splits = [0] + sorted(random.sample(range(1, len(data)), k=number_split)) + [len(data)]
        queue = [0 if random.random() <= 0.5 else 1 for _ in range(number_split + 1)]
        parents = [parent1, parent2]
        return [
            list(chain.from_iterable(
                parents[queue[j]][i][splits[j]: splits[j + 1]]
                for j in range(number_split + 1)
            ))
            for i in range(self.number_of_sets)
        ]

    def crossover_best_subset(self, population: Population, _of: np.array, data: np.array, best_split: float):

        parent1, parent2 = self.choose_chromosome(data, population, _of, 2, best_split, self.method_choose_parent)

        parent1, parent2 = list(map(lambda parent:
                                    self.sort_individual(data, parent, best_split),
                                    [parent1, parent2]
                                    ))
        child = []
        counter1, counter2 = 0, 0
        for _ in range(self.number_of_sets):
            if random.random() <= 0.5:
                child.append(parent1[counter1])
                counter1 += 1
            else:
                child.append(parent2[counter2])
                counter2 += 1

        return self.restore_individual(data, child)

    # ###############################################################################################################
    '''Mutation'''

    def mutation(self, data: np.array, individual, opt_mut='min-max'):
        return self.encode_individual(
            data,
            SmallChanges(
                self.decode_individual(data, individual)
            ).swap(method=opt_mut)
        )

    # ###############################################################################################################
    '''Create new population'''

    def create_new_population(self, children, data, best_split, _of_best, best_individual, epoch):

        if self.num_population > 1 and epoch % self.frequency_migration == 0:
            if self.method_migration == 'best':
                all_children = list(chain.from_iterable([
                    sorted(child, key=lambda individual: self.objective_function(
                        self.decode_individual(data, individual), best_split
                    ))[:5]
                    for child in children
                ]))
            else:
                all_children = list(chain.from_iterable(children))

            for _ in range(self.num_migration):
                children = [
                    np.append(children[i], [random.choice(all_children)], axis=0)
                    for i in range(self.num_population)
                ]

        _ofs = np.array([
            list(map(
                lambda individual: self.objective_function(self.decode_individual(data, individual), best_split),
                children_one_pop
            ))
            for children_one_pop in children
        ])

        childrens = []
        _of = []

        for i in range(self.num_population):
            child = self.choose_chromosome(
                data, children[i], _ofs[i], self.size_population, best_split, self.method_choose_next_population
            )
            childrens.append(child)
            _of.append(
                list(map(
                    lambda individual: self.objective_function(self.decode_individual(data, individual), best_split),
                    child
                ))
            )

        if np.min(_of) < _of_best:
            i, j = np.argwhere(_of == np.min(_of))[0]
            _of_best = _of[i][j]
            best_individual = children[i][j]

        return childrens, _of, _of_best, best_individual
    # ###############################################################################################################
    '''Subprograms'''

    @staticmethod
    def indexes(lst, value):
        return [i for i, v in enumerate(lst) if v == value]

    def choose_chromosome(self, data, population, _of, num, best_split, method_choose_chromosome):
        if method_choose_chromosome == 'tournament':
            return [
                min(random.choices(population, k=2), key=lambda individual: self.objective_function(
                    self.decode_individual(data, individual), best_split))
                for _ in range(num)
            ]

        elif method_choose_chromosome == 'roulette':
            return random.choices(
                sorted(population, key=lambda individual: self.objective_function(
                    self.decode_individual(data, individual), best_split
                ))[:num * 5], k=num)

        elif method_choose_chromosome == 'best':
            return sorted(population, key=lambda individual: self.objective_function(
                    self.decode_individual(data, individual), best_split
            ))[:num]

        else:
            return random.choices(population, weights=(1 - _of / sum(_of)), k=num)

    def sort_individual(self, data: np.array, individual: list, best_split: float):
        return sorted(individual, key=lambda x: abs(sum(self.decode_chromosome(data, x)) - best_split))

    def restore_individual(self, data: np.array, individual: list):
        individual_dec = self.decode_individual(data, individual)
        dct_el = defaultdict(int)
        for el in data:
            dct_el[el] += 1

        for i in range(self.number_of_sets):
            for el in reversed(individual_dec[i]):
                if dct_el[el] > 0:
                    dct_el[el] -= 1
                else:
                    individual_dec[i].remove(el)

        for k, v in sorted(dct_el.items(), reverse=True):
            while v != 0:
                individual_dec[np.argmin(list(map(sum, individual_dec)))].append(k)
                v -= 1

        return self.encode_individual(data, individual_dec)
