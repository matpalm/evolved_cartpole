import numpy as np
import random


class SimpleGA(object):

    def __init__(self, popn_size,
                 new_member_fn,
                 fitness_fn,
                 cross_over_fn,
                 proportion_new_members=0,
                 proportion_elite=0):

        self.popn_size = popn_size

        if proportion_new_members < 0 or proportion_new_members > 1:
            raise Exception("expect proportion_new_members to be (0, 1)")
        self.num_new_members = int(self.popn_size * proportion_new_members)

        self.fitness_fn = fitness_fn

        if proportion_elite < 0 or proportion_elite > 1:
            raise Exception("expect proportion_elite to be (0, 1)")
        self.num_elite = int(self.popn_size * proportion_elite)

        self.new_member_fn = new_member_fn
        self.cross_over_fn = cross_over_fn

        self.members = [new_member_fn() for _ in range(popn_size)]
        self.selection_array = None

    def get_members(self):
        return self.members

    def get_elite_member(self):
        if self.selection_array is None:
            raise Exception(
                "no selection_array; need to call calc_fitnesses?")
        return self.members[np.argmax(self.selection_array)]

    def calc_fitnesses(self):
        # keep raw fitness values as member for debug only
        self.raw_fitness_values = [self.fitness_fn(m) for m in self.members]
        self.selection_array = np.zeros_like(self.raw_fitness_values)
        normaliser = (self.popn_size * (self.popn_size+1)) / 2
        for rank, idx in enumerate(np.argsort(self.raw_fitness_values)):
            self.selection_array[idx] = (rank+1) / normaliser
        assert np.isclose(np.sum(self.selection_array), 1.0)

    def breed_next_gen(self):
        if self.selection_array is None:
            # TODO: just make breed explicit after set_raw_fitness ?
            raise Exception(
                "need to call calc_fitnesses() before each breed_next_gen() call")

        # prep next generation
        next_gen_members = []

        # fill some number of random new members
        for _ in range(self.num_new_members):
            next_gen_members.append(self.new_member_fn())

        # keep some number of elite members from last population
        if self.num_elite > 0:
            elite_idxs = np.argsort(self.selection_array)[-self.num_elite:]
            for i in elite_idxs:
                next_gen_members.append(self.members[i])

        # fill rest with cross over generated members
        while len(next_gen_members) < self.popn_size:
            p1_idx = self._select_member_idx()
            p2_idx = self._select_member_idx()
            child1, child2 = self.cross_over_fn(self.members[p1_idx],
                                                self.members[p2_idx])
            next_gen_members.append(child1)
            if len(next_gen_members) < self.popn_size:
                next_gen_members.append(child2)

        # stack into single array and invalidate old selection array
        self.members = next_gen_members
        self.selection_array = None

    def _select_member_idx(self):
        return np.random.choice(range(self.popn_size),
                                p=self.selection_array)

    def _crossover(self, p1_idx, p2_idx):
        p1 = self.members[p1_idx]
        p2 = self.members[p2_idx]
        crossover_idx = random.randint(0, self.member_size-1)
        c1 = np.concatenate([p1[:crossover_idx], p2[crossover_idx:]])
        c2 = np.concatenate([p2[:crossover_idx], p1[crossover_idx:]])
        return c1, c2
