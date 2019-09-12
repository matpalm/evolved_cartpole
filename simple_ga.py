#!/usr/bin/env python3

from collections import Counter
import numpy as np
import random


class SimpleGA(object):

    def __init__(self, popn_size, new_member_fn,
                 proportion_new_members=0,
                 proportion_elite=0):
        self.popn_size = popn_size

        if proportion_new_members < 0 or proportion_new_members > 1:
            raise Exception("expect proportion_new_members to be (0, 1)")
        self.num_new_members = int(self.popn_size * proportion_new_members)

        if proportion_elite < 0 or proportion_elite > 1:
            raise Exception("expect proportion_elite to be (0, 1)")
        self.num_elite = int(self.popn_size * proportion_elite)

        # create first member as a way of determining member size
        self.new_member_fn = new_member_fn
        first_member = new_member_fn()
        assert len(first_member.shape) == 1
        self.member_size = first_member.shape[0]

        self.members = np.empty((popn_size, self.member_size))
        for i in range(popn_size):
            self.members[i] = first_member if i == 0 else new_member_fn()

        self.selection_array = None

    def get_members(self):
        return self.members

    def set_raw_fitness_values(self, raw_fitness_values):
        if len(raw_fitness_values) != self.popn_size:
            raise Exception("%d fitnesses provided != popn_size of %d" % (
                len(raw_fitness_values), self.popn_size))
        self.selection_array = np.zeros_like(raw_fitness_values)
        normaliser = (self.popn_size * (self.popn_size+1)) / 2
        for rank, idx in enumerate(np.argsort(raw_fitness_values)):
            self.selection_array[idx] = (rank+1) / normaliser
        assert np.isclose(np.sum(self.selection_array), 1.0)

    def breed_next_gen(self):
        if self.selection_array is None:
            # TODO: just make breed explicit after set_raw_fitness ?
            raise Exception(
                "need to call set_raw_fitness_values() before each breed_next_gen() call")

        # prep next generation
        next_gen_members = []

        # fill some number of random new members
        for _ in range(self.num_new_members):
            next_gen_members.append(self.new_member_fn())

        # keep some number of elite members from last population
        # print("members", self.members)
        # print("self.selection_array", self.selection_array)
        # print("elite_idxs", np.argsort(self.selection_array)[-self.num_elite:])
        if self.num_elite > 0:
            elite_idxs = np.argsort(self.selection_array)[-self.num_elite:]
            for i in elite_idxs:
                next_gen_members.append(self.members[i])

        # fill rest with cross over generated members
        while len(next_gen_members) < self.popn_size:
            parent1_idx = self._select_member_idx()
            parent2_idx = self._select_member_idx()
            child1, child2 = self._crossover(parent1_idx, parent2_idx)
            next_gen_members.append(child1)
            if len(next_gen_members) < self.popn_size:
                next_gen_members.append(child2)

        # stack into single array and invalidate old selection array
        self.members = np.stack(next_gen_members)
       # print("NEXT GEN MEMBERS", self.members)
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


def new_member():
    return np.random.normal(size=(20,))


ga = SimpleGA(popn_size=10,
              new_member_fn=new_member,
              proportion_new_members=0.2,
              proportion_elite=0.1)

while True:  # for _ in range(10000):
    raw_fitnesses = np.sum(ga.get_members(), axis=1)
    print("raw_fitnesses", np.max(raw_fitnesses))  # , raw_fitnesses)
    ga.set_raw_fitness_values(raw_fitnesses)
    ga.breed_next_gen()

raw_fitnesses = np.sum(ga.members, axis=1)
print("raw_fitnesses", np.max(raw_fitnesses))
