#!/usr/bin/env python3

import random
import numpy as np
import simple_ga
import util


def np_new_member():
    return np.random.normal(size=(20,))


def sum_fitness(member):
    return np.sum(member)


ga = simple_ga.SimpleGA(popn_size=10,
                        new_member_fn=np_new_member,
                        fitness_fn=sum_fitness,
                        cross_over_fn=util.numpy_array_crossover,
                        proportion_new_members=0.2,
                        proportion_elite=0.1)

for _ in range(100):
    ga.calc_fitnesses()
    elite = ga.get_elite_member()
    print("ELITE", len(elite), np.sum(elite))
    ga.breed_next_gen()
