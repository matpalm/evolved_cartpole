#!/usr/bin/env python3

import random
import numpy as np
import simple_ga


def np_new_member():
    member_size = random.randint(10, 30)
    return np.random.normal(size=(member_size,))


def np_crossover(p1, p2):
    crossover_idx = random.randint(0, min(len(p1), len(p2)))
    c1 = np.concatenate([p1[:crossover_idx], p2[crossover_idx:]])
    c2 = np.concatenate([p2[:crossover_idx], p1[crossover_idx:]])
    return c1, c2


def fitness(m):
    return np.sum(m)


ga = simple_ga.SimpleGA(popn_size=10,
                        new_member_fn=np_new_member,
                        fitness_fn=fitness,
                        cross_over_fn=np_crossover,
                        proportion_new_members=0.2,
                        proportion_elite=0.1)

for _ in range(100):
    ga.calc_fitnesses()
    elite = ga.get_elite_member()
    print("ELITE", len(elite), np.sum(elite))
    ga.breed_next_gen()
