#!/usr/bin/env python3

import argparse
import simple_ga
import agents
import cartpole_fitness
import util as u
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--popsize', type=int, default=20,
                    help='cma population size')
parser.add_argument('--fitness-log-file', type=str, default='/dev/null',
                    help="where to write generation-fitness info")
parser.add_argument('--weights-dir', type=str, default=None,
                    help="if set, save per generation best weights numpy file here")
opts = parser.parse_args()


cartpole = cartpole_fitness.CartPoleFitness()


def new_agent():
    return agents.NeuralAgent()


def calc_cartpole_fitness(member):
    return cartpole.fitness(member)


def crossover_agents(p1, p2):
    p1_weights = p1.get_flattened_weights_of_model()
    p2_weights = p2.get_flattened_weights_of_model()

    c1_weights, c2_weights = u.numpy_array_crossover(p1_weights, p2_weights)

    c1 = agents.NeuralAgent()
    c1.set_weights_of_model(c1_weights)
    c2 = agents.NeuralAgent()
    c2.set_weights_of_model(c2_weights)

    return c1, c2


ga = simple_ga.SimpleGA(popn_size=opts.popsize,
                        new_member_fn=new_agent,
                        fitness_fn=calc_cartpole_fitness,
                        cross_over_fn=crossover_agents,
                        proportion_new_members=0.1,
                        proportion_elite=0.1)


fitness_log = u.Log(opts.fitness_log_file)

for generation in range(100):

    ga.calc_fitnesses()

    fitness_log.log(generation, np.max(ga.raw_fitness_values))

    if opts.weights_dir is not None:
        u.ensure_dir_exists(opts.weights_dir)
        elite = ga.get_elite_member()
        np.save("%s/%03d" % (opts.weights_dir, generation),
                elite.get_flattened_weights_of_model())

    ga.breed_next_gen()
