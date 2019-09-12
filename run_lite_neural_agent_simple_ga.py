#!/usr/bin/env python3

import argparse
import simple_ga
import agents
import cartpole_fitness
import util as u
import numpy as np
import convert_to_tflite
import io
import random

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

# in the neural_lite_agent the representation is the bytes of
# tflite file


def new_member_bytes():
    random_agent = agents.NeuralAgent()
    flat_buffer_bytes = convert_to_tflite.convert_to_file_bytes(random_agent)
    return flat_buffer_bytes


def calc_cartpole_fitness(flat_buffer_bytes):
    agent = agents.NeuralLiteAgent(tflite_bytes=flat_buffer_bytes)
    return cartpole.fitness(agent)


def crossover_member_bytes(p1, p2):
    assert len(p1) == len(p2)
    crossover_idx = random.randint(0, len(p1))
    c1 = p1[:crossover_idx] + p2[crossover_idx:]
    c2 = p2[:crossover_idx] + p1[crossover_idx:]
    return c1, c2


ga = simple_ga.SimpleGA(popn_size=opts.popsize,
                        new_member_fn=new_member_bytes,
                        fitness_fn=calc_cartpole_fitness,
                        cross_over_fn=crossover_member_bytes,
                        proportion_new_members=0.1,
                        proportion_elite=0.1)

fitness_log = u.Log(opts.fitness_log_file)

for generation in range(100):

    ga.calc_fitnesses()

    fitness_log.log(generation, np.max(ga.raw_fitness_values))

    if opts.weights_dir is not None:
        u.ensure_dir_exists(opts.weights_dir)
        elite = ga.get_elite_member()
        filepath = "%s/%03d.tflite" % (opts.weights_dir, generation)
        open(filepath, "wb").write(elite)

    ga.breed_next_gen()
