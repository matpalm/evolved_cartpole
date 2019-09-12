#!/usr/bin/env python3

import argparse
import cma
import agents
import cartpole_fitness
import numpy as np
import sys
import util as u

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

agent = agents.NeuralAgent()
eg_weights = agent.get_flattened_weights_of_model()
num_weights = len(eg_weights)
print(num_weights)

cartpole = cartpole_fitness.CartPoleFitness()

es = cma.CMAEvolutionStrategy([0] * num_weights,  # x0
                              1.0,                # sigma0
                              {'popsize': opts.popsize})

fitness_log = u.Log(opts.fitness_log_file)

generation = 0
while not es.stop():
    # fetch next set of trials
    trial_weights = es.ask()

    # run eval
    fitnesses = []
    for member_idx, weights in enumerate(trial_weights):
        agent.set_weights_of_model(weights)
        fitness = cartpole.fitness(agent)
        fitnesses.append(fitness)

    # update es
    # note: cma es is trying to minimise
    es.tell(trial_weights, -1 * np.array(fitnesses))

    # save best weights
    if opts.weights_dir is not None:
        u.ensure_dir_exists(opts.weights_dir)
        np.save("%s/%05d.npy" % (opts.weights_dir, generation), es.result[0])

    # give results
    generation += 1
    fitness_log.log(generation, np.max(fitnesses))
    es.result_pretty()
    sys.stdout.flush()

es.disp()
