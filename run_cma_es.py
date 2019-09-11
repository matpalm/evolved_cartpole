#!/usr/bin/env python3

import argparse
import cma
import agents
import cartpole_fitness
import numpy as np
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--popsize', type=int, default=20,
                    help='cma population size')
parser.add_argument('--fitness-log-file', type=str, default='/dev/null',
                    help="where to write epoch-fitness info")
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

fitness_log_file = open(opts.fitness_log_file, "w")
print("epoch\tfitness", file=fitness_log_file)

epoch = 0
while not es.stop():
    # fetch next set of trials
    trial_weights = es.ask()
    # print("trial_weights", trial_weights)

    # run eval
    fitnesses = []
    for member_idx, weights in enumerate(trial_weights):
        agent.set_weights_of_model(weights)
        fitness = cartpole.fitness(agent)
        fitnesses.append(-fitness)

    # update es
    es.tell(trial_weights, fitnesses)

    # save best weights
    if opts.weights_dir is not None:
        np.save("%s/%05d.npy" % (opts.weights_dir, epoch), es.result[0])

    # give results
    epoch += 1
    for f in fitnesses:
        print("%d\t%f" % (epoch, f), file=fitness_log_file)
    fitness_log_file.flush()
    print("\t".join(map(str, ["F", epoch, fitnesses, np.min(fitnesses),
                              np.mean(fitnesses), np.max(fitnesses)])))
    es.result_pretty()
    sys.stdout.flush()

es.disp()
fitness_log_file.close()
