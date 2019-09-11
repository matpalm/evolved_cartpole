#!/usr/bin/env python3

import cma
import agents
import cartpole_fitness
import numpy as np
import sys

agent = agents.NeuralAgent()
eg_weights = agent.get_flattened_weights_of_model()
num_weights = len(eg_weights)
print(num_weights)

cartpole = cartpole_fitness.CartPoleFitness()

es = cma.CMAEvolutionStrategy([0] * num_weights,  # x0
                              1.0,                # sigma0
                              {'popsize': 20})

fitnesses_log = open("cms_es_fitnesses.tsv", "w")
print("epoch\tfitness", file=fitnesses_log)

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
        # print("F", epoch, member_idx, fitness)
        fitnesses.append(-fitness)

    # update es
    es.tell(trial_weights, fitnesses)

    # save best weights
    np.save("solutions/%05d.npy" % epoch,
            es.result[0])

    # give results
    epoch += 1
    for f in fitnesses:
        print("\t".join(map(str, [epoch, f])), file=fitnesses_log)
    fitnesses_log.flush()
    print("\t".join(map(str, ["F", epoch, fitnesses, np.min(fitnesses),
                              np.mean(fitnesses), np.max(fitnesses)])))
    es.result_pretty()
    sys.stdout.flush()

es.disp()
fitnesses_log.close()
