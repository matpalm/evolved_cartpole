#!/usr/bin/env python3

import agents
import cartpole_fitness
import numpy as np
import argparse
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env-render', action='store_true',
                    help='if set runs gym env render')
parser.add_argument('--agent', type=str, default='random',
                    help="agent type ['random', 'neural', 'neural_lite']")
parser.add_argument('--weights', type=str, default=None,
                    help="weights files, only valid for --agent=neural")
parser.add_argument('--lite-weights', type=str, default=None,
                    help="tf lite weights files, must be set for --agent=neural_lite")
parser.add_argument('--trials', type=int, default=10,
                    help='num trials to run; new agent per trial')
opts = parser.parse_args()

evaluator = cartpole_fitness.CartPoleFitness(render=opts.env_render)

print("trial\ttotal_reward")
for trial_idx in range(opts.trials):
    if opts.agent == 'random':
        agent = agents.RandomAgent()
    elif opts.agent == 'neural':
        agent = agents.NeuralAgent()
        if opts.weights is not None:
            agent.set_weights_of_model(np.load(opts.weights))
    elif opts.agent == 'neural_lite':
        agent = agents.NeuralLiteAgent(opts.lite_weights)
    else:
        raise Exception("unexpected agent type [%s]" % opts.agent)
    print("%d\t%d" % (trial_idx, evaluator.fitness(agent)))
    sys.stdout.flush()
