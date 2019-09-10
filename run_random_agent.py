#!/usr/bin/env python3

import agents
import cartpole_fitness

agent = agents.RandomAgent()
evaluator = cartpole_fitness.CartPoleFitness(render=True)

print(evaluator.fitness(agent))
