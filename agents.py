import random


class RandomAgent(object):
    def decide_action(self, observation):
        return random.choice([0, 1])
