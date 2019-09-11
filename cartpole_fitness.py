# given an agent, eval fitness against cartpole

import gym


class CartPoleFitness(object):

    def __init__(self, render=False):
        # prep cart pole env
        self.env = gym.make('CartPole-v0')
        self.render = render

    def fitness(self, agent):
        total_reward = 0
        for _ in range(10):  # num_trials ?
            observation = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.decide_action(observation)
                observation, reward, done, _info = self.env.step(action)
                episode_reward += reward
                if self.render:
                    self.env.render()
            total_reward += episode_reward
        return total_reward
