import gym
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import util as u


class SumEvaluator(object):

    def num_weights(self):
        return 10

    def fitness(self, flattened_weights):
        return np.sum(flattened_weights)


class CartPoleEvaluator(object):

    def __init__(self):
        # build simple model
        inp = Input(shape=(4,))
        hidden1 = Dense(8, activation='relu')(inp)
        hidden2 = Dense(8, activation='relu')(hidden1)
        output = Dense(1, activation='sigmoid')(hidden2)
        self.model = Model(inp, output)
        # prep cart pole env
        self.env = gym.make('CartPole-v0')

    def num_weights(self):
        return u.total_weights_of_shapes(u.weight_shapes_of(self.model))

    def fitness(self, flattened_weights):
        # set trial weights in model
        u.set_weights_of_model(self.model, flattened_weights)
        # run trials
        total_reward = 0
        for _ in range(10):  # num_trials ?
            observation = self.env.reset()
            done = False
            while not done:
                # run observation through model
                output = self.model.predict(np.expand_dims(observation, 0))
                action_prob = output[0][0]
                action = 1 if action_prob > 0.5 else 0
                # step simulation forward
                observation, reward, done, _info = self.env.step(action)
                total_reward += reward
                # env.render()
        return total_reward


if __name__ == '__main__':
    e = SumEvaluator()
    weights = np.random.normal(size=(e.num_weights(),))
    print(weights)
    print(e.fitness(weights))
