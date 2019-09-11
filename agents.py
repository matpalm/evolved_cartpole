import random
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import util as u


class RandomAgent(object):

    def decide_action(self, observation):
        return random.choice([0, 1])


class NeuralAgent(object):

    def __init__(self):
        # build simple model
        inp = Input(shape=(4,))
        hidden1 = Dense(8, activation='relu')(inp)
        hidden2 = Dense(8, activation='relu')(hidden1)
        output = Dense(1, activation='sigmoid')(hidden2)
        self.model = Model(inp, output)

    def get_flattened_weights_of_model(self):
        return np.concatenate([w.flatten() for w in self.model.get_weights()])

    def set_weights_of_model(self, flattened_weights):
        shapes = u.weight_shapes_of(self.model)
        if flattened_weights.shape != (u.total_weights_of_shapes(shapes),):
            raise Exception("expected weights shaped (%d,) not %s" % (
                total_weights_of_shapes(shapes), flattened_weights.shape))
        idx = 0
        weights_to_set = []
        for s in shapes:
            offset = u.num_weights(s)
            weight_slice = flattened_weights[idx: idx+offset]
            weights_to_set.append(
                np.array(weight_slice, dtype=np.float32).reshape(s))
            idx += offset
        self.model.set_weights(weights_to_set)

    def decide_action(self, observation):
        output = self.model.predict(np.expand_dims(observation, 0))
        action_prob = output[0][0]
        action = 1 if action_prob > 0.5 else 0
        return action
