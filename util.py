import numpy as np
import random


def num_weights(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def weight_shapes_of(model):
    return [w.shape for w in model.get_weights()]


def total_weights_of_shapes(shapes):
    return sum([num_weights(s) for s in shapes])


def numpy_array_crossover(p1, p2):
    assert p1.shape == p2.shape
    crossover_idx = random.randint(0, len(p1))
    c1 = np.concatenate([p1[:crossover_idx], p2[crossover_idx:]])
    c2 = np.concatenate([p2[:crossover_idx], p1[crossover_idx:]])
    return c1, c2
