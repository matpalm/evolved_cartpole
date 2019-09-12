import numpy as np
import random
import os


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


def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


def ensure_dir_exists_for_file(f):
    ensure_dir_exists(os.path.dirname(f))


class Log(object):

    def __init__(self, fname):
        ensure_dir_exists_for_file(fname)
        self.f = open(fname, "w")
        self.f.write("generation\telite_fitness\n")

    def log(self, generation, fitness):
        self.f.write("%d\t%f\n" % (generation, fitness))
        self.f.flush()
