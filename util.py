import numpy as np


def num_weights(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def weight_shapes_of(model):
    return [w.shape for w in model.get_weights()]


def total_weights_of_shapes(shapes):
    return sum([num_weights(s) for s in shapes])


def get_flattened_weights_of_model(model):
    return np.concatenate([w.flatten() for w in model.get_weights()])


def set_weights_of_model(model, flattened_weights):
    shapes = weight_shapes_of(model)
    if flattened_weights.shape != (total_weights_of_shapes(shapes),):
        raise Exception("expected weights shaped (%d,) not %s" % (
            total_weights_of_shapes(shapes), flattened_weights.shape))
    idx = 0
    weights_to_set = []
    for s in shapes:
        weight_slice = flattened_weights[idx: idx+num_weights(s)]
        weights_to_set.append(
            np.array(weight_slice, dtype=np.float32).reshape(s))
        idx += num_weights(s)
    model.set_weights(weights_to_set)
