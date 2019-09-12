#!/usr/bin/env python3

import tensorflow as tf
import agents
import numpy as np


def convert_to_file_bytes(agent):
    converter = tf.lite.TFLiteConverter.from_keras_model(agent.model)
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    lite_model_bytes = converter.convert()
    return lite_model_bytes


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-weights', type=str, default=None,
                        help="input numpy weights files. used to init neural agent")
    parser.add_argument('--output-tflite-file', type=str, default=None,
                        help="output for tf lite weights files. to be used for neural_lite agent")
    opts = parser.parse_args()

    agent = agents.NeuralAgent()
    agent.set_weights_of_model(np.load(opts.input_weights))
    lite_model_bytes = convert_to_file_bytes(agent)
    open(opts.output_tflite_file, "wb").write(lite_model_bytes)
