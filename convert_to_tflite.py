#!/usr/bin/env python3

import tensorflow as tf
import argparse
import agents
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input-weights', type=str, default=None,
                    help="input numpy weights files. used to init neural agent")
parser.add_argument('--output-tflite-file', type=str, default=None,
                    help="output for tf lite weights files. to be used for neural_lite agent")
opts = parser.parse_args()

agent = agents.NeuralAgent()
agent.set_weights_of_model(np.load(opts.input_weights))

converter = tf.lite.TFLiteConverter.from_keras_model(agent.model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
lite_model = converter.convert()
open(opts.output_tflite_file, "wb").write(lite_model)
