# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Training Hamiltonian Neural Networks (HNNs) for Bayesian inference problems
# Original authors of HNNs code: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
# Available at https://github.com/greydanus/hamiltonian-nn under the Apache License 2.0
# Modified by Som Dhulipala at Idaho National Laboratory for Bayesian inference problems
# Modifications include:
# - Generalizing the code to any number of dimensions
# - Introduce latent parameters to HNNs to improve expressivity
# - Reliance on the leap frog integrator for improved dynamics stability
# - Obtain the training from probability distribution space
# - Use a deep HNN arichtecture to improve predictive performance

import tensorflow as tf
import keras
from keras import layers, initializers
import os,sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from utils_tf import choose_nonlinearity
from get_args import get_args
args = get_args()

@keras.saving.register_keras_serializable()
class MLP(keras.Model):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine', num_layers=5):
    super(MLP, self).__init__()
    self.layers_list = []
    self.layers_list.append(layers.Dense(hidden_dim, input_shape=(input_dim,), activation=None,
                                    kernel_initializer=initializers.Orthogonal()))
    for _ in range(num_layers-2):
      self.layers_list.append(layers.Dense(hidden_dim, input_shape=(hidden_dim,), activation=None,
                                kernel_initializer=initializers.Orthogonal()))
    self.layers_list.append(layers.Dense(output_dim, input_shape=(hidden_dim,), activation=None, 
                                use_bias=False, kernel_initializer=initializers.Orthogonal()))

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def call(self, x, separate_fields=False):
    for i, layer in enumerate(self.layers_list):
      if i == 0:
        h = self.nonlinearity(layer(x))
      elif i == len(self.layers_list)-1:
        h = layer(h)
      else:
        h = self.nonlinearity(layer(h))
    return h