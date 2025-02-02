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

def choose_nonlinearity(name):
  '''
  Description:
    choose the nonlinearity function given the name
  '''
  nl = None
  if name == 'tanh':
    nl = keras.activations.tanh
  elif name == 'relu':
    nl = keras.activations.relu
  elif name == 'sigmoid':
    nl = keras.activations.sigmoid
  elif name == 'softplus':
    nl = keras.activations.softplus
  elif name == 'selu':
    nl = keras.activations.selu
  elif name == 'elu':
    nl = keras.activations.elu
  elif name == 'swish':
    nl = keras.activations.swish
  elif name == 'sine':
    nl = tf.sin
  else:
    raise ValueError("nonlinearity not recognized")
  return nl

@keras.saving.register_keras_serializable()
class MLP(keras.Model):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine', num_layers=5):
    super(MLP, self).__init__()
    self.layers_list = []
    self.layers_list.append(layers.Dense(hidden_dim, input_shape=(input_dim,), activation=None,
                                    kernel_initializer=initializers.Orthogonal()))
    self.layers_list.append(layers.BatchNormalization())
    for _ in range(num_layers-2):
      self.layers_list.append(layers.Dense(hidden_dim, input_shape=(hidden_dim,), activation=None,
                                kernel_initializer=initializers.Orthogonal()))
      self.layers_list.append(layers.BatchNormalization())
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

@keras.saving.register_keras_serializable()
class CNN_MLP(keras.Model):
  '''
  Description:
    ** convolution-based architecture **
    5 convolution layers + 1 fully connected layers, zero padding is added after convolution whenever the shape is unmatched;
    hyperparameter settings are:
    - number of filters: [10, 10, 10, 10, 1]
    - kernel size: [10, 10, 7, 10, 10]
    - stride size: [4, 4, 4, 4, 4]
  '''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine', num_layers=5):
    super(CNN_MLP, self).__init__()
    self.conv_layers_list = []
    # a series of convolutional layers
    self.conv_layers_list.append(layers.Conv1D(filters=10, kernel_size=10, strides=4, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.Conv1D(filters=10, kernel_size=10, strides=4, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.Conv1D(filters=10, kernel_size=7, strides=4, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.ZeroPadding1D(padding=1))
    self.conv_layers_list.append(layers.Conv1D(filters=10, kernel_size=10, strides=4, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.Conv1D(filters=1, kernel_size=10, strides=4, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())

    # fully connected 
    self.fc_layers_list = []
    self.fc_layers_list.append(layers.Dense(hidden_dim, input_shape=(122,), activation=None,
                                    kernel_initializer=initializers.Orthogonal()))
    self.fc_layers_list.append(layers.BatchNormalization())
    self.fc_layers_list.append(layers.Dense(output_dim, input_shape=(hidden_dim,), activation=None, 
                                use_bias=False, kernel_initializer=initializers.Orthogonal()))
    
    self.nonlinearity = choose_nonlinearity(nonlinearity)
    
  def call(self, x):
    for i, layer in enumerate(self.conv_layers_list):
      if i == 0:
        h = layer(tf.expand_dims(x, axis=-1))
      else:
        h = layer(h)
    h = tf.squeeze(h, axis=-1)
    for i, layer in enumerate(self.fc_layers_list):
      if i != len(self.fc_layers_list) - 1:
        h = self.nonlinearity(layer(h))
      else:
        h = layer(h)
    return h


@keras.saving.register_keras_serializable()
class Info_MLP(keras.Model):
  '''
  Description:
    ** MLP-based architecture that focuses on learning gradients of target density **
    the last entry of output is directly set as 0.5*(u^Tu + rho^Trho + p^Tp)
    and only theta and u are input to the MLP layers
  '''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine', num_layers=5):
    super(Info_MLP, self).__init__()
    
    self.target_dim = 13
    self.aux_dim = 500 * 128
    
    self.layers_list = []
    self.layers_list.append(layers.Dense(hidden_dim, input_shape=(self.target_dim + self.aux_dim,),
                                         activation=None, kernel_initializer=initializers.Orthogonal()))
    self.layers_list.append(layers.BatchNormalization())
    for _ in range(num_layers-2):
      self.layers_list.append(layers.Dense(hidden_dim, input_shape=(hidden_dim,), activation=None,
                                kernel_initializer=initializers.Orthogonal()))
      self.layers_list.append(layers.BatchNormalization())
    self.layers_list.append(layers.Dense(output_dim-1, input_shape=(hidden_dim,), activation=None, 
                                use_bias=False, kernel_initializer=initializers.Orthogonal()))

    self.nonlinearity = choose_nonlinearity(nonlinearity)
    

  def call(self, x):
    theta, rho, u, p = tf.split(x, num_or_size_splits=[self.target_dim, self.target_dim, self.aux_dim, self.aux_dim], axis=-1)
    known_component = 0.5 * tf.reduce_sum(tf.multiply(rho, rho), 1, keepdims=True) + \
                      0.5 * tf.reduce_sum(tf.multiply(u, u), 1, keepdims=True) + \
                      0.5 * tf.reduce_sum(tf.multiply(p, p), 1, keepdims=True)
    h = tf.concat([theta, u], axis=-1)
    for i, layer in enumerate(self.layers_list):
      if i == len(self.layers_list)-1:
        h = layer(h)
      else:
        h = self.nonlinearity(layer(h))
    h = tf.concat([h, known_component], axis=-1)
    return h
  
@keras.saving.register_keras_serializable()
class Info_CNN_MLP(keras.Model):
  '''
  Description:
    ** convolution-based architecture that focuses on learning gradients of target density **
    the last entry of output is directly set as 0.5*(u^Tu + rho^Trho + p^Tp)
    and only theta and u are input to the convolution layers;
    
    6 convolution layers + 1 FC layer
    hyperparameter settings are:
    - number of filters: [20, 20, 20, 20, 20, 1]
    - kernel size: [9, 9, 10, 9, 10, 9]
    - stride size: [2, 2, 2, 2, 2, 2]
  '''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine', num_layers=5):
    super(Info_CNN_MLP, self).__init__()
    
    self.target_dim = 13
    self.aux_dim = 500 * 128
    
    self.conv_layers_list = []
    # a series of convolutional layers
    self.conv_layers_list.append(layers.Conv1D(filters=20, kernel_size=9, strides=2, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.Conv1D(filters=20, kernel_size=9, strides=2, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.Conv1D(filters=20, kernel_size=10, strides=2, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.Conv1D(filters=20, kernel_size=9, strides=2, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.Conv1D(filters=20, kernel_size=10, strides=2, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())
    self.conv_layers_list.append(layers.Conv1D(filters=1, kernel_size=9, strides=2, activation='sigmoid'))
    self.conv_layers_list.append(layers.BatchNormalization())

    # fully connected 
    self.fc_layers_list = []
    self.fc_layers_list.append(layers.Dense(hidden_dim, input_shape=(993,), activation=None,
                                    kernel_initializer=initializers.Orthogonal()))
    self.fc_layers_list.append(layers.BatchNormalization())
    self.fc_layers_list.append(layers.Dense(output_dim-1, input_shape=(hidden_dim,), activation=None, 
                                use_bias=False, kernel_initializer=initializers.Orthogonal()))
    
    self.nonlinearity = choose_nonlinearity(nonlinearity)
    

  def call(self, x):
    theta, rho, u, p = tf.split(x, num_or_size_splits=[self.target_dim, self.target_dim, self.aux_dim, self.aux_dim], axis=-1)
    known_component = 0.5 * tf.reduce_sum(tf.multiply(rho, rho), 1, keepdims=True) + \
                      0.5 * tf.reduce_sum(tf.multiply(u, u), 1, keepdims=True) + \
                      0.5 * tf.reduce_sum(tf.multiply(p, p), 1, keepdims=True)
    h = tf.concat([theta, u], axis=-1)
    for i, layer in enumerate(self.conv_layers_list):
      if i == 0:
        h = layer(tf.expand_dims(h, axis=-1))
      else:
        h = layer(h)
    h = tf.squeeze(h, axis=-1)
    for i, layer in enumerate(self.fc_layers_list):
      if i != len(self.fc_layers_list) - 1:
        h = self.nonlinearity(layer(h))
      else:
        h = layer(h)
    h = tf.concat([h, known_component], axis=-1)
    return h


if __name__ == '__main__':
  nn_model = Info_CNN_MLP(128026, 100, 26, 'sine')
  nn_model(tf.random.normal(shape=[2, 128026], dtype=tf.float32))
  for x in nn_model.trainable_weights:
    print(x.name, x.shape)
  print(nn_model.summary())
  out = nn_model(tf.random.normal(shape=[2, 128026], dtype=tf.float32))
  print(out.shape)