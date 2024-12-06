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
import sys
import pickle

def leapfrog_tf(dydt, tspan, y0, n, dim):
  # dt: step size
  # n: number of steps
  # should be leapfrog integration scheme mentioned in Eq.7/8 of the paper
  # aold, anew: d[q,p]/dt
  t0 = tspan[0]
  tstop = tspan[1]
  dt = (tstop - t0) / n

  t = [0.0 for _ in range(n+1)]
  y = [tf.zeros(dim) for _ in range(n+1)]

  for i in range(0, n + 1):
    if i == 0:
      t[0] = t0
      y[0] = y0
      anew = dydt(t, y[i])
    else:
      t[i] = t[i-1] + dt
      aold = anew
      y[i] = tf.tensor_scatter_nd_update(y[i], [[j] for j in range(int(dim/2))], y[i-1][:int(dim/2)] + dt * (y[i-1][int(dim/2):] + 0.5 * dt * aold[int(dim/2):]))
      anew = dydt(t, y[i])
      y[i] = tf.tensor_scatter_nd_update(y[i], [[j + int(dim/2)] for j in range(int(dim/2))], y[i-1][int(dim/2):] + 0.5 * dt * (aold[int(dim/2):] + anew[int(dim/2):]))
  y = tf.stack(y, axis=-1)
  return y 

def lfrog(fun, y0, t, dt, *args, **kwargs):
  k1 = fun(y0, t-dt, *args, **kwargs)
  k2 = fun(y0, t+dt, *args, **kwargs)
  dy = (k2-k1) / (2*dt)
  return dy

def L2_loss(u, v):
  return tf.reduce_mean(tf.square(u - v))

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

def choose_nonlinearity(name):
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

class Transcript(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def log_start(filename):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Transcript(filename)

def log_stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal