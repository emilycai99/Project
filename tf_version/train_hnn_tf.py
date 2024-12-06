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

import os
import sys
import tensorflow as tf
import keras
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from nn_models_tf import MLP
from hnn_tf import HNN
from data_tf import get_dataset_tf
from utils_tf import L2_loss, log_start, log_stop
from get_args import get_args

def train(args):
  # set random seed
  tf.random.set_seed(args.seed)
  
  output_dim = args.input_dim
  # Model building
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity, 
                 num_layers=args.num_layers)
  model = HNN(args.input_dim, differentiable_model=nn_model,
            grad_type=args.grad_type)
  model(tf.random.normal([args.batch_size, args.input_dim]))
  optim = keras.optimizers.Adam(learning_rate=args.learn_rate, weight_decay=1e-4)

  # define one-step training function
  @tf.function
  def train_step(x, true):
    with tf.GradientTape() as tape:
      dxdt_hat = model.time_derivative(x)
      loss = L2_loss(true, dxdt_hat)
    grads = tape.gradient(loss, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))
    return loss

  # arrange data
  data = get_dataset_tf(seed=args.seed)
  x = tf.Variable(data['coords'], dtype=tf.float32)
  test_x = tf.Variable(data['test_coords'], dtype=tf.float32)
  dxdt = data['dcoords']
  test_dxdt = data['test_dcoords']

  print('x.shape', x.shape)
  print('test_x.shape', test_x.shape)

  # vanilla train loop
  print('Training HNN begins...')
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    # train step (batch)
    ixs = tf.random.shuffle(tf.range(x.shape[0]))[:args.batch_size]
    loss = train_step(tf.gather(x, ixs), tf.gather(dxdt, ixs))

    # run test data
    test_loss = tf.constant(0.0, dtype=tf.float32)
    for i in range(test_x.shape[0] // args.batch_size_test):
      current_idx = tf.range(args.batch_size_test*i, min(args.batch_size_test*(i+1), test_x.shape[0]))
      test_dxdt_hat = model.time_derivative(tf.gather(test_x, current_idx))
      test_loss += L2_loss(tf.gather(test_dxdt, current_idx), test_dxdt_hat) * current_idx.shape[0]
    test_loss = test_loss / (args.batch_size_test * (i+1))

    # logging
    stats['train_loss'].append(loss.numpy())
    stats['test_loss'].append(test_loss.numpy())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.numpy(), test_loss.numpy()))

  train_dxdt_hat = model.time_derivative(x)
  train_dist = (dxdt - train_dxdt_hat)**2 
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(tf.reduce_mean(train_dist).numpy(), tf.math.reduce_std(train_dist).numpy()/math.sqrt(train_dist.shape[0]),
            tf.reduce_mean(test_dist).numpy(), tf.math.reduce_std(test_dist).numpy()/math.sqrt(test_dist.shape[0])))
  return model, stats

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(str(args.gpu_id))
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    result_path = '{}/results/{}_d{}_ns{}_ls{}_ss{}_{}'.format(args.save_dir, args.dist_name, args.input_dim,
                                                            args.num_samples, args.len_sample, args.step_size, args.grad_type)
    if not os.path.exists(result_path):
      os.makedirs(result_path)

    log_start(result_path+'/log.txt')
    model, stats = train(args)
    log_stop()

    # save
    save_path = '{}/ckp/{}_d{}_n{}_l{}_t{}_{}'.format(args.save_dir, args.dist_name, args.input_dim, args.num_samples, args.len_sample, args.total_steps, args.grad_type)
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    path = save_path + '/{}_d{}_n{}_l{}_t{}_{}.ckpt'.format(args.dist_name, args.input_dim, args.num_samples, args.len_sample, args.total_steps, args.grad_type)
    model.save_weights(path)