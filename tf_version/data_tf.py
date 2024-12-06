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

import os, sys
import tensorflow as tf
import tensorflow_probability as tfp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from functions_tf import functions_tf
from utils_tf import leapfrog_tf, to_pickle, from_pickle
from get_args import get_args
args = get_args()

def dynamics_fn_tf(t, coords):
    # dcoords: gradient of "functions" evaluated at "coords"
    dcoords = tfp.math.value_and_gradient(functions_tf, coords)[-1]
    dic1 = tf.split(dcoords, args.input_dim)
    S = tf.concat([dic1[int(args.input_dim/2)]], axis=0)
    for ii in range(int(args.input_dim/2)+1, args.input_dim, 1):
        S = tf.concat([S, dic1[ii]], axis=0)
    for ii in range(0, int(args.input_dim/2), 1):
        S = tf.concat([S, -dic1[ii]], axis=0)
    return S

def get_trajectory_tf(t_span=[0, args.len_sample], timescale=args.step_size, y0=None, **kwargs):
    n_steps = int((t_span[1] - t_span[0]) / timescale)

    if y0 is None:
        y0 = tf.random.normal(shape=[int(args.input_dim/2)])
        y0 = tf.concat([y0, tf.zeros(shape=[args.input_dim - int(args.input_dim/2)])], axis=0)
    lp_ivp = leapfrog_tf(dynamics_fn_tf, t_span, y0, n_steps, args.input_dim)
    # lp_ivp: input_dim x n_steps
    dic1 = tf.split(lp_ivp, args.input_dim)
    dydt = [dynamics_fn_tf(None, lp_ivp[:, ii]) for ii in range(0, lp_ivp.shape[1])]
    dydt = tf.stack(dydt, axis=1)
    ddic1 = tf.split(dydt, args.input_dim)
    # dic1: should be (position, momentum) state
    # dydt: numerical gradient
    # dic1: a list, len(dic1) = args.input_dim, each element is of 1 x n_steps
    # ddic1: a list, len(ddic1) = args.input_dim, each element is of 1 x n_steps
    return dic1, ddic1

def get_dataset_tf(seed=0, samples=args.num_samples, y_init=None, **kwargs):
    
    if args.should_load:
        path = '{}/data/{}.pkl'.format(args.load_dir, args.load_file_name)
        data = from_pickle(path)
        print("Successfully loaded data")
    else:
        data = {'meta': locals()}
        # randomly sample inputs
        tf.random.set_seed(seed) #
        xs, dxs = [], []

        if args.dist_name == 'Elliptic_PDE':
            tmp_path = '{}/data/{}_d{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, args.input_dim,
                                                args.num_samples, args.len_sample, args.step_size)
            if (not os.path.exists(tmp_path+'_obs.pkl')) or (not os.path.exists(tmp_path+'_pos.pkl')):
                raise Exception('Please run get_pos_elliptic_pde separately first.')
            
        count1 = 0
        if y_init is None:
            # two different ways to initialize the first and second half of y_init
            y_init = tf.zeros(shape=[int(args.input_dim/2)])
            # generate a random sample from normal distribution
            y_init = tf.concat([y_init, tf.random.normal(shape=[args.input_dim - int(args.input_dim/2)])], axis=0)

        print('Generating HMC samples for HNN training')

        for s in range(samples):
            print('Sample number ' + str(s+1) + ' of ' + str(samples))
            # print(y_init)
            dic1, ddic1 = get_trajectory_tf(y0=y_init, **kwargs)
            # the adding element is of shape step x args.input_dim
            dic1_tmp = tf.stack([tf.reshape(dic1[i], -1) for i in range(args.input_dim)], axis=1)
            xs.append(dic1_tmp)
            dxs.append(tf.stack([tf.reshape(ddic1[i], -1) for i in range(args.input_dim)], axis=1))

            # y_init = tf.zeros(args.input_dim)
            count1 = count1 + 1
            y_init = tf.concat([dic1_tmp[-1][:int(args.input_dim/2)], tf.random.normal(shape=[args.input_dim - int(args.input_dim/2)])], axis=0)

        data['coords'] = tf.concat(xs, axis=0)
        data['dcoords'] = tf.squeeze(tf.concat(dxs, axis=0))

        test_xs = []
        test_dxs = []
        test_samples = int(samples * args.test_fraction)
        for s in range(test_samples):
            print('Sample number (test) ' + str(s+1) + ' of ' + str(test_samples))
            # print(y_init)
            dic1, ddic1 = get_trajectory_tf(y0=y_init, **kwargs)
            # the adding element is of shape step x args.input_dim
            dic1_tmp = tf.stack([tf.reshape(dic1[i], -1) for i in range(args.input_dim)], axis=1)
            test_xs.append(dic1_tmp)
            test_dxs.append(tf.stack([tf.reshape(ddic1[i], -1) for i in range(args.input_dim)], axis=1))

            # y_init = tf.zeros(args.input_dim)
            count1 = count1 + 1
            y_init = tf.concat([dic1_tmp[-1][:int(args.input_dim/2)], tf.random.normal(shape=[args.input_dim - int(args.input_dim/2)])], axis=0)

        data['test_coords'] = tf.concat(test_xs, axis=0)
        data['test_dcoords'] = tf.squeeze(tf.concat(test_dxs, axis=0))

        # save data
        path = '{}/data/{}_d{}_ns{}_ls{}_ss{}.pkl'.format(args.save_dir, args.dist_name, args.input_dim,
                                                 args.num_samples, args.len_sample, args.step_size)
        tmp_dir = os.path.dirname(path)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        to_pickle(data, path)

    # return a dictionary with keys: '['coords', 'test_coords', 'dcoords', 'test_dcoords']'
    return data

def get_pos_elliptic_pde(seed=0):
    # randomly sample inputs
    tf.random.set_seed(seed)

    pos = tf.random.uniform(shape=[args.num_pos, int(args.input_dim/2)], minval=0, maxval=3, dtype=tf.float32)
    noises = tf.random.normal(shape=[args.num_pos], mean=0, stddev=1, dtype=tf.float32)
    obs = []
    for ii in range(args.num_pos):
        tmp = 2 * tf.math.cos(2 * pos[ii][0]) - 4 * (pos[ii][0] + pos[ii][1]) * tf.math.sin(2 * pos[ii][0]) \
                + 2 * tf.math.cos(2 * pos[ii][1]) - 4 * (pos[ii][0] + pos[ii][1]) * tf.math.sin(2 * pos[ii][1])
        tmp += noises[ii]
        obs.append(tmp)
    obs = tf.stack(obs, axis=0)
    
    # save data
    path = '{}/data/{}_d{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, args.input_dim,
                                                args.num_samples, args.len_sample, args.step_size)
    tmp_dir = os.path.dirname(path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    to_pickle(obs, path + '_obs.pkl')
    to_pickle(pos, path + '_pos.pkl')

# if __name__ == '__main__':
#     data = get_dataset_tf(args.seed)