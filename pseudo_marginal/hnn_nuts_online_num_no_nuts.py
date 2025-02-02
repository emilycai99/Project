# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Coded by Som Dhulipala at Idaho National Laboratory
# Parts of this code were borrowed from https://github.com/mfouesneau/NUTS which has an MIT License
# No-U-Turn Sampling with HNNs

import os, sys
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from get_args import get_args
from utils import log_start, log_stop
from grad import calculate_grad, numerical_grad_debug
from utils import integrator

##### Sampling code below #####
def print_result(coords, args):
    beta = coords[:args.p]
    mu1 = coords[args.p].numpy()
    mu2 = coords[args.p+1].numpy()
    lambda1 = np.exp(coords[args.p+2])
    lambda2 = np.exp(coords[args.p+3])
    w = tf.math.sigmoid(coords[args.p+4]).numpy()   
    print('param', mu1, mu2, lambda1, lambda2, w)

################# Sampling Begins ###############################################
def sample(args):
    ##### User-defined sampling parameters #####
    '''
    N = args.num_hmc_samples # number of samples
    burn = args.num_burnin_samples # number of burn-in samples
    epsilon = args.epsilon # step size
    '''

    ##### log file #####
    result_path = '{}/results/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_nhs{}_{}_e{}_nonuts'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, 
                                                                           args.num_hmc_samples,
                                                                           args.nn_model_name, args.epsilon)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    log_start(result_path+'/log.txt')
    # print the hyperparameters
    print('The following arguments are used: ')
    for arg in vars(args):
        print('  {} {}'.format(arg, getattr(args, arg) or ''))

    ##### initialize variables #####
    # input dimension
    args.input_dim = 2 * (args.target_dim + args.aux_dim)
    # M: number of samples
    M = args.num_hmc_samples
    Madapt = 0
    # theta0: initial value (refer to pg.35 of Alenlov et al., 2021)
    # theta0 = tf.constant([0.5838, 0.3805, -1.5062, -0.0442, 0.4717, -0.1435, 0.6371, -0.0522,
    #                       0.0, 0.0, math.log(1.0), math.log(0.1), 0.0], dtype=tf.float32)
    theta0 = tf.random.normal(shape=[args.target_dim], dtype=tf.float32)
    # u_sto: initial value
    u_sto = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)
    # samples: to store the samples
    samples = [None for _ in range(M + Madapt)]
    samples[0] = theta0
    # y0: to store both position and momentum variables
    y0 = tf.concat([theta0, tf.random.normal(shape=[args.target_dim], dtype=tf.float32),
                    u_sto, tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)], axis=0)

    # H_store: record the Hamitonian value
    H_store = [tf.zeros(shape=[]) for _ in range(M)]

    # add a new gradient calculation
    cal_grad = calculate_grad(args)
    args.grad_func = cal_grad.grad_total
    func = cal_grad.calculate_H

    for m in range(1, M + Madapt, 1):
        if args.verbose and m % args.print_every == 0:
            print(m)
        # resample the momentum variables (note that: theta, u must be the stored samples from the last iteration)
        rho_sto = tf.random.normal(shape=[args.target_dim], dtype=tf.float32)
        p_sto = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)
        y0 = tf.concat([samples[m-1], rho_sto, u_sto, p_sto], axis=0)

        # initializations
        ## if samples[m] not changed, then it keeps the original value
        samples[m] = samples[m - 1]
        hnn_ivp1 = integrator(coords=y0, func=args.grad_func, derivs_func=numerical_grad_debug, h=args.epsilon, steps=50, 
                              target_dim=args.target_dim, aux_dim=args.aux_dim)
        # new theta, rho, u, p
        theta, rho, u, p = tf.split(hnn_ivp1[:, -1], num_or_size_splits=[args.target_dim, args.target_dim, args.aux_dim, args.aux_dim], axis=0)
        if (tf.random.uniform(shape=[]) < max(1.0, tf.math.exp(func(y0) - func(hnn_ivp1[:, -1])))):
            samples[m] = theta
            u_sto = u
            rho_sto = rho
            p_sto = p
        H_store[m] = func(tf.concat([samples[m], rho_sto, u_sto, p_sto], axis=0))
        print('H_store', H_store[m].numpy())
        print_result(samples[m], args)

        if m % 100 == 0:
            print('Save results at {}th iterations'.format(m))
            # record the results
            np.save(result_path+'/samples.npz', tf.stack(samples[:m], axis=0).numpy())
            np.save(result_path+'/H_store.npz', tf.stack(H_store[:m], axis=0).numpy())
           
    # samples: N x args.input_dim
    samples = tf.stack(samples, axis=0) 
    # H_store: N
    H_store = tf.stack(H_store, axis=0)

    # record the results
    np.save(result_path+'/samples.npz', samples.numpy())
    np.save(result_path+'/H_store.npz', H_store.numpy())
        
    hnn_tf = samples[args.num_burnin_samples:M, :]
    ess_hnn = tfp.mcmc.effective_sample_size(hnn_tf).numpy()
    print(ess_hnn)

    log_stop()

    return samples, H_store

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(str(args.gpu_id))
    sample(args)

    