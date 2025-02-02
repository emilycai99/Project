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
from nn_models import MLP, CNN_MLP, Info_MLP, Info_CNN_MLP
from hnn import HNN
from get_args import get_args
from utils import log_start, log_stop, integrator, numerical_grad, to_pickle
from functions import Hamiltonian_func_debug

##### Sampling code below #####
# To load the trained HNN model
def get_model(args):
    '''
    Description:
        load the model from the checkpoints
    '''
    args.input_dim = 2 * (args.target_dim + args.aux_dim)
    # nn_model = MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity,
    #                num_layers=args.num_layers)
    if args.nn_model_name == 'mlp':
        nn_model = MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                    num_layers=args.num_layers)
    elif args.nn_model_name == 'cnn':
        nn_model = CNN_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                    num_layers=args.num_layers)
    elif args.nn_model_name == 'info':
        nn_model = Info_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                    num_layers=args.num_layers)
    elif args.nn_model_name == 'infocnn':
        nn_model = Info_CNN_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                    num_layers=args.num_layers)
    else:
        raise NotImplementedError
    model = HNN(args, differentiable_model=nn_model, grad_type=args.grad_type)
    folder = '{}/ckp/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_{}'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, args.nn_model_name)
    # load the best checkpoint
    path = os.path.join(folder, 'best.ckpt')
    model.load_weights(path)
    model(tf.random.normal([1, args.input_dim]))
    return model

def hnn_grad(coords, model, target_dim, aux_dim):
    '''
    Description:
        HNN version of numerical_grad
    Args:
        coords: input 
        model: HNN
    '''
    # expand the batch size
    input = tf.expand_dims(tf.Variable(coords), axis=0)
    # squeeze the batch size
    dcoords = tf.squeeze(model.time_derivative(input), axis=0)
    return dcoords

# The stopping criterion in Eq.(19) of the paper
def stop_criterion(thetaminus, thetaplus, rhominus, rhoplus, uminus, uplus, pminus, pplus):
    '''
    Description:
        Check whether the stop criterion has been met.
        This version treats (theta, u) as a variable and (rho, p) as a variable.
    Return:
        True: if the stop criterion is not met;
        False: if the stop criterion is met.
    '''
    dtheta = thetaplus - thetaminus
    du = uplus - uminus
    cond1 = tf.tensordot(dtheta, rhoplus, axes=1) + tf.tensordot(du, pplus, axes=1)
    cond2 = tf.tensordot(dtheta, rhominus, axes=1) + tf.tensordot(du, pminus, axes=1)
    return (cond1 >= 0) & (cond2 >= 0)

log_counter_lf = 0

def build_tree_tf(theta, rho, u, p, log_slice_var, v, j, epsilon, joint0, call_lf, hnn_model, func, args):
    '''
    Description:
        the main recursion
    Args: 
        theta: position variable (target)
        rho: momentum variable (target)
        u: position variable (aux)
        p: momentum variable (aux)
        log_slice_var: the original logu in paper
        v in [-1, 1]: direction
        epsilon: step size
        joint0: previous Hamiltonian value
        call_lf: whether to use numerical gradient
        hnn_model: trained HNN
        func: Hamiltonian function
        args: arguments
    '''
    global log_counter_lf

    if j == 0:
        # y1.shape = args.input_dim = 2 * (args.target_dim + args.aux_dim)
        y1 = tf.concat((theta, rho, u, p), axis=0)
        # one leapfrog step
        # hnn_ivp1.shape =  args.input_dim x steps (steps=2 in this case)
        hnn_ivp1 = integrator(coords=y1, func=hnn_model, derivs_func=hnn_grad, h=v*epsilon, steps=1, 
                              target_dim=args.target_dim, aux_dim=args.aux_dim)
        # new theta, rho, u, p
        thetaprime, rhoprime, uprime, pprime = tf.split(hnn_ivp1[:, -1], 
                                                        num_or_size_splits=[args.target_dim, args.target_dim, args.aux_dim, args.aux_dim],
                                                        axis=0)
        # get Hamiltonian
        joint = func(hnn_ivp1[:, -1]) 
        
        # monitor: record the integration error
        monitor = log_slice_var + joint
        # call_lf as a flag whether to call the leapfrog method with numerical gradient
        call_lf = call_lf or int(monitor > args.hnn_threshold) 
        # sprime is to see whether the integration error is too large, sprime = 1 means continuing
        # note that different threshold is used for hnn and numerical gradient (hnn_threshold and lf_threshold)
        sprime = int(monitor <= args.hnn_threshold)
        
        # if call_if, completely discard the hnn's generated sample
        if call_lf:
            y1 = tf.concat((theta, rho, u, p), axis=0)
            # if call_lf, then directly using the numerical gradient
            # hnn_ivp1.shape = args.input_dim x steps
            hnn_ivp1 = integrator(coords=y1, func=func, derivs_func=numerical_grad, h=v*epsilon,
                                  steps=1, target_dim=args.target_dim, aux_dim=args.aux_dim)
            # new theta, rho, u, p
            thetaprime, rhoprime, uprime, pprime = tf.split(hnn_ivp1[:, -1], 
                                                            num_or_size_splits=[args.target_dim, args.target_dim, args.aux_dim, args.aux_dim],
                                                            axis=0)
            joint = func(hnn_ivp1[:, -1])
            sprime = int((log_slice_var + joint) <= args.lf_threshold)
            log_counter_lf += 1
        
        # nprime represents the size of the subtree
        # nprime = int(slice_var <= tf.math.exp(-joint))
        nprime = int(log_slice_var <= -joint)
        # since there is only one node, thetaminus = thetaplus and same for rho, u, p
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rhominus = rhoprime[:]
        rhoplus = rhoprime[:]
        uminus = uprime[:]
        uplus = uprime[:]
        pminus = pprime[:]
        pplus = pprime[:]
        alphaprime = tf.math.minimum(tf.ones(shape=[1]), tf.math.exp(joint0 - joint))
        # what is nalphaprime for?
        nalphaprime = 1
    else:
        # see Algorithm 3 in Hoffman et al., (2014)
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rhominus, uminus, pminus, thetaplus, rhoplus, uplus, pplus, thetaprime, rhoprime, uprime, pprime,\
            nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = build_tree_tf(theta, rho, u, p, log_slice_var, v, j - 1, epsilon, joint0, call_lf, hnn_model, func, args)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if sprime == 1:
            # Build the left tree
            if v == -1:
                thetaminus, rhominus, uminus, pminus, _, _, _, _, thetaprime2, rhoprime2, uprime2, pprime2, \
                    nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree_tf(thetaminus, rhominus, uminus, pminus, log_slice_var, v, j - 1, epsilon, joint0, call_lf, hnn_model, func, args)
            else:
                _, _, _, _, thetaplus, rhoplus, uplus, pplus, thetaprime2, rhoprime2, uprime2, pprime2,\
                    nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree_tf(thetaplus, rhoplus, uplus, pplus, log_slice_var, v, j - 1, epsilon, joint0, call_lf, hnn_model, func, args)
            # Choose which subtree to propagate a sample up from. (see Algorithm 3 in Hoffman)
            if (tf.random.uniform(shape=[]) < (float(nprime2) / max(float(nprime + nprime2), 1.))):
                thetaprime = thetaprime2[:]
                rhoprime = rhoprime2[:]
                uprime = uprime2[:]
                pprime = pprime2[:]
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rhominus, rhoplus, uminus, uplus, pminus, pplus))
            # Update the acceptance probability statistics. (what is this for?)
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rhominus, uminus, pminus, thetaplus, rhoplus, uplus, pplus, thetaprime, rhoprime, uprime, pprime, \
           nprime, sprime, alphaprime, nalphaprime, monitor, call_lf

################# Sampling Begins ###############################################
def sample(args):
    ##### User-defined sampling parameters #####
    '''
    N = args.num_hmc_samples # number of samples
    burn = args.num_burnin_samples # number of burn-in samples
    epsilon = args.epsilon # step size
    N_lf = args.num_cool_down # number of cool-down samples when HNN integration errors are high (see https://arxiv.org/abs/2208.06120)
    hnn_threshold = args.hnn_threshold # HNN integration error threshold (see https://arxiv.org/abs/2208.06120)
    lf_threshold = args.lf_threshold # Numerical gradient integration error threshold
    '''

    ##### log file #####
    result_path = '{}/results/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_nhs{}_{}'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, 
                                                                           args.num_hmc_samples,
                                                                           args.nn_model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    log_start(result_path+'/log.txt')
    # print the hyperparameters
    print('The following arguments are used: ')
    for arg in vars(args):
        print('  {} {}'.format(arg, getattr(args, arg) or ''))

    ##### initialize variables #####
    assert args.epsilon == args.step_size, 'unmatched epsilon and step size'
    # input dimension
    args.input_dim = 2 * (args.target_dim + args.aux_dim)
    # M: number of samples
    M = args.num_hmc_samples
    Madapt = 0
    # theta0: initial value (refer to pg.35 of Alenlov et al., 2021)
    theta0 = tf.constant([0.5838, 0.3805, -1.5062, -0.0442, 0.4717, -0.1435, 0.6371, -0.0522,
                          0.0, 0.0, math.log(1.0), math.log(0.1), 0.0], dtype=tf.float32)
    # u_sto: initial value
    u_sto = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)
    # samples: to store the samples
    samples = [None for _ in range(M + Madapt)]
    samples[0] = theta0
    # y0: to store both position and momentum variables
    y0 = tf.concat([theta0, tf.random.normal(shape=[args.target_dim], dtype=tf.float32),
                    u_sto, tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)], axis=0)

    # traj_len: record the trajectory length for each sample
    traj_len = [0 for _ in range(M)]
    # alpha_req: record the sum of probability (what is this for?)
    alpha_req = [tf.zeros(shape=[1]) for _ in range(M)]
    # H_store: record the Hamitonian value
    H_store = [tf.zeros(shape=[]) for _ in range(M)]
    # monitor_err: record the intergation error
    monitor_err = [tf.zeros(shape=[]) for _ in range(M)]

    # record the use of numerical gradient
    call_lf = 0
    counter_lf = 0
    # is_lf: record whether each sample is produced with numerical gradient or hnn
    is_lf = [0 for _ in range(M)]
    log_counter_lf_list = [0 for _ in range(M)]

    log_counter_lf = 0
    hnn_model = get_model(args)

    # get the Hamiltonian function
    ham = Hamiltonian_func_debug(args)
    func = ham.get_func()

    for m in range(1, M + Madapt, 1):
        if args.verbose and m % args.print_every == 0:
            print(m)
        # resample the momentum variables (note that: theta, u must be the stored samples from the last iteration)
        rho_sto = tf.random.normal(shape=[args.target_dim], dtype=tf.float32)
        p_sto = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)
        y0 = tf.concat([samples[m-1], rho_sto, u_sto, p_sto], axis=0)
        # get the Hamiltonian value
        joint = func(y0)
        # sample slice_var from uniform distribution
        # slice_var = tf.random.uniform(shape=[], minval=0.0, maxval=tf.math.exp(-joint))
        # MODIFIED !! a new way to generate slice_var (to avoid exp)
        tmp_slice_var = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
        log_slice_var = tf.math.log(tmp_slice_var) - joint

        # initializations
        ## if samples[m] not changed, then it keeps the original value
        samples[m] = samples[m - 1]
        ## initialize the tree
        thetaminus, rhominus, uminus, pminus = tf.split(y0, num_or_size_splits=[args.target_dim, args.target_dim, 
                                                                                args.aux_dim, args.aux_dim], axis=0)
        thetaplus = thetaminus
        rhoplus = rhominus
        uplus = uminus
        pplus = pminus
        ## initial height
        j = 0
        ## initially, the only valid point is the initial point.
        n = 1  
        ## main loop: will keep going until s == 0.
        s = 1  
        ## to count how many steps are using leapfrog with numerical gradient
        if call_lf:
            counter_lf +=1
        if counter_lf == args.num_cool_down:
            call_lf = 0
            counter_lf = 0

        while s == 1:
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * tf.cast(tf.random.uniform(shape=[]) < 0.5, tf.int32) - 1)

            # Double the size of the tree.
            if v == -1:
                thetaminus, rhominus, uminus, pminus, _, _, _, _, thetaprime, rhoprime, uprime, pprime, \
                    nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree_tf(thetaminus, rhominus, uminus, pminus, log_slice_var, v, j, args.epsilon, joint, call_lf, hnn_model, func, args)
                  
            else:
                _, _, _, _, thetaplus, rhoplus, uplus, pplus, thetaprime, rhoprime, uprime, pprime, \
                    nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree_tf(thetaplus, rhoplus, uplus, pplus, log_slice_var, v, j, args.epsilon, joint, call_lf, hnn_model, func, args)

            # Use Metropolis-Hastings to decide whether or not to move to 
            # a point from the half-tree we just generated.
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (tf.random.uniform(shape=[]) < _tmp):
                samples[m] = thetaprime[:]
                rho_sto = rhoprime[:]
                u_sto = uprime[:]
                p_sto = pprime[:]
            # Update number of valid points we've seen.
            n += nprime
            # Decide if it's time to stop.
            s = int(sprime and stop_criterion(thetaminus, thetaplus, rhominus, rhoplus, uminus, uplus, pminus, pplus))
            # Increasing depth.
            j += 1
            monitor_err[m] = monitor

        # at the end of iteration, record the results
        is_lf[m] = call_lf
        traj_len[m] = j
        alpha_req[m] = alpha
        H_store[m] = func(tf.concat([samples[m], rho_sto, u_sto, p_sto], axis=0))
        # try to calculate how many numerical gradient calculation is used
        log_counter_lf_list[m] = log_counter_lf
        log_counter_lf = 0

        if m % 500 == 0:
            print('Save results at {}th iterations'.format(m))
            # record the results
            np.save(result_path+'/samples.npz', tf.stack(samples[:m], axis=0).numpy())
            np.save(result_path+'/traj_len.npz', tf.stack(traj_len[:m], axis=0).numpy())
            np.save(result_path+'/alpha_req.npz', tf.concat(alpha_req[:m], axis=0).numpy())
            np.save(result_path+'/H_store.npz', tf.stack(H_store[:m], axis=0).numpy())
            np.save(result_path+'/monitor_err.npz',tf.stack(monitor_err[:m], axis=0).numpy())
            np.save(result_path+'/is_lf.npz', tf.stack(is_lf[:m], axis=0).numpy())
            np.save(result_path+'/log_counter_lf_list.npz', tf.stack(log_counter_lf_list[:m], axis=0).numpy())

    # samples: N x args.input_dim
    samples = tf.stack(samples, axis=0) 
    # traj_len: N
    traj_len = tf.stack(traj_len, axis=0)
    # alpha_req: N
    alpha_req = tf.concat(alpha_req, axis=0)
    # H_store: N
    H_store = tf.stack(H_store, axis=0)
    # monitor_err: N
    monitor_err = tf.stack(monitor_err, axis=0)
    # is_lf: N
    is_lf = tf.stack(is_lf, axis=0)
    # log_counter_lf_list
    num_gradients = sum(log_counter_lf_list)
    log_counter_lf_list = tf.stack(log_counter_lf_list, axis=0)

    # record the results
    np.save(result_path+'/samples.npz', samples.numpy())
    np.save(result_path+'/traj_len.npz', traj_len.numpy())
    np.save(result_path+'/alpha_req.npz', alpha_req.numpy())
    np.save(result_path+'/H_store.npz', H_store.numpy())
    np.save(result_path+'/monitor_err.npz', monitor_err.numpy())
    np.save(result_path+'/is_lf.npz', is_lf.numpy())
    np.save(result_path+'/log_counter_lf_list.npz', log_counter_lf_list.numpy())
        
    hnn_tf = samples[args.num_burnin_samples:M, :]
    ess_hnn = tfp.mcmc.effective_sample_size(hnn_tf).numpy()
    print(ess_hnn)
    print('num_grads: ', num_gradients)

    log_stop()

    return samples, traj_len, alpha_req, H_store, monitor_err, is_lf

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(str(args.gpu_id))
    sample(args)

    