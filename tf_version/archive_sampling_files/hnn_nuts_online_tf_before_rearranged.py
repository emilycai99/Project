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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.nn_models_tf import MLP
from tf_version.hnn_tf import HNN
from tf_version.get_args import get_args
from tf_version.utils_tf import leapfrog_tf, log_start, log_stop
from tf_version.functions_tf import dist_func
from tf_version.data_tf import dynamics_fn_tf

args = get_args()
dist_func_obj = dist_func(args)
functions_tf = dist_func_obj.get_Hamiltonian

os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(str(args.gpu_id))

##### result_path #####
result_path = '{}/results/{}_d{}_ns{}_ls{}_ss{}_{}'.format(args.save_dir, args.dist_name, args.input_dim,
                                                           args.num_samples, args.len_sample, args.step_size, args.grad_type)
if not os.path.exists(result_path):
      os.makedirs(result_path)
    
log_start(result_path+'/log.txt')

##### User-defined sampling parameters #####
N = args.num_hmc_samples # number of samples
burn = args.num_burnin_samples # number of burn-in samples
epsilon = args.epsilon # step size
N_lf = args.num_cool_down # number of cool-down samples when HNN integration errors are high (see https://arxiv.org/abs/2208.06120)
hnn_threshold = args.hnn_threshold # HNN integration error threshold (see https://arxiv.org/abs/2208.06120)
lf_threshold = args.lf_threshold # Numerical gradient integration error threshold

##### Sampling code below #####
# To load the trained HNN model
def get_model(args, baseline):
    output_dim = args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity,
                   num_layers=args.num_layers)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              grad_type=args.grad_type, baseline=baseline)
    path = '{}/ckp/{}_d{}_n{}_l{}_t{}_{}/{}_d{}_n{}_l{}_t{}_{}.ckpt'.format(args.save_dir, args.dist_name, args.input_dim, args.num_samples, 
                                                      args.len_sample, args.total_steps, args.grad_type, args.dist_name, args.input_dim, 
                                                      args.num_samples, args.len_sample, args.total_steps, args.grad_type)
    model.load_weights(path)
    model(tf.random.normal([args.batch_size, args.input_dim]))
    return model

# To write a leapfrog with HNN graidents
def integrate_model_tf(model, t_span, y0, n, **kwargs):
    def fun(t, tf_x):
        x = tf.expand_dims(tf.Variable(tf_x), axis=0)
        dx = tf.reshape(model.time_derivative(x), -1)
        return dx
    return leapfrog_tf(fun, t_span, y0, n, args.input_dim)

# hnn_model = get_model(args, baseline=False)

# The stopping criterion in Eq.(19) of the paper
def stop_criterion_tf(thetaminus, thetaplus, rminus, rplus):
    dtheta = thetaplus - thetaminus
    return (tf.tensordot(dtheta, rminus, axes=1) >= 0) & (tf.tensordot(dtheta, rplus, axes=1) >= 0)

log_counter_lf = 0

def build_tree_tf(theta, r, u, v, j, epsilon, joint0, call_lf, hnn_model):
    """The main recursion."""
    """
    Arguments 
    theta: position variable
    r: momentum variable
    v in [-1, 1]: direction
    epsilon: step size
    joint0: previous Hamiltonian value
    call_lf: whether to use numerical gradient
    """
    global log_counter_lf

    if j == 0:
        t_span1 = [0, v * epsilon]
        # y1.shape = args.input_dim
        y1 = tf.concat((theta, r), axis=0)
        # one leapfrog step
        # hnn_ivp1.shape =  args.input_dim x steps(steps=2 in this case)
        hnn_ivp1 = integrate_model_tf(hnn_model, t_span1, y1, 1)
        # new theta
        thetaprime = hnn_ivp1[:int(args.input_dim/2), -1]
        # new rprime
        rprime = hnn_ivp1[int(args.input_dim/2):, -1]
        # get hamiltonian
        joint = functions_tf(hnn_ivp1[:,-1]) 
        
        # monitor: record the integration error
        monitor = tf.math.log(u) + joint
        # call_lf as a flag whether to call the leapfrog method with numerical gradient
        call_lf = call_lf or int(monitor > hnn_threshold) 
        # sprime is to see whether the integration error is too large, sprime = 1 means continuing
        # note that different threshold is used for hnn and numerical gradient (hnn_threshold and lf_threshold)
        sprime = int(monitor <= hnn_threshold)
        
        # if call_if, completely discard the hnn's generated sample
        if call_lf:
            t_span1 = [0, v * epsilon]
            y1 = tf.concat((theta, r), axis=0)
            # if call_lf, then directly using the numerical gradient
            # hnn_ivp1.shape = args.input_dim x steps
            hnn_ivp1 = leapfrog_tf(dynamics_fn_tf, t_span1, y1, 1, int(args.input_dim))
            thetaprime = hnn_ivp1[:int(args.input_dim/2), -1]
            rprime = hnn_ivp1[int(args.input_dim/2):, -1]
            joint = functions_tf(hnn_ivp1[:, -1])
            sprime = int((tf.math.log(u) + joint) <= lf_threshold)
            log_counter_lf += 1
        
        # nprime represents the size of the subtree
        nprime = int(u <= tf.math.exp(-joint))
        # since there is only one node, thetaminus = thetaplus and same for r
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        alphaprime = tf.math.minimum(tf.ones(shape=[1]), tf.math.exp(joint0 - joint))
        # what is nalphaprime for?
        nalphaprime = 1
    else:
        # see Algorithm 3 in Hoffman et al., (2014)
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = build_tree_tf(theta, r, u, v, j - 1, epsilon, joint0, call_lf, hnn_model)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if sprime == 1:
            # Build the left tree
            if v == -1:
                thetaminus, rminus, _, _, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree_tf(thetaminus, rminus, u, v, j - 1, epsilon, joint0, call_lf, hnn_model)
            else:
                _, _, thetaplus, rplus, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree_tf(thetaplus, rplus, u, v, j - 1, epsilon, joint0, call_lf, hnn_model)
            # Choose which subtree to propagate a sample up from. (see Algorithm 3 in Hoffman)
            if (tf.random.uniform(shape=[]) < (float(nprime2) / max(float(nprime + nprime2), 1.))):
                thetaprime = thetaprime2[:]
                rprime = rprime2[:]
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion_tf(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics. (what is this for?)
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf

################# Sampling Begins ###############################################
if __name__ == '__main__':
    # print the hyperparameters
    print('The following arguments are used: ')
    for arg in vars(args):
        print('  {} {}'.format(arg, getattr(args, arg) or ''))
    # D: dimension of sample
    D = int(args.input_dim/2)
    # M: number of samples
    M = N
    Madapt = 0
    # theta0: initial value
    theta0 = tf.ones(D)
    # samples: to store the samples
    samples = [None for _ in range(M +Madapt)]
    samples[0] = theta0
    # y0: to store both position and momentum variables
    y0 = tf.random.normal(shape=[args.input_dim])

    # traj_len: record the trajectory length for each sample
    traj_len = [0 for _ in range(M)]
    # alpha_req: record the sum of probability (what is this for?)
    alpha_req = [tf.zeros(shape=[1]) for _ in range(M)]
    # H_store: record the Hamitonian value
    H_store = [tf.zeros(shape=[1]) for _ in range(M)]
    # monitor_err: record the intergation error
    monitor_err = [tf.zeros(shape=[1]) for _ in range(M)]
    call_lf = 0
    counter_lf = 0
    # is_lf: record whether each sample is produced with numerical gradient or hnn
    is_lf = [0 for _ in range(M)]
    log_counter_lf_list = [0 for _ in range(M)]

    hnn_model = get_model(args, baseline=False)

    for m in range(1, M + Madapt, 1):
        if args.verbose and m % args.print_every == 0:
            print(m)
        # resample the momentum variable
        y0 = tf.concat((y0[:int(args.input_dim/2)], tf.random.normal(shape=[int(args.input_dim) - int(args.input_dim/2)])), axis=0)
        # get the Hamiltonian value
        joint = functions_tf(y0)
        # sample u from uniform distribution
        u = tf.random.uniform(shape=[], minval=0.0, maxval=tf.math.exp(-joint))

        # if samples[m] not changed, then it keeps the original value
        samples[m] = samples[m - 1]

        # initialize the tree
        thetaminus = samples[m - 1]
        thetaplus = samples[m - 1]
        rminus = y0[int(args.input_dim/2):]
        rplus = y0[int(args.input_dim/2):]
        ## initial height
        j = 0
        ## initially, the only valid point is the initial point.
        n = 1  
        ## main loop: will keep going until s == 0.
        s = 1  

        # to count how many steps are using leapfrog with numerical gradient
        if call_lf:
            counter_lf +=1
        if counter_lf == N_lf:
            call_lf = 0
            counter_lf = 0

        while s == 1:
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * tf.cast(tf.random.uniform(shape=[]) < 0.5, tf.int32) - 1)

            # Double the size of the tree.
            if v == -1:
                thetaminus, rminus, _, _, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree_tf(thetaminus, rminus, u, v, j, epsilon, joint, call_lf, hnn_model)
            else:
                _, _, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree_tf(thetaplus, rplus, u, v, j, epsilon, joint, call_lf, hnn_model)

            # Use Metropolis-Hastings to decide whether or not to move to 
            # a point from the half-tree we just generated.
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (tf.random.uniform(shape=[]) < _tmp):
                samples[m] = thetaprime[:]
                r_sto = rprime
            # Update number of valid points we've seen.
            n += nprime
            # Decide if it's time to stop.
            s = int(sprime and stop_criterion_tf(thetaminus, thetaplus, rminus, rplus))
            # Increasing depth.
            j += 1
            monitor_err[m] = monitor

        is_lf[m] = call_lf
        traj_len[m] = j
        alpha_req[m] = alpha
        y0 = tf.concat([samples[m], y0[int(args.input_dim/2):]], axis=0)
        H_store[m] = functions_tf(tf.concat((samples[m], r_sto), axis=0))
        # try to calculate how many numerical gradient calculation is used
        log_counter_lf_list[m] = log_counter_lf
        log_counter_lf = 0


    # samples: N x args.input_dim / 2
    samples = tf.stack(samples, axis=0) 
    # traj_len: N
    traj_len = tf.stack(traj_len, axis=0)
    # alpha_req: N
    alpha_req = tf.concat(alpha_req, axis=0)
    # H_store: N
    H_store = tf.concat(H_store, axis=0)
    # monitor_err: N
    monitor_err = tf.concat(monitor_err, axis=0)
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
        
    hnn_tf = samples[burn:M, :]
    ess_hnn = tfp.mcmc.effective_sample_size(hnn_tf).numpy()
    print(ess_hnn)
    print('num_grads: ', num_gradients)

    log_stop()