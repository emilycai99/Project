import os, sys
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.get_args import get_args
from pseudo_marginal.pm_nuts import PseudoMarginal_NoUTurnSampler
from pseudo_marginal.utils import log_start, log_stop, to_pickle
from pseudo_marginal.grad import calculate_grad

def sample_with_dual_averaging(args):
    # Though it can be easily extended to HNN version, we limit it to the case when num_flag = True and grad_flag = True for the debugging purpose
    assert args.num_flag and args.grad_flag, 'Dual averaging only works with num_flag=True and grad_flag=True cases for now!'
    
    ##### result_path #####
    result_path = '{}/results/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_nhs{}_{}_e{}_mass{}_grad{}_num{}_dual_averaging'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, 
                                                                           args.num_hmc_samples,
                                                                           args.nn_model_name, args.epsilon, args.rho_var,
                                                                           args.grad_flag, args.num_flag)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    log_start(result_path+'/log.txt')

    args.input_dim = 2 * (args.target_dim + args.aux_dim)
    # print the hyperparameters
    print('The following arguments are used: ')
    for arg in vars(args):
        print('  {} {}'.format(arg, getattr(args, arg) or ''))

    prob_func = calculate_grad(args)

    theta_init = tf.constant([0.5838, 0.3805, -1.5062, -0.0442, 0.4717, -0.1435, 0.6371, -0.0522,
                          0.0, 0.0, math.log(1.0), math.log(0.1), 0.0], dtype=tf.float32)
    u_init = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)

    def trace_step_size_and_leapfrogs_taken(states, previous_kernel_results):
        global iter_count
        if iter_count % args.print_every == 0:
            print('Iter {}'.format(iter_count))
        iter_count += 1
        return [previous_kernel_results.inner_results.step_size,
                previous_kernel_results.inner_results.leapfrogs_taken]


    kernel = PseudoMarginal_NoUTurnSampler(
            target_log_prob_fn=prob_func.get_target_log_prob,
            step_size=args.epsilon,
            max_energy_diff=args.lf_threshold,
            pm_nuts_args=args)
    
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel, 
        num_adaptation_steps=args.adapt_iter,
        target_accept_prob=args.delta
    )

    states, [step_size, leapfrogs_taken] = tfp.mcmc.sample_chain(
        num_results=args.num_hmc_samples,
        num_burnin_steps=args.num_burnin_samples,
        current_state=tf.concat([theta_init, u_init], axis=0),
        kernel=kernel,
        trace_fn=trace_step_size_and_leapfrogs_taken)

    states, _ = tf.split(states, [args.target_dim, args.aux_dim], axis=-1)
    try:
        np.save(result_path+'/samples.npz', states.numpy())
    except:
        to_pickle(states, result_path+'/samples.pkl')
    try:
        np.save(result_path+'/step_size.npz', step_size.numpy())
    except:
        to_pickle(step_size, result_path+'/step_size.pkl')
    try:
        np.save(result_path+'/leapfrogs_taken.npz', leapfrogs_taken.numpy())
    except:
        to_pickle(leapfrogs_taken, result_path+'/leapfrogs_taken.pkl')

    ess_hnn = tfp.mcmc.effective_sample_size(states).numpy()
    print(ess_hnn)

    log_stop()
    return states

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(str(args.gpu_id))
    
    iter_count = 0 
    sample_with_dual_averaging(args)
