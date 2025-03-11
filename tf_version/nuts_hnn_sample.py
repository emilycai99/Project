import os, sys
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from tf_version.get_args import get_args
from tf_version.nuts_hnn import NoUTurnSampler_HNN
from tf_version.utils_tf import log_start, log_stop, to_pickle
from tf_version.functions_tf import dist_func

################# Sampling Begins ###############################################
def sample(args):
    ##### result_path #####
    result_path = '{}/results/{}_d{}_ns{}_ls{}_ss{}_nhmc{}_{}'.format(args.save_dir, args.dist_name, args.input_dim,
                                                            args.num_samples, args.len_sample, args.step_size, args.num_hmc_samples, 
                                                            args.grad_type)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    log_start(result_path+'/log.txt')

    # print the hyperparameters
    print('The following arguments are used: ')
    for arg in vars(args):
        print('  {} {}'.format(arg, getattr(args, arg) or ''))
    
    dist_func_obj = dist_func(args)

    # trace total counts of numerical gradient steps at each iteration
    # also print the current iteration
    def trace_total_num_grad_steps(states, previous_kernel_results):
        global iter_count
        if iter_count % args.print_every == 0:
            print('Iter {}'.format(iter_count))
        iter_count += 1
        return previous_kernel_results.total_num_grad_steps_count

    states, total_num_grad_steps = tfp.mcmc.sample_chain(
        num_results=args.num_hmc_samples,
        num_burnin_steps=args.num_burnin_samples,
        current_state=tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32),
        kernel=NoUTurnSampler_HNN(
            target_log_prob_fn=dist_func_obj.get_target_log_prob_func,
            step_size=args.epsilon,
            max_energy_diff=args.lf_threshold,
            hnn_model_args=args
        ),
        trace_fn=trace_total_num_grad_steps
    )
    try:
        np.save(result_path+'/samples.npz', states.numpy())
        np.save(result_path+'/total_num_grad_steps.npz', total_num_grad_steps.numpy())
    except:
        to_pickle(states, result_path+'/samples.pkl')
        to_pickle(total_num_grad_steps, result_path+'/total_num_grad_steps.pkl')

    ess_hnn = tfp.mcmc.effective_sample_size(states).numpy()
    print(ess_hnn)

    log_stop()
    return states

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(str(args.gpu_id))
    
    iter_count = 0 
    sample(args)