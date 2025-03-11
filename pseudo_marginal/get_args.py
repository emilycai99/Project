import argparse
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_float_list(input_str):
    try:
        float_list = [float(num) for num in input_str.split(',')]
        if len(float_list) == 1:
            return float_list[0]
        else:
            return float_list
    except ValueError:
        raise argparse.ArgumentTypeError("All values must be valid floats separated by commas")

def get_args(args):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--dist_name', default='gauss_mix', type=str, help='name of the probability distribution function')

    ################### Hyperparameters related to GLMM ########################################
    parser.add_argument('--data_pth', default=os.path.join(THIS_DIR, 'data'), type=str, help='where to load the data')
    parser.add_argument('--p', default=8, type=int, help='the dimension of fixed effect')
    parser.add_argument('--T', default=500, type=int, help='sample size')
    parser.add_argument('--N', default=128, type=int, help='number of samples used for importance sampling')
    parser.add_argument('--n', default=6, type=int, help='number of samples for each subject')

    ################### Hyperparameters related to HNN training ################################
    parser.add_argument('--input_dim', default=None, type=int, help='dimension of input dim for HNN')
    parser.add_argument('--target_dim', default=13, type=int, help='dimension of the parameters of interest')
    parser.add_argument('--aux_dim', default=500*128, type=int, help='dimension of auxiliary variables')
    parser.add_argument('--num_samples', default=10, type=int, help='number of training samples simulated using Hamiltonian Monte Carlo')
    parser.add_argument('--len_sample', default=50, type=float, help='length of Hamiltonian trajectory for each training sample')
    parser.add_argument('--step_size', default=0.025, type=float, help='step size for time integration')

    parser.add_argument('--test_fraction', default=0.1, type=float, help='fraction of testing samples')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--load_dir', default=THIS_DIR, type=str, help='where to load the training data from')
    parser.add_argument('--should_load', default=False, action='store_true', help='should load training data?')
    parser.add_argument('--load_file_name', default='nD_standard_Gaussian', type=str, help='if load training data, the file name (.pkl format)')

    ################### Hyperparameters related to HNN setting ################################
    parser.add_argument('--nn_out_dim', default=26, type=int, help='dimension of HNN last layer')
    parser.add_argument('--hidden_dim', default=100, type=int, help='hidden dimension of mlp')
    parser.add_argument('--num_layers', default=3, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--batch_size_test', default=1000, type=int, help='batch_size_test')
    parser.add_argument('--nonlinearity', default='sine', type=str, help='neural net nonlinearity')
    parser.add_argument('--grad_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--total_steps', default=5000, type=int, help='number of gradient steps')

    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")

    parser.add_argument('--shuffle_buffer_size', default=100, type=int, help='the buffer size used for shuffling the dataset')

    parser.add_argument('--penalty_strength', default=0.0, type=float, help='penalty for l1 loss')
    parser.add_argument('--nn_model_name', default='mlp', type=str, help='which nn_model to use')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='decay rate for the learning rate')
    parser.add_argument('--retrain', default=False, action='store_true', help='whether to retrain the model')
    parser.add_argument('--retrain_lr', default=None, type=float, help='retrain learning rate')

    ################### Hyperparameters related to HMC setting ################################
    parser.add_argument('--num_hmc_samples', default=14000, type=int, help='number of hmc samples')
    parser.add_argument('--num_burnin_samples', default=7000, type=int, help='number of burn-in samples')
    parser.add_argument('--epsilon', default=0.02, type=float, help='step size for time integration in hmc')
    parser.add_argument('--num_cool_down', default=20, type=int, help='number of cool-down samples when HNN integration errors are high')
    parser.add_argument('--hnn_threshold', default=10.0, type=float, help='HNN integration error threshold')
    parser.add_argument('--lf_threshold', default=1000.0, type=float, help='Numerical gradient integration error threshold')
    parser.add_argument('--adapt_iter', default=0, type=int, help='number of adaptive iterations used for tune epsilon')
    parser.add_argument('--delta', default=0.65, type=float, help='expected acceptance rate')

    parser.add_argument('--grad_flag', default=False, action='store_true', help='whether to use the manual grad')
    parser.add_argument('--grad_mass_flag', default=False, action='store_true', help='whether to use the non identity mass matrix')
    parser.add_argument('--rho_var', default=1.0, type=parse_float_list, help='the variance for rho')
    parser.add_argument('--num_flag', default=False, action='store_true', help='whether to use numerical gradients or HNN')

    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    return parser.parse_args(args)