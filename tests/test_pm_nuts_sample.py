import os
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from pseudo_marginal.pm_nuts_sample import sample


class mock_args:
    def __init__(self):
        self.save_dir = 'pseudo_marginal'
        self.dist_name = 'gauss_mix'
        self.T = 500
        self.n = 6
        self.p = 8
        self.N = 128
        self.num_samples = 400
        self.len_sample = 2.0
        self.step_size = 0.002
        self.learn_rate = 1e-4
        self.num_hmc_samples = 100
        self.nn_model_name = 'cnn'
        self.epsilon = 0.1
        self.rho_var = 1.0
        self.grad_flag = True
        self.num_flag = True
        self.target_dim = self.p + 5
        self.aux_dim = self.T * self.N
        self.num_burnin_samples = 50
        self.print_every = 1
        self.data_pth = 'pseudo_marginal/data'
        self.lf_threshold = 1000.0
        self.adapt_iter = 10
        self.delta = 0.75
        self.input_dim = 2 * self.target_dim + 2 * self.aux_dim, 
        self.hidden_dim = 100
        self.nn_out_dim = 26
        self.nonlinearity = 'sine'
        self.num_layers=3
        self.grad_type='solenoidal'

# @pytest.fixture
# def mock_os_exists(mocker):
#     yield mocker.patch('pseudo_marginal.pm_nuts_sample.os.path.exists')

@pytest.fixture
def mock_os_makedirs(mocker):
    yield mocker.patch('pseudo_marginal.pm_nuts_sample.os.makedirs')

@pytest.fixture
def mock_np(mocker):
    yield mocker.patch('pseudo_marginal.pm_nuts_sample.np.save')

@pytest.fixture
def mock_logging_start(mocker):
    yield mocker.patch('pseudo_marginal.pm_nuts_sample.log_start')

@pytest.fixture
def mock_logging_stop(mocker):
    yield mocker.patch('pseudo_marginal.pm_nuts_sample.log_stop')

def test_sample(mocker, mock_os_makedirs, mock_np, mock_logging_start, mock_logging_stop):
    args = mock_args()
    
    for x, y in zip([True, True, False, False], [True, False, True, False]):
        args.num_flag = x
        args.grad_flag = y

        mocker.patch('pseudo_marginal.pm_nuts_sample.tfp.mcmc.sample_chain', 
                    return_value=(tf.random.normal(shape=[1, args.target_dim+args.aux_dim], dtype=tf.float32), tf.constant(10.0)))

        # Call the sample function
        result = sample(args)

        # Assert that the result path was created
        result_path = '{}/results/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_nhs{}_{}_e{}_mass{}_grad{}_num{}'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, 
                                                                           args.num_hmc_samples,
                                                                           args.nn_model_name, args.epsilon, args.rho_var,
                                                                           args.grad_flag, args.num_flag)


        # Assert that log_start and log_stop were called
        mock_logging_start.assert_called_with(f'{result_path}/log.txt')
        mock_logging_stop.assert_called_with()

        # Assert that the result is as expected
        assert isinstance(result, tf.Tensor)
        assert result.shape == (1, 13)  # Assuming the mock state has this shape

        if args.num_flag:
            # Assert that np.save was called
            assert np.array_equal(mock_np.call_args[0][1], result.numpy())
            assert mock_np.call_args[0][0] == f'{result_path}/samples.npz'
        else:
            assert np.array_equal(mock_np.call_args[0][1], tf.constant(10.0).numpy())
            assert mock_np.call_args[0][0] == f'{result_path}/total_num_grad_steps.npz'
