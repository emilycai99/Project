import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import pickle

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts import *
from pseudo_marginal.grad import calculate_grad

def test_print_result():
    '''
    test for function print_result
    '''
    # Mock input coordinates
    coords = tf.constant([1.0, 2.0, 0.0, 0.0, 0.0], dtype=tf.float32)

    # Mock arguments
    args = mock_args()
    args.p = 0

    # Capture the printed output
    import io
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the print_result function
    print_result(coords, args)
    # Reset the standard output
    sys.stdout = sys.__stdout__
    # Get the printed output
    printed_text = captured_output.getvalue()

    # Check if the expected values are present in the printed output
    assert "param 1.0 2.0 1.0 1.0 0.5" in printed_text

class mock_args():
    def __init__(self):
        self.step_size = 0.02
        self.len_sample = 0.04
        self.target_dim = 13
        self.aux_dim = 1000
        self.input_dim = 2 * (self.target_dim + self.aux_dim)
        # self.input = tf.random.normal(shape=[self.input_dim], dtype=tf.float32, seed=42)
        self.num_samples = 2
        self.should_load = False
        self.dist_name = 'gauss_mix'
        self.T = 500
        self.n = 6
        self.p = 8
        self.N = 2
        self.test_fraction = 1.0
        self.save_dir = PARENT_DIR
        self.batch_size = 2
        self.batch_size_test = 2
        self.shuffle_buffer_size = 2
        self.nn_model_name = 'mlp'
        self.learn_rate = 1e-4
        self.seed = 0
        self.total_steps = 2
        self.hidden_dim = 10
        self.nn_out_dim = 26
        self.nonlinearity = 'sine'
        self.num_layers = 3
        self.data_pth = os.path.join(PARENT_DIR, 'data')
        self.grad_type = None
        self.penalty_strength = 0.0
        self.verbose = True
        self.print_every = 1
        self.grad_flag = False
        self.retrain = False
        self.hnn_threshold = 0.0
        self.lf_threshold = self.input_dim + 1
        self.num_hmc_samples = 2
        self.epsilon = self.step_size
        self.num_cool_down = 20
        self.num_burnin_samples = 0
        self.grad_mass_flag = False

@pytest.fixture
def mock_integrator():
    def _mock_integrator(coords, func, derivs_func, h, steps, target_dim, aux_dim):
        return tf.ones(shape=[2*(target_dim + aux_dim), steps+1])
    return _mock_integrator

@pytest.fixture
def mock_func():
    def _mock_func(self, coords):
        return tf.reduce_sum(coords)
    return _mock_func

@pytest.fixture
def mock_tf_random_normal():
    def _mock_tf_random_normal(shape, dtype=tf.float32):
        return tf.ones(shape=shape, dtype=dtype)
    return _mock_tf_random_normal

def test_sample(mocker, mock_tf_random_normal, mock_integrator, mock_func, tmp_path):
    '''
    unit test for sample
    '''
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.np.save')
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.log_start')
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.log_stop')
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.print')
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.tf.random.normal', mock_tf_random_normal)
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.integrator', mock_integrator)
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.calculate_grad.calculate_H', mock_func)

    args = mock_args()
    args.save_dir = tmp_path
    samples, H_store = sample(args)
    expected_samples = tf.ones(shape=[args.target_dim], dtype=tf.float32)
    print('samples', samples)
    print('expected_samples', expected_samples)
    assert tf.reduce_all(tf.math.equal(samples[-1], expected_samples))
    assert tf.math.equal(H_store[-1], tf.constant(args.input_dim, dtype=tf.float32)) 


def test_integration_sample(mocker):
    '''
    integration test for sample
    '''
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.np.save')
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.os.makedirs')
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.log_start')
    mocker.patch('pseudo_marginal.archive_sampling_files.hnn_nuts_online_num_no_nuts.log_stop')
    args = mock_args()
    cal_grad = calculate_grad(args)
    args.grad_func = cal_grad.grad_total

    samples, H_store = sample(args)

    assert samples.shape[0] == args.num_hmc_samples
    assert H_store.shape[0] == args.num_hmc_samples