import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import pickle

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from pseudo_marginal.hnn_nuts_online import *
from pseudo_marginal.grad import calculate_grad
from pseudo_marginal.nn_models import MLP
from pseudo_marginal.hnn import HNN

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
        self.save_dir = os.path.join(PARENT_DIR, 'pseudo_marginal')
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
        self.data_pth = os.path.join(PARENT_DIR, 'pseudo_marginal/data')
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

@pytest.fixture
def mock_load_weights():
    def _mock_load_weights(pth, *args):
        return
    return _mock_load_weights

def test_get_model(mocker, mock_load_weights):
    '''
    unit test for get_model function
    '''
    mocker.patch('pseudo_marginal.hnn_nuts_online.HNN.load_weights', mock_load_weights)
    args = mock_args()
    model = get_model(args)
    out = model(tf.random.normal(shape=[args.batch_size, args.input_dim]))
    assert (out[0].shape[0] == args.batch_size) and (out[0].shape[1] == args.nn_out_dim // 2) 

def test_hnn_grad():
    '''
    unit test for hnn_grad function
    '''
    # Mock input coordinates
    coords = tf.constant([[1.0, 2.0, 3.0]])

    # Mock HNN model
    class MockHNNModel:
        def time_derivative(self, input):
            return tf.constant([[2.0, 3.0, 4.0]])

    model = MockHNNModel()
    target_dim = 1
    aux_dim = 2

    expected_output = tf.constant([2.0, 3.0, 4.0])
    # Call the function
    result = hnn_grad(coords, model, target_dim, aux_dim)

    # Assert the output matches the expected result
    assert tf.reduce_all(tf.math.equal(result, expected_output))

def test_stop_criterion():
    '''
    unit test for stop_criterion function
    '''
    thetaminus = tf.constant([1.0, 2.0, 3.0])
    thetaplus = tf.constant([4.0, 5.0, 6.0])
    rhominus = tf.constant([0.1, 0.2, 0.3])
    rhoplus = tf.constant([0.4, 0.5, 0.6])
    uminus = tf.constant([0.7, 0.8, 0.9])
    uplus = tf.constant([1.0, 1.1, 1.2])
    pminus = tf.constant([1.3, 1.4, 1.5])
    pplus = tf.constant([1.6, 1.7, 1.8])

    result = stop_criterion(thetaminus, thetaplus, rhominus, rhoplus, uminus, uplus, pminus, pplus)
    expected_output = tf.constant(True)

    print(result)
    print(expected_output)
    assert tf.reduce_all(tf.math.equal(result, expected_output))

@pytest.fixture
def mock_integrator():
    def _mock_integrator(coords, func, derivs_func, h, steps, target_dim, aux_dim):
        return tf.ones(shape=[2*(target_dim + aux_dim), steps+1])
    return _mock_integrator

def mock_func(coords):
    return tf.reduce_sum(coords)

def test_build_tree_tf(mocker, mock_integrator):
    '''
    test for build_tree_tf
    '''
    args = mock_args()
    mocker.patch('pseudo_marginal.hnn_nuts_online.integrator', mock_integrator)
    log_slice_var = tf.zeros(shape=[], dtype=tf.float32)
    theta = tf.random.normal(shape=[args.target_dim])
    rho = tf.random.normal(shape=[args.target_dim])
    u = tf.random.normal(shape=[args.aux_dim])
    p = tf.random.normal(shape=[args.aux_dim])

    # j = 0 case
    thetaminus, rhominus, uminus, pminus, thetaplus, rhoplus, uplus, pplus, thetaprime, rhoprime, uprime, pprime, \
           nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = \
            build_tree_tf(theta, rho, u, p, log_slice_var, v=1, j=0, epsilon=0.01, joint0=tf.constant(0, dtype=tf.float32), 
                  call_lf=0, hnn_model=None, func=mock_func, args=args)
    
    assert tf.reduce_all(tf.math.equal(thetaminus, tf.ones_like(thetaminus)))
    assert tf.reduce_all(tf.math.equal(rhominus, tf.ones_like(rhominus)))
    assert tf.reduce_all(tf.math.equal(uminus, tf.ones_like(uminus)))
    assert tf.reduce_all(tf.math.equal(pminus, tf.ones_like(pminus)))
    assert tf.reduce_all(tf.math.equal(thetaplus, tf.ones_like(thetaplus)))
    assert tf.reduce_all(tf.math.equal(rhoplus, tf.ones_like(rhoplus)))
    assert tf.reduce_all(tf.math.equal(uplus, tf.ones_like(uplus)))
    assert tf.reduce_all(tf.math.equal(pplus, tf.ones_like(pplus)))
    assert tf.reduce_all(tf.math.equal(thetaprime, tf.ones_like(thetaprime)))
    assert tf.reduce_all(tf.math.equal(rhoprime, tf.ones_like(rhoprime)))
    assert tf.reduce_all(tf.math.equal(uprime, tf.ones_like(uprime)))
    assert tf.reduce_all(tf.math.equal(pprime, tf.ones_like(pprime)))

    assert nprime == 0
    assert sprime == 1
    assert alphaprime == tf.constant([0], dtype=tf.float32)
    assert nalphaprime == 1
    assert monitor == tf.constant(args.input_dim, dtype=tf.float32)
    assert call_lf == 1

    # j = 1 case

    thetaminus, rhominus, uminus, pminus, thetaplus, rhoplus, uplus, pplus, thetaprime, rhoprime, uprime, pprime, \
           nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = \
            build_tree_tf(theta, rho, u, p, log_slice_var, v=1, j=1, epsilon=0.01, joint0=tf.constant(0, dtype=tf.float32), 
                  call_lf=0, hnn_model=None, func=mock_func, args=args)
    
    assert tf.reduce_all(tf.math.equal(thetaminus, tf.ones_like(thetaminus)))
    assert tf.reduce_all(tf.math.equal(rhominus, tf.ones_like(rhominus)))
    assert tf.reduce_all(tf.math.equal(uminus, tf.ones_like(uminus)))
    assert tf.reduce_all(tf.math.equal(pminus, tf.ones_like(pminus)))
    assert tf.reduce_all(tf.math.equal(thetaplus, tf.ones_like(thetaplus)))
    assert tf.reduce_all(tf.math.equal(rhoplus, tf.ones_like(rhoplus)))
    assert tf.reduce_all(tf.math.equal(uplus, tf.ones_like(uplus)))
    assert tf.reduce_all(tf.math.equal(pplus, tf.ones_like(pplus)))
    assert tf.reduce_all(tf.math.equal(thetaprime, tf.ones_like(thetaprime)))
    assert tf.reduce_all(tf.math.equal(rhoprime, tf.ones_like(rhoprime)))
    assert tf.reduce_all(tf.math.equal(uprime, tf.ones_like(uprime)))
    assert tf.reduce_all(tf.math.equal(pprime, tf.ones_like(pprime)))

    assert nprime == 0
    assert sprime == 1
    assert alphaprime == tf.constant([0], dtype=tf.float32)
    assert nalphaprime == 2
    assert monitor == tf.constant(args.input_dim, dtype=tf.float32)
    assert call_lf == 1

@pytest.fixture
def mock_build_tree_tf():
    def mock_build_tree_tf(*args, **kwargs):
        args = mock_args()
        x = tf.ones(shape=[args.target_dim], dtype=tf.float32)
        y = tf.ones(shape=[args.aux_dim], dtype=tf.float32)
        return x, x, y, y, x, x, y, y, x, x, y, y, 0, 0, tf.constant([0], dtype=tf.float32),\
                2, tf.constant(args.input_dim, dtype=tf.float32), 1
    return mock_build_tree_tf

@pytest.fixture
def mock_get_model():
    def _mock_get_model(*args, **kwargs):
        return None
    return _mock_get_model

def test_sample(mocker, mock_build_tree_tf, mock_get_model):
    '''
    unit test for sample
    '''
    mocker.patch('pseudo_marginal.hnn_nuts_online.build_tree_tf', mock_build_tree_tf)
    mocker.patch('pseudo_marginal.hnn_nuts_online.get_model', mock_get_model)
    mocker.patch('pseudo_marginal.hnn_nuts_online.np.save')
    mocker.patch('pseudo_marginal.hnn_nuts_online.log_start')
    mocker.patch('pseudo_marginal.hnn_nuts_online.log_stop')
    mocker.patch('pseudo_marginal.hnn_nuts_online.print')
    mocker.patch('pseudo_marginal.hnn_nuts_online_epsilon.os.makedirs')

    args = mock_args()
    samples, traj_len, alpha_req, H_store, monitor_err, is_lf = sample(args)
    
    theta0 = tf.constant([0.5838, 0.3805, -1.5062, -0.0442, 0.4717, -0.1435, 0.6371, -0.0522,
                          0.0, 0.0, math.log(1.0), math.log(0.1), 0.0], dtype=tf.float32)
    expected_samples = tf.stack([theta0, theta0], axis=0)
    expected_traj_len = tf.constant([0, 1], dtype=tf.int32)
    expected_alpha_req = tf.constant([0, 0], dtype=tf.float32)

    expected_monitor_err = tf.constant([0, 2*(args.target_dim + args.aux_dim)], dtype=tf.float32)
    expected_is_lf = tf.constant([0, 1], dtype=tf.int32)
    
    assert tf.reduce_all(tf.math.equal(samples, expected_samples))
    assert tf.reduce_all(tf.math.equal(traj_len, expected_traj_len))
    assert tf.reduce_all(tf.math.equal(alpha_req, expected_alpha_req))
    assert tf.reduce_all(tf.math.equal(monitor_err, expected_monitor_err))
    assert tf.reduce_all(tf.math.equal(is_lf, expected_is_lf))

@pytest.fixture
def mock_get_model_integration():
    def _mock_get_model(*args, **kwargs):
        tmp = mock_args()
        nn_model = MLP(tmp.input_dim, tmp.hidden_dim, tmp.nn_out_dim)
        model = HNN(tmp, nn_model)
        model(tf.random.normal(shape=[tmp.batch_size, tmp.input_dim]))
        return model
    return _mock_get_model

def test_integration_sample(mocker, mock_get_model_integration):
    '''
    integration test for sample
    '''
    mocker.patch('pseudo_marginal.hnn_nuts_online.get_model', mock_get_model_integration)
    mocker.patch('pseudo_marginal.hnn_nuts_online.np.save')
    mocker.patch('pseudo_marginal.hnn_nuts_online.log_start')
    mocker.patch('pseudo_marginal.hnn_nuts_online.log_stop')
    mocker.patch('pseudo_marginal.hnn_nuts_online.os.makedirs')
    args = mock_args()
    cal_grad = calculate_grad(args)
    args.grad_func = cal_grad.grad_total

    samples, traj_len, alpha_req, H_store, monitor_err, is_lf = sample(args)

    assert samples.shape[0] == args.num_hmc_samples
    assert traj_len.shape[0] == args.num_hmc_samples
    assert alpha_req.shape[0] == args.num_hmc_samples
    assert H_store.shape[0] == args.num_hmc_samples
    assert monitor_err.shape[0] == args.num_hmc_samples
    assert is_lf.shape[0] == args.num_hmc_samples