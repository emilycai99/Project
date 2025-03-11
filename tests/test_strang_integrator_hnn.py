import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import math

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.strang_integrator_hnn import _one_step, _one_step_hnn, StrangIntegrator, StrangIntegrator_HNN,\
    _hnn_value_and_gradients, hnn_fn_and_grads, StrangIntegrator_grad, _one_step_grad, manual_fn_and_grads, _manual_value_and_gradients,\
    process_args, process_args_grad, process_args_hnn
from pseudo_marginal.utils import numerical_grad
import tensorflow_probability.python.mcmc.internal.util as mcmc_util


def func(x):
    return tf.tensordot(x, x, axes=1)

class mock_args:
    def __init__(self):
        self.input_dim = 10
        self.target_dim = 3
        self.aux_dim = 2
        self.rho_var = 2.0
        self.rho_precision_mat = tf.eye(self.target_dim, dtype=tf.float32) * 1.0 / self.rho_var

def log_prior(theta):
    return tf.tensordot(theta, theta, axes=1)

def log_phat(theta, u):
    return tf.tensordot(theta, theta, axes=1) + 2.0 * tf.tensordot(u, u, axes=1)

args = mock_args()
def mock_H_func(coords):
    rho_precision_mat = args.rho_precision_mat
    theta, rho, u, p = tf.split(coords, [args.target_dim, args.target_dim,
                                         args.aux_dim, args.aux_dim], axis=0)
    rho_expand = tf.expand_dims(rho, axis=-1)
    H = -log_prior(theta) -log_phat(theta, u) + 0.5 * tf.squeeze(tf.matmul(tf.matmul(rho_expand, rho_precision_mat, transpose_a=True), rho_expand)) \
        + 0.5 * (tf.tensordot(u, u, axes=1) + tf.tensordot(p, p, axes=1))
    return H

def mock_target_func(coords):
    theta, u = tf.split(coords, [args.target_dim, args.aux_dim], axis=0)
    log_prob = log_prior(theta) + log_phat(theta, u) - 0.5 * (tf.tensordot(u, u, axes=1))
    return log_prob

def integrator_one_step_mass(coords, func, derivs_func, h, target_dim, aux_dim, args):
    '''
    Description:
        Implement the one-step integrator in Appendix A of PM-HMC with rho's covariance mat not identity
    Return:
        coords_new: after one-step integration, same size as coords
    '''

    theta, rho, u, p = tf.split(coords, [target_dim, target_dim, aux_dim, aux_dim])
    if isinstance(args.rho_var, float):
        rho_precision_mat = tf.eye(target_dim, dtype=tf.float32) * 1.0 / args.rho_var
    else:
        rho_precision_mat = tf.linalg.diag(1.0 / tf.constant(args.rho_var, dtype=tf.float32))

    # rho_precision_mat = tf.eye(target_dim, dtype=tf.float32) * 1.0 / args.rho_var
    theta_tmp = theta + 0.5 * h * tf.squeeze(tf.matmul(rho_precision_mat, tf.expand_dims(rho, axis=-1)))
    u_tmp = u * math.cos(0.5 * h) + p * math.sin(0.5 * h)
    p_tmp = p * math.cos(0.5 * h) - u * math.sin(0.5 * h)

    coords_tmp = tf.concat([theta_tmp, rho, u_tmp, p_tmp], axis=0)
    dcoords = derivs_func(coords_tmp, func, target_dim, aux_dim)
    
    p_tmp = p_tmp + h * (dcoords[-aux_dim:] + u_tmp)
    rho_new = rho + h * dcoords[target_dim:2*target_dim]
    theta_new = theta_tmp + 0.5 * h * tf.squeeze(tf.matmul(rho_precision_mat, tf.expand_dims(rho_new, axis=-1)))
    u_new = u_tmp * math.cos(0.5 * h) + p_tmp * math.sin(0.5 * h)
    p_new = p_tmp * math.cos(0.5 * h) - u_tmp * math.sin(0.5 * h)

    coords_new = tf.concat([theta_new, rho_new, u_new, p_new], axis=0)
    return coords_new

def test_one_step():
    '''
    test for _one_step
    '''
    args = mock_args()
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target = tf.constant(100, dtype=tf.float32)
    [next_momentum_parts, next_state_parts, next_target] = _one_step(mock_target_func, [0.01], momentum_parts, 
                                                                      state_parts, target, args.rho_precision_mat, args.target_dim)
    new_coords = tf.concat([next_state_parts[0][:args.target_dim], next_momentum_parts[0][:args.target_dim],
                        next_state_parts[0][args.target_dim:], next_momentum_parts[0][args.target_dim:]], axis=0)

    coords = tf.concat([state_parts[0][:args.target_dim], momentum_parts[0][:args.target_dim],
                        state_parts[0][args.target_dim:], momentum_parts[0][args.target_dim:]], axis=0)
    new_coords_expected = integrator_one_step_mass(coords, mock_H_func, numerical_grad, 0.01, args.target_dim, args.aux_dim, args)
    next_target_expected = mock_target_func(tf.concat([new_coords_expected[:args.target_dim], 
                                                      new_coords_expected[2*args.target_dim: 2*args.target_dim+args.aux_dim]], axis=0))
    print('new_coords', new_coords)
    print('new_coords_expected', new_coords_expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(new_coords, new_coords_expected))
    assert tf.experimental.numpy.allclose(next_target, next_target_expected)

def test_process_args():
    'test process_args function'
    # Mock target function that returns a value and its gradient
    def mock_target_fn(state_parts):
        return tf.reduce_sum(state_parts)

    # Test case 1: target is None
    momentum_parts = [tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)]
    state_parts = [tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)]
    target = None

    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that momentum_parts and state_parts are converted to tensors
    assert all(isinstance(v, tf.Tensor) for v in momentum_parts_tensor)
    assert all(isinstance(v, tf.Tensor) for v in state_parts_tensor)

    # Check that the target is computed correctly
    assert isinstance(target_tensor, tf.Tensor)
    assert target_tensor.numpy() == pytest.approx(15.0)  # 4.0 + 5.0 + 6.0 = 15.0

    # Test case 2: target is provided
    target = 10.0
    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that the provided target is converted to a tensor
    assert isinstance(target_tensor, tf.Tensor)
    assert target_tensor.numpy() == pytest.approx(10.0)

    # Test case 3: Check dtype_hint
    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that the tensors have the correct dtype
    assert all(v.dtype == tf.float32 for v in momentum_parts_tensor)
    assert all(v.dtype == tf.float32 for v in state_parts_tensor)
    assert target_tensor.dtype == tf.float32

def test_StrangIntegrator():
    '''
    test for StrangIntegrator
    '''
    args = mock_args()
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target = tf.constant(100, dtype=tf.float32)

    integrator = StrangIntegrator(mock_target_func, [0.01], 1, args.target_dim, args.aux_dim, args.rho_precision_mat)
    
    [next_momentum_parts, next_state_parts, next_target] = integrator(momentum_parts, state_parts, target)
    new_coords = tf.concat([next_state_parts[0][:args.target_dim], next_momentum_parts[0][:args.target_dim],
                        next_state_parts[0][args.target_dim:], next_momentum_parts[0][args.target_dim:]], axis=0)

    coords = tf.concat([state_parts[0][:args.target_dim], momentum_parts[0][:args.target_dim],
                        state_parts[0][args.target_dim:], momentum_parts[0][args.target_dim:]], axis=0)
    new_coords_expected = integrator_one_step_mass(coords, mock_H_func, numerical_grad, 0.01, args.target_dim, args.aux_dim, args)
    next_target_expected = mock_target_func(tf.concat([new_coords_expected[:args.target_dim], 
                                                      new_coords_expected[2*args.target_dim: 2*args.target_dim+args.aux_dim]], axis=0))
    print('new_coords', new_coords)
    print('new_coords_expected', new_coords_expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(new_coords, new_coords_expected))
    assert tf.experimental.numpy.allclose(next_target, next_target_expected)

class mock_hnn(tf.Module):
    def __init__(self, args, ham_func=None, assume_canonical_coords=True, name=None, **kwargs):
        super().__init__(name)
        self.args = args
        self.ham_func = ham_func
        
    def time_derivative(self, x):
        conservative_field = tf.zeros_like(x) # start out with both components set to 0
        solenoidal_field = tf.zeros_like(x)

        grads = tfp.math.value_and_gradient(self.ham_func, tf.squeeze(x, axis=0))[-1]
    
        dF2 = tf.expand_dims(grads, axis=0) # gradients for solenoidal field
        dtheta, drho, du, dp = tf.split(dF2, num_or_size_splits=[self.args.target_dim, self.args.target_dim, 
                                                                    self.args.aux_dim, self.args.aux_dim], axis=-1)
   
        solenoidal_field = tf.concat([drho, -dtheta, dp, -du], axis=-1)

        return conservative_field + solenoidal_field

def test__hnn_value_and_gradients():
    '''
    test to see whether _hnn_value_and_gradients can yield the same results
    as mcmc_util._value_and_gradients
    '''
    # hamiltonian function
    args = mock_args()
    fn_arg_list = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    hnn_model = mock_hnn(args, ham_func=mock_H_func)

    results, grads_list = _hnn_value_and_gradients(mock_target_func, fn_arg_list, momentum_parts, hnn_model, target_dim=args.target_dim, aux_dim=args.aux_dim)
    results_expected, grads_list_expected = mcmc_util._value_and_gradients(mock_target_func, fn_arg_list)

    print(results, results_expected)
    print(grads_list, grads_list_expected)
    assert tf.reduce_all(tf.equal(results, results_expected))
    for x, y in zip(grads_list, grads_list_expected):
        assert tf.reduce_all(tf.equal(x, y))

def test_hnn_fn_and_grads():
    '''
    test to see whether hnn_fn_and_grads can yield the same results
    as mcmc_util.maybe_call_fn_and_grads
    '''
    # hamiltonian function
    args = mock_args()
    fn_arg_list = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    hnn_model = mock_hnn(args, ham_func=mock_H_func)

    results, grads_list = hnn_fn_and_grads(mock_target_func, fn_arg_list, momentum_parts, hnn_model, target_dim=args.target_dim, aux_dim=args.aux_dim)
    results_expected, grads_list_expected = mcmc_util.maybe_call_fn_and_grads(mock_target_func, fn_arg_list)
    print(results, results_expected)
    print(grads_list, grads_list_expected)
    assert tf.reduce_all(tf.equal(results, results_expected))
    for x, y in zip(grads_list, grads_list_expected):
        assert tf.reduce_all(tf.equal(x, y))

def test_one_step_hnn():
    '''
    test for _one_step_hnn
    '''
    args = mock_args()
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target = tf.constant(100, dtype=tf.float32)
    hnn_model = mock_hnn(args, ham_func=mock_H_func)
    [next_momentum_parts, next_state_parts, next_target] = _one_step_hnn(mock_target_func, [0.01], momentum_parts, 
                                                                      state_parts, target, args.rho_precision_mat, args.target_dim,
                                                                      args.aux_dim, hnn_model)
    new_coords = tf.concat([next_state_parts[0][:args.target_dim], next_momentum_parts[0][:args.target_dim],
                        next_state_parts[0][args.target_dim:], next_momentum_parts[0][args.target_dim:]], axis=0)

    coords = tf.concat([state_parts[0][:args.target_dim], momentum_parts[0][:args.target_dim],
                        state_parts[0][args.target_dim:], momentum_parts[0][args.target_dim:]], axis=0)
    new_coords_expected = integrator_one_step_mass(coords, mock_H_func, numerical_grad, 0.01, args.target_dim, args.aux_dim, args)
    next_target_expected = mock_target_func(tf.concat([new_coords_expected[:args.target_dim], 
                                                      new_coords_expected[2*args.target_dim: 2*args.target_dim+args.aux_dim]], axis=0))
    print('new_coords', new_coords)
    print('new_coords_expected', new_coords_expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(new_coords, new_coords_expected))
    assert tf.experimental.numpy.allclose(next_target, next_target_expected)

@pytest.fixture
def mock_hnn_and_grads():
    def _mock_hnn_and_grads(target_fn, state_parts, *args, **kwargs):
        return tfp.math.value_and_gradient(target_fn, *state_parts)
    return _mock_hnn_and_grads

def test_process_args_hnn(mocker, mock_hnn_and_grads):
    'test process_args_hnn function'
    mocker.patch('pseudo_marginal.strang_integrator_hnn.hnn_fn_and_grads', mock_hnn_and_grads)
    # Mock target function that returns a value and its gradient
    def mock_target_fn(state_parts):
        return tf.reduce_sum(state_parts)

    # Test case 1: target is None
    momentum_parts = [tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)]
    state_parts = [tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)]
    target = None

    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args_hnn(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that momentum_parts and state_parts are converted to tensors
    assert all(isinstance(v, tf.Tensor) for v in momentum_parts_tensor)
    assert all(isinstance(v, tf.Tensor) for v in state_parts_tensor)

    # Check that the target is computed correctly
    assert isinstance(target_tensor, tf.Tensor)
    assert target_tensor.numpy() == pytest.approx(15.0)  # 4.0 + 5.0 + 6.0 = 15.0

    # Test case 2: target is provided
    target = 10.0
    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that the provided target is converted to a tensor
    assert isinstance(target_tensor, tf.Tensor)
    assert target_tensor.numpy() == pytest.approx(10.0)

    # Test case 3: Check dtype_hint
    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that the tensors have the correct dtype
    assert all(v.dtype == tf.float32 for v in momentum_parts_tensor)
    assert all(v.dtype == tf.float32 for v in state_parts_tensor)
    assert target_tensor.dtype == tf.float32

def test_StrangIntegrator_HNN():
    '''
    test for StrangIntegrator
    '''
    args = mock_args()
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target = tf.constant(100, dtype=tf.float32)
    hnn_model = mock_hnn(args, ham_func=mock_H_func)
    integrator = StrangIntegrator_HNN(mock_target_func, [0.01], 1, args.target_dim, args.aux_dim, args.rho_precision_mat, hnn_model)
    
    [next_momentum_parts, next_state_parts, next_target] = integrator(momentum_parts, state_parts, target)
    new_coords = tf.concat([next_state_parts[0][:args.target_dim], next_momentum_parts[0][:args.target_dim],
                        next_state_parts[0][args.target_dim:], next_momentum_parts[0][args.target_dim:]], axis=0)

    coords = tf.concat([state_parts[0][:args.target_dim], momentum_parts[0][:args.target_dim],
                        state_parts[0][args.target_dim:], momentum_parts[0][args.target_dim:]], axis=0)
    new_coords_expected = integrator_one_step_mass(coords, mock_H_func, numerical_grad, 0.01, args.target_dim, args.aux_dim, args)
    next_target_expected = mock_target_func(tf.concat([new_coords_expected[:args.target_dim], 
                                                      new_coords_expected[2*args.target_dim: 2*args.target_dim+args.aux_dim]], axis=0))
    print('new_coords', new_coords)
    print('new_coords_expected', new_coords_expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(new_coords, new_coords_expected))
    assert tf.experimental.numpy.allclose(next_target, next_target_expected)

def mock_grad_total(coords):
    grads = tfp.math.value_and_gradient(mock_H_func, coords)[-1]
    return grads

def test_one_step_grad():
    '''
    test for _one_step_grad
    '''
    args = mock_args()
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target = tf.constant(100, dtype=tf.float32)
    [next_momentum_parts, next_state_parts, next_target] = _one_step_grad(mock_target_func, [0.01], momentum_parts, 
                                                                      state_parts, target, args.rho_precision_mat, args.target_dim, args.aux_dim,
                                                                      grad_func=mock_grad_total)
    new_coords = tf.concat([next_state_parts[0][:args.target_dim], next_momentum_parts[0][:args.target_dim],
                        next_state_parts[0][args.target_dim:], next_momentum_parts[0][args.target_dim:]], axis=0)

    coords = tf.concat([state_parts[0][:args.target_dim], momentum_parts[0][:args.target_dim],
                        state_parts[0][args.target_dim:], momentum_parts[0][args.target_dim:]], axis=0)
    new_coords_expected = integrator_one_step_mass(coords, mock_H_func, numerical_grad, 0.01, args.target_dim, args.aux_dim, args)
    next_target_expected = mock_target_func(tf.concat([new_coords_expected[:args.target_dim], 
                                                      new_coords_expected[2*args.target_dim: 2*args.target_dim+args.aux_dim]], axis=0))
    print('new_coords', new_coords)
    print('new_coords_expected', new_coords_expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(new_coords, new_coords_expected))
    assert tf.experimental.numpy.allclose(next_target, next_target_expected)

def test__manual_value_and_gradients():
    '''
    test to see whether _manual_value_and_gradients can yield the same results
    as mcmc_util._value_and_gradients
    '''
    # hamiltonian function
    args = mock_args()
    fn_arg_list = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]

    results, grads_list = _manual_value_and_gradients(mock_target_func, fn_arg_list, momentum_parts, target_dim=args.target_dim, aux_dim=args.aux_dim, grad_func=mock_grad_total)
    results_expected, grads_list_expected = mcmc_util._value_and_gradients(mock_target_func, fn_arg_list)

    print(results, results_expected)
    print(grads_list, grads_list_expected)
    assert tf.reduce_all(tf.equal(results, results_expected))
    for x, y in zip(grads_list, grads_list_expected):
        assert tf.reduce_all(tf.equal(x, y))

def test_manual_fn_and_grads():
    '''
    test to see whether manual_fn_and_grads can yield the same results
    as mcmc_util.maybe_call_fn_and_grads
    '''
    # hamiltonian function
    args = mock_args()
    fn_arg_list = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    
    results, grads_list = manual_fn_and_grads(mock_target_func, fn_arg_list, momentum_parts, target_dim=args.target_dim, aux_dim=args.aux_dim, grad_func=mock_grad_total)
    results_expected, grads_list_expected = mcmc_util.maybe_call_fn_and_grads(mock_target_func, fn_arg_list)
    print(results, results_expected)
    print(grads_list, grads_list_expected)
    assert tf.reduce_all(tf.equal(results, results_expected))
    for x, y in zip(grads_list, grads_list_expected):
        assert tf.reduce_all(tf.equal(x, y))

class mock_calculate_grad_class:
    def grad_total(coords):
        return mock_grad_total(coords)

@pytest.fixture
def mock_calculate_grad():
    def _mock_calculate_grad(args):
        return mock_calculate_grad_class
    return _mock_calculate_grad

def test_StrangIntegrator_grad(mocker, mock_calculate_grad):
    '''
    test for StrangIntegrator_grad
    '''
    mocker.patch('pseudo_marginal.strang_integrator_hnn.calculate_grad', mock_calculate_grad)
    args = mock_args()
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target = tf.constant(100, dtype=tf.float32)

    integrator = StrangIntegrator_grad(mock_target_func, [0.01], 1, args.target_dim, args.aux_dim, args.rho_precision_mat, args)
    
    [next_momentum_parts, next_state_parts, next_target] = integrator(momentum_parts, state_parts, target)
    new_coords = tf.concat([next_state_parts[0][:args.target_dim], next_momentum_parts[0][:args.target_dim],
                        next_state_parts[0][args.target_dim:], next_momentum_parts[0][args.target_dim:]], axis=0)

    coords = tf.concat([state_parts[0][:args.target_dim], momentum_parts[0][:args.target_dim],
                        state_parts[0][args.target_dim:], momentum_parts[0][args.target_dim:]], axis=0)
    new_coords_expected = integrator_one_step_mass(coords, mock_H_func, numerical_grad, 0.01, args.target_dim, args.aux_dim, args)
    next_target_expected = mock_target_func(tf.concat([new_coords_expected[:args.target_dim], 
                                                      new_coords_expected[2*args.target_dim: 2*args.target_dim+args.aux_dim]], axis=0))
    print('new_coords', new_coords)
    print('new_coords_expected', new_coords_expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(new_coords, new_coords_expected))
    assert tf.experimental.numpy.allclose(next_target, next_target_expected)

def test_process_args_grad(mocker, mock_hnn_and_grads):
    'test process_args_grad function'
    mocker.patch('pseudo_marginal.strang_integrator_hnn.manual_fn_and_grads', mock_hnn_and_grads)
    # Mock target function that returns a value and its gradient
    def mock_target_fn(state_parts):
        return tf.reduce_sum(state_parts)

    # Test case 1: target is None
    momentum_parts = [tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)]
    state_parts = [tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)]
    target = None

    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args_grad(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that momentum_parts and state_parts are converted to tensors
    assert all(isinstance(v, tf.Tensor) for v in momentum_parts_tensor)
    assert all(isinstance(v, tf.Tensor) for v in state_parts_tensor)

    # Check that the target is computed correctly
    assert isinstance(target_tensor, tf.Tensor)
    assert target_tensor.numpy() == pytest.approx(15.0)  # 4.0 + 5.0 + 6.0 = 15.0

    # Test case 2: target is provided
    target = 10.0
    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that the provided target is converted to a tensor
    assert isinstance(target_tensor, tf.Tensor)
    assert target_tensor.numpy() == pytest.approx(10.0)

    # Test case 3: Check dtype_hint
    momentum_parts_tensor, state_parts_tensor, target_tensor = process_args(
        mock_target_fn, momentum_parts, state_parts, target
    )

    # Check that the tensors have the correct dtype
    assert all(v.dtype == tf.float32 for v in momentum_parts_tensor)
    assert all(v.dtype == tf.float32 for v in state_parts_tensor)
    assert target_tensor.dtype == tf.float32
