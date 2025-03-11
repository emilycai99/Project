import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

import pseudo_marginal.pm_hmc as pm_hmc
import tensorflow_probability.python.mcmc.hmc as hmc_base

"""
This test file only tests the modifications in pm_hmc compared to hmc_base
"""


def test_compute_log_acceptance_correction():
    '''
    test whether the compute_log_acceptance_criterion function outputs the same as the original version
        when the precision mat is identity
    '''
    target_dim = 2
    aux_dim = 3

    # first test the case where the precision mat is identity
    precision_mat = tf.linalg.diag(tf.constant([1.0, 1.0], dtype=tf.float32))

    current_momentums = [tf.random.normal(shape=[target_dim+aux_dim])]
    proposed_momentums = [tf.random.normal(shape=[target_dim+aux_dim])]

    results = pm_hmc._compute_log_acceptance_correction(
        current_momentums, proposed_momentums, 0,
        target_dim=target_dim, aux_dim=aux_dim, precision_mat=precision_mat, 
    )
    results_expected = hmc_base._compute_log_acceptance_correction(
        current_momentums, proposed_momentums, 0
    )   

    assert tf.experimental.numpy.allclose(results, results_expected)
    
    # next test the case where the precision mat is non-identity
    precision_mat = tf.linalg.diag(tf.constant([2.0, 3.0], dtype=tf.float32))
    current_momentums = [tf.concat([tf.ones(shape=[target_dim], dtype=tf.float32),
                                    tf.ones(shape=[aux_dim], dtype=tf.float32) * 2.0], axis=-1)]
    proposed_momentums = [tf.concat([tf.ones(shape=[target_dim], dtype=tf.float32) * 2.0,
                                    tf.ones(shape=[aux_dim], dtype=tf.float32)], axis=-1)]
    results = pm_hmc._compute_log_acceptance_correction(
        current_momentums, proposed_momentums, 0,
        target_dim=target_dim, aux_dim=aux_dim, precision_mat=precision_mat, 
    )
    results_expected = tf.constant(-3.0, dtype=tf.float32)
    assert tf.experimental.numpy.allclose(results, results_expected)

def func(x):
    return tf.tensordot(x, x, axes=1)

class mock_args:
    def __init__(self, rho_var=1.0):
        self.input_dim = 10
        self.target_dim = 3
        self.aux_dim = 2
        self.rho_var = rho_var
        self.rho_precision_mat = tf.eye(self.target_dim, dtype=tf.float32) * 1.0 / self.rho_var


def log_prior(theta):
    return tf.tensordot(theta, theta, axes=1)

def log_phat(theta, u):
    return tf.tensordot(theta, theta, axes=1) + 2.0 * tf.tensordot(u, u, axes=1)

args = mock_args()
def mock_H_func(coords, args):
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

def test_prepare_args():
    '''
    test whether the _prepare_args function outputs the same as the original version
    '''
    args = mock_args()
    state = tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)
    step_size = [0.01]
    results_state, results_step_size, results_target_prob = pm_hmc._prepare_args(mock_target_func, state, step_size)
    expected_state, expected_step_size, expected_target_prob, _ = hmc_base._prepare_args(mock_target_func, state, step_size)

    for x, y in zip(results_state, expected_state):
        assert tf.equal(x, y)
    
    assert tf.equal(results_step_size, expected_step_size)
    assert tf.equal(results_target_prob, expected_target_prob)

def test_bootstrap_results():
    '''
    test whether the _bootstrap_results function outputs the same as the original version
    '''
    args = mock_args(1.0)
    hmc_obj = pm_hmc.UncalibratedPseudoMarginalHamiltonianMonteCarlo(mock_target_func, 0.01, 1, pm_args=args)
    hmc_base_obj = hmc_base.UncalibratedHamiltonianMonteCarlo(mock_target_func, 0.01, 1)
    init_state = tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)
    
    results = hmc_obj.bootstrap_results(init_state)
    expected = hmc_base_obj.bootstrap_results(init_state)

    assert tf.equal(results.log_acceptance_correction, expected.log_acceptance_correction)
    assert tf.equal(results.target_log_prob, expected.target_log_prob)
    for x, y in zip(results.initial_momentum, expected.initial_momentum):
        assert tf.reduce_all(tf.equal(x, y))
    for x, y in zip(results.final_momentum, expected.final_momentum):
        assert tf.reduce_all(tf.equal(x, y))
    assert results.step_size == expected.step_size
    assert results.num_leapfrog_steps == expected.num_leapfrog_steps

class mock_StrangIntegrator():
    def __init__(self, target_fn, step_sizes, num_steps, target_dim, aux_dim, target_momentum_precision_mat):
        self.integrator = pm_hmc.StrangIntegrator(target_fn, step_sizes, num_steps, target_dim, aux_dim, target_momentum_precision_mat)
    
    def __call__(self, momentum_parts,
               state_parts,
               target, *args, **kwds):
        a, b, c = self.integrator(momentum_parts, state_parts, target)
        return a, b, c, tf.ones_like(c)

@pytest.fixture
def mock_SimpleLeapfrogIntegrator():
    def _mock_SimpleLeapfrogIntegrator(target_log_prob_fn, step_sizes, num_leapfrog_steps):
        args = mock_args()
        return mock_StrangIntegrator(target_log_prob_fn, step_sizes, num_leapfrog_steps,
                                       target_dim=args.target_dim, aux_dim=args.aux_dim,
                                       target_momentum_precision_mat=args.rho_precision_mat)
    return _mock_SimpleLeapfrogIntegrator

@pytest.fixture
def mock_normal():
    def _mock_normal(shape, dtype, *args, **kwargs):
        return tf.ones(shape=shape, dtype=dtype)
    return _mock_normal


def test_one_step(mocker, mock_SimpleLeapfrogIntegrator, mock_normal):
    '''
    test whether the UncalibratedPseudoMarginalHamiltonianMonteCarlo.one_step
      function outputs the same as the original version
    '''
    args = mock_args(1.0)
    hmc_obj = pm_hmc.UncalibratedPseudoMarginalHamiltonianMonteCarlo(mock_target_func, 0.01, 1, pm_args=args)
    hmc_base_obj = hmc_base.UncalibratedHamiltonianMonteCarlo(mock_target_func, 0.01, 1)
    init_state = tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)
    current_state = tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)

    mocker.patch('tensorflow_probability.python.mcmc.hmc.leapfrog_impl.SimpleLeapfrogIntegrator', mock_SimpleLeapfrogIntegrator)
    mocker.patch('tensorflow_probability.python.mcmc.hmc.samplers.normal', mock_normal)
    mocker.patch('pseudo_marginal.pm_hmc.samplers.normal', mock_normal)

    results_state, results = hmc_obj.one_step(current_state, hmc_obj.bootstrap_results(init_state))
    expected_state, expected = hmc_base_obj.one_step(current_state, hmc_base_obj.bootstrap_results(init_state))

    assert tf.reduce_all(tf.equal(results_state, expected_state))
    assert tf.experimental.numpy.allclose(results.log_acceptance_correction, expected.log_acceptance_correction)
    assert tf.equal(results.target_log_prob, expected.target_log_prob)
    for x, y in zip(results.initial_momentum, expected.initial_momentum):
        assert tf.reduce_all(tf.equal(x, y))
    for x, y in zip(results.final_momentum, expected.final_momentum):
        assert tf.reduce_all(tf.equal(x, y))
    assert results.step_size == expected.step_size
    assert results.num_leapfrog_steps == expected.num_leapfrog_steps

@pytest.fixture
def mock_uniform():
    def _mock_uniform(shape, dtype, *args, **kwargs):
        return tf.ones(shape=shape, dtype=dtype) * 0.5
    return _mock_uniform

def test_PseudoMarginalHamiltonianMonteCarlo_bootstrap_results(mocker, mock_uniform):
    '''
    test whether the _bootstrap_results function outputs the same as the original version
    '''
    mocker.patch('tensorflow_probability.python.mcmc.metropolis_hastings.samplers.uniform', mock_uniform)
    args = mock_args(1.0)
    hmc_obj = pm_hmc.PseudoMarginalHamiltonianMonteCarlo(mock_target_func, 0.01, 1, pm_args=args)
    hmc_base_obj = hmc_base.HamiltonianMonteCarlo(mock_target_func, 0.01, 1)
    init_state = tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)
    
    results = hmc_obj.bootstrap_results(init_state)
    expected = hmc_base_obj.bootstrap_results(init_state)
    
    assert tf.equal(results.is_accepted, expected.is_accepted)
    assert tf.equal(results.log_accept_ratio, expected.log_accept_ratio)
    assert tf.reduce_all(tf.equal(results.proposed_state, expected.proposed_state))

def test_PseudoMarginalHamiltonianMonteCarlo_one_step(mocker, mock_SimpleLeapfrogIntegrator, mock_normal, mock_uniform):
    '''
    test whether the PseudoMarginalHamiltonianMonteCarlo.one_step
      function outputs the same as the original version
    '''
    args = mock_args(1.0)
    hmc_obj = pm_hmc.PseudoMarginalHamiltonianMonteCarlo(mock_target_func, 0.01, 1, pm_args=args)
    hmc_base_obj = hmc_base.HamiltonianMonteCarlo(mock_target_func, 0.01, 1)
    init_state = tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)
    current_state = tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)

    mocker.patch('tensorflow_probability.python.mcmc.metropolis_hastings.samplers.uniform', mock_uniform)
    mocker.patch('tensorflow_probability.python.mcmc.hmc.leapfrog_impl.SimpleLeapfrogIntegrator', mock_SimpleLeapfrogIntegrator)
    mocker.patch('tensorflow_probability.python.mcmc.hmc.samplers.normal', mock_normal)
    mocker.patch('pseudo_marginal.pm_hmc.samplers.normal', mock_normal)

    results_state, results = hmc_obj.one_step(current_state, hmc_obj.bootstrap_results(init_state))
    expected_state, expected = hmc_base_obj.one_step(current_state, hmc_base_obj.bootstrap_results(init_state))

    assert tf.reduce_all(tf.equal(results_state, expected_state))
    assert tf.equal(results.is_accepted, expected.is_accepted)
    assert tf.experimental.numpy.allclose(results.log_accept_ratio, expected.log_accept_ratio)
    assert tf.reduce_all(tf.equal(results.proposed_state, expected.proposed_state))




