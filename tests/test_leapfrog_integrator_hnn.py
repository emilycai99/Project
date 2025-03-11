import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from tf_version.functions_tf import dist_func
from tf_version.leapfrog_integrator_hnn import _hnn_value_and_gradients, hnn_fn_and_grads, \
    process_args,_one_step, SimpleLeapfrogIntegrator_HNN
import tensorflow_probability.python.mcmc.internal.util as mcmc_util
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl

class mock_hnn(tf.Module):
    def __init__(self, input_dim, *args, ham_func=None, assume_canonical_coords=True, name=None, **kwargs):
        super().__init__(name)
        self.input_dim = input_dim
        self.ham_func = ham_func
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        
    
    def time_derivative(self, x):
        conservative_field = tf.zeros_like(x) # start out with both components set to 0
        solenoidal_field = tf.zeros_like(x)

        grads = tfp.math.value_and_gradient(self.ham_func, tf.squeeze(x, axis=0))[-1]
    
        dF2 = tf.expand_dims(grads, axis=0) # gradients for solenoidal field
        solenoidal_field = tf.matmul(dF2, tf.transpose(self.M))

        return conservative_field + solenoidal_field
    
    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = tf.eye(n)
            M = tf.concat([M[n//2:], -M[:n//2]], axis=0)
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = tf.ones(shape=(n,n)) # matrix of ones
            M *= 1 - tf.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1

            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M

class mock_args():
    def __init__(self):
        self.input_dim = 4
        self.dist_name = '2D_Gauss_mix'

def test__hnn_value_and_gradients():
    '''
    test to see whether _hnn_value_and_gradients can yield the same results
    as mcmc_util._value_and_gradients
    '''
    # hamiltonian function
    args = mock_args()
    dist_func_obj = dist_func(args)
    ham_func = dist_func_obj.get_Hamiltonian
    target_func = dist_func_obj.get_target_log_prob_func
    fn_arg_list = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    hnn_model = mock_hnn(input_dim=args.input_dim, ham_func=ham_func)

    results, grads_list = _hnn_value_and_gradients(target_func, fn_arg_list, momentum_parts, hnn_model, input_dim=args.input_dim)
    results_expected, grads_list_expected = mcmc_util._value_and_gradients(target_func, fn_arg_list)

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
    dist_func_obj = dist_func(args)
    ham_func = dist_func_obj.get_Hamiltonian
    target_func = dist_func_obj.get_target_log_prob_func
    fn_arg_list = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]

    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    hnn_model = mock_hnn(input_dim=args.input_dim, ham_func=ham_func)

    results, grads_list = hnn_fn_and_grads(target_func, fn_arg_list, momentum_parts, hnn_model, input_dim=args.input_dim)
    results_expected, grads_list_expected = mcmc_util.maybe_call_fn_and_grads(target_func, fn_arg_list)
    print(results, results_expected)
    print(grads_list, grads_list_expected)
    assert tf.reduce_all(tf.equal(results, results_expected))
    for x, y in zip(grads_list, grads_list_expected):
        assert tf.reduce_all(tf.equal(x, y))

def test_process_args():
    '''
    test to see whether process_args can yield the same results
    as leapfrog_impl.process_args
    '''
    # hamiltonian function
    args = mock_args()
    dist_func_obj = dist_func(args)
    ham_func = dist_func_obj.get_Hamiltonian
    target_func = dist_func_obj.get_target_log_prob_func

    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    hnn_model = mock_hnn(input_dim=args.input_dim, ham_func=ham_func)

    momentum_parts_get, state_parts_get, target_get, target_grad_parts_get = \
        process_args(target_func, momentum_parts, state_parts, hnn_model=hnn_model, input_dim=args.input_dim)
    momentum_parts_expected, state_parts_expected, target_expected, target_grad_parts_expected = \
        leapfrog_impl.process_args(target_func, momentum_parts, state_parts)
    
    for x, y in zip(momentum_parts_get, momentum_parts_expected):
        assert tf.reduce_all(tf.equal(x, y))
    for x, y in zip(state_parts_get, state_parts_expected):
        assert tf.reduce_all(tf.equal(x, y))
    assert tf.reduce_all(tf.equal(target_get, target_expected))
    for x, y in zip(target_grad_parts_get, target_grad_parts_get):
        assert tf.reduce_all(tf.equal(x, y))

def test__one_step():
    '''
    test to see whether _one_step can yield the same results
    as leapfrog_impl._one_step
    '''
    # hamiltonian function
    args = mock_args()
    dist_func_obj = dist_func(args)
    ham_func = dist_func_obj.get_Hamiltonian
    target_func = dist_func_obj.get_target_log_prob_func
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target = tf.constant(100, dtype=tf.float32)
    target_grad_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    hnn_model = mock_hnn(input_dim=args.input_dim, ham_func=ham_func)

    results = _one_step(target_func, [0.01], lambda x: x, momentum_parts, state_parts, target,
                        target_grad_parts, hnn_model=hnn_model, input_dim=args.input_dim)
    results_expected = leapfrog_impl._one_step(target_func, [0.01], lambda x: x, momentum_parts, state_parts, target,
                            target_grad_parts)
    for i in range(len(results)):
        if i == 2:
            assert tf.reduce_all(tf.equal(results[i], results_expected[i]))
        else:
            for x, y in zip(results[i], results_expected[i]):
                assert tf.reduce_all(tf.equal(x, y))
    
def test_SimpleLeapfrogIntegrator_HNN():
    '''
    test to see whether SimpleLeapfrogIntegrator_HNN can yield the same results
    as leapfrog_impl.SimpleLeapfrogIntegrator
    '''

    # hamiltonian function
    args = mock_args()
    dist_func_obj = dist_func(args)
    ham_func = dist_func_obj.get_Hamiltonian
    target_func = dist_func_obj.get_target_log_prob_func
    hnn_model = mock_hnn(input_dim=args.input_dim, ham_func=ham_func)

    leapfrog_hnn = SimpleLeapfrogIntegrator_HNN(target_func, [0.01], 1, hnn_model=hnn_model, input_dim=args.input_dim)
    leapfrog = leapfrog_impl.SimpleLeapfrogIntegrator(target_func, [0.01], 1)

    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]

    results = leapfrog_hnn(momentum_parts, state_parts)
    results_expected = leapfrog(momentum_parts, state_parts)
    for i in range(len(results)):
        if i == 2:
            assert tf.reduce_all(tf.equal(results[i], results_expected[i]))
        else:
            for x, y in zip(results[i], results_expected[i]):
                assert tf.reduce_all(tf.equal(x, y))







    



