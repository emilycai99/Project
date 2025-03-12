import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import random

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.pm_nuts_hnn import compute_hamiltonian
from tensorflow_probability.python.mcmc.nuts import compute_hamiltonian as compute_ham_base

import pseudo_marginal.pm_nuts_hnn as nuts_hnn
import tensorflow_probability.python.mcmc.nuts as nuts
nuts.GENERALIZED_UTURN = False
nuts.MULTINOMIAL_SAMPLE = False


def func(x):
    return tf.tensordot(x, x, axes=1)

class mock_args:
    def __init__(self, rho_var=1.0):
        self.save_dir = '/tmp'
        self.dist_name = 'gauss_mix'
        self.T = 500
        self.n = 6
        self.p = 8
        self.N = 128
        self.num_samples = 2
        self.len_sample = 50
        self.step_size = 0.1
        self.learn_rate = 0.01
        self.num_hmc_samples = 100
        self.nn_model_name = 'mlp'
        self.epsilon = 0.1
        self.rho_var = rho_var
        self.grad_flag = True
        self.target_dim = self.p + 5
        self.aux_dim = self.T * self.N
        self.num_burnin_samples = 1
        self.print_every = 1
        self.input_dim = 2 * (self.target_dim + self.aux_dim)
        self.data_pth = 'pseudo_marginal/data'
        if isinstance(self.rho_var, float):
            self.rho_precision_mat = tf.eye(self.target_dim, dtype=tf.float32) * 1.0 / self.rho_var
        else:
            self.rho_precision_mat = tf.linalg.diag(1.0 / tf.constant(self.rho_var, dtype=tf.float32))
        self.hidden_dim = 100
        self.num_layers = 3
        self.nonlinearity = 'sine'
        self.nn_out_dim = 26
        self.hnn_threshold = 10
    
    def __getattr__(self, name):
        return None


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

def test_compute_hamiltonian():
    '''
    test by
    - compare with the original implementation of tfp when rho_var = 1.0
    - compare with the self-written hamiltonian function when rho_var != 1.0
    '''
    args = mock_args(rho_var=1.0)
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target_log_prob = mock_target_func(*state_parts)
    result = compute_hamiltonian(target_log_prob, momentum_parts, args.rho_precision_mat, 
                                 target_dim=args.target_dim, aux_dim=args.aux_dim)
    result_expected = compute_ham_base(target_log_prob, momentum_parts)

    print('result', result)
    print('result_expected', result_expected)
    assert tf.experimental.numpy.allclose(result, result_expected)

    args = mock_args(rho_var=[random.gauss(0.0, 1.0) for _ in range(args.target_dim)])
    momentum_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target_log_prob = mock_target_func(*state_parts)
    result = compute_hamiltonian(target_log_prob, momentum_parts, args.rho_precision_mat, 
                                 target_dim=args.target_dim, aux_dim=args.aux_dim)
    coords = tf.concat([state_parts[0][:args.target_dim], momentum_parts[0][:args.target_dim],
                        state_parts[0][args.target_dim:], momentum_parts[0][args.target_dim:]], axis=0)
    result_expected = -1.0 * mock_H_func(coords, args)

    print('result', result)
    print('result_expected', result_expected)
    assert tf.experimental.numpy.allclose(result, result_expected)

class mock_hnn(tf.Module):
    def __init__(self, model_args, *args, ham_func=None, assume_canonical_coords=True, name=None, **kwargs):
        super().__init__(name)
        self.input_dim = (model_args.target_dim + model_args.aux_dim) * 2
        self.ham_func = ham_func
        self.assume_canonical_coords = assume_canonical_coords
        self.args = model_args
        
    def __call__(self, x):
        return x
    
    def load_weights(self, x):
        pass

    def time_derivative(self, x):
        conservative_field = tf.zeros_like(x) # start out with both components set to 0
        solenoidal_field = tf.zeros_like(x)

        grads = tfp.math.value_and_gradient(self.ham_func, tf.squeeze(x, axis=0))[-1]
    
        dF2 = tf.expand_dims(grads, axis=0) # gradients for solenoidal field
        dtheta, drho, du, dp = tf.split(dF2, num_or_size_splits=[self.args.target_dim, self.args.target_dim, 
                                                                     self.args.aux_dim, self.args.aux_dim], axis=-1)
        solenoidal_field = tf.concat([drho, -dtheta, dp, -du], axis=-1)

        return conservative_field + solenoidal_field
    

def mock_Ham(coords):
    rho_precision_mat = args.rho_precision_mat
    theta, rho, u, p = tf.split(coords, [args.target_dim, args.target_dim,
                                         args.aux_dim, args.aux_dim], axis=0)
    rho_expand = tf.expand_dims(rho, axis=-1)
    H = -log_prior(theta) -log_phat(theta, u) + 0.5 * tf.squeeze(tf.matmul(tf.matmul(rho_expand, rho_precision_mat, transpose_a=True), rho_expand)) \
        + 0.5 * (tf.tensordot(u, u, axes=1) + tf.tensordot(p, p, axes=1))
    return H

@pytest.fixture
def mock_hnn_func():
    args_tmp = mock_args()
    def _mock_hnn_func(*args, **kwargs):
        hnn_model = mock_hnn(args_tmp, *args, ham_func=mock_Ham, **kwargs)
        return hnn_model
    return _mock_hnn_func

def _copy(v):
    return v * nuts_hnn.ps.ones(
        nuts_hnn.ps.pad(
            [2], paddings=[[0, nuts_hnn.ps.rank(v)]], constant_values=1),
        dtype=v.dtype)

@pytest.fixture
def mock_uniform():
    def _mock_uniform(shape, dtype, *args, **kwargs):
        return tf.cast(tf.ones(shape) * 0.5, dtype=dtype)
    return _mock_uniform

def test_loop_build_sub_tree(mocker, mock_uniform):
    '''
    This tests whether the loop_build_sub_tree function is the same as before
    when rho_var = 1.0
    '''
    mocker.patch('pseudo_marginal.pm_nuts_hnn.samplers.uniform', mock_uniform)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.samplers.uniform', mock_uniform)
    args = mock_args()
    # grad_flag = True case
    args.grad_flag = True
    nuts_hnn_obj = nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN(mock_target_func,
                                                        step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)
    log_slice_sample = tf.random.uniform(shape=[], maxval=1.0)
    init_energy = tf.constant(100, dtype=tf.float32)
    integrator = nuts_hnn.strang_impl_hnn.StrangIntegrator(mock_target_func, [0.01], 1, args.target_dim, args.aux_dim, args.rho_precision_mat)

    def integrator_base(momentum, state, target, grads):
        a, b, c = integrator(momentum, state, target)
        return a, b, c, grads

    # common settings
    momentum = [tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)]
    state = [tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)]
    target_prob = tf.constant(120, dtype=tf.float32)

    iter_ = tf.zeros([], dtype=tf.int32, name='iter')
    energy_diff_sum_previous = tf.zeros_like(init_energy,
                                      name='energy_diff_sum')
    momentum_cumsum_previous = [tf.zeros(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)]
    leapfrogs_taken = tf.zeros(shape=[], dtype=tf.int32)

    direction = tf.cast(
        tf.random.uniform(shape=[], maxval=2, dtype=tf.int32),
          dtype=tf.bool)
    
    continue_tree=tf.ones_like(init_energy, dtype=tf.bool)
    not_divergence=tf.ones_like(init_energy, dtype=tf.bool)

    # for nuts_hnn_obj
    write_instruction = tf.TensorArray(
          tf.int32,
          size=len(nuts_hnn_obj._write_instruction),
          clear_after_read=False).unstack(nuts_hnn_obj._write_instruction)
    read_instruction = tf.TensorArray(
          tf.int32,
          size=len(nuts_hnn_obj._read_instruction),
          clear_after_read=False).unstack(nuts_hnn_obj._read_instruction)

    current_step_meta_info = nuts_hnn.OneStepMetaInfo(
        log_slice_sample=log_slice_sample,
        init_energy=init_energy,
        write_instruction=write_instruction,
        read_instruction=read_instruction
        )
    
    initial_state = nuts_hnn.TreeDoublingState(
          momentum=momentum,
          state=state,
          target=target_prob,)
    initial_step_state = tf.nest.map_structure(_copy, initial_state)
    
    tree_start_states = tf.nest.map_structure(
        lambda v: nuts_hnn.bu.where_left_justified_mask(direction, v[1], v[0]),
        initial_step_state)

    directions_expanded = [
        nuts_hnn.bu.left_justified_expand_dims_like(direction, state)
        for state in tree_start_states.state
    ]

    initial_state_candidate = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
    
    momentum_state_memory = nuts_hnn.MomentumStateSwap(
          momentum_swap=nuts_hnn_obj.init_momentum_state_memory(momentum),
          state_swap=nuts_hnn_obj.init_momentum_state_memory(state))
    
    # for nuts_obj
    write_instruction_base = tf.TensorArray(
          tf.int32,
          size=len(nuts_obj._write_instruction),
          clear_after_read=False).unstack(nuts_obj._write_instruction)
    read_instruction_base = tf.TensorArray(
          tf.int32,
          size=len(nuts_obj._read_instruction),
          clear_after_read=False).unstack(nuts_obj._read_instruction)

    current_step_meta_info_base = nuts.OneStepMetaInfo(
        log_slice_sample=log_slice_sample,
        init_energy=init_energy,
        write_instruction=write_instruction_base,
        read_instruction=read_instruction_base
        )
    
    initial_state_base = nuts.TreeDoublingState(
          momentum=momentum,
          state=state,
          target=target_prob,
          target_grad_parts=[tf.random.normal(shape=[args.target_dim + args.aux_dim], dtype=tf.float32)])
    initial_step_state_base = tf.nest.map_structure(_copy, initial_state_base)
    
    tree_start_states_base = tf.nest.map_structure(
        lambda v: nuts.bu.where_left_justified_mask(direction, v[1], v[0]),
        initial_step_state_base)

    directions_expanded_base = [
        nuts.bu.left_justified_expand_dims_like(direction, state)
        for state in tree_start_states_base.state
    ]

    initial_state_candidate_base = nuts.TreeDoublingStateCandidate(
          state=initial_state_base.state,
          target=initial_state_base.target,
          energy=initial_state_base.target,
          target_grad_parts=initial_state_base.target_grad_parts,
          weight=tf.zeros([], dtype=tf.int32))
    
    momentum_state_memory_base = nuts.MomentumStateSwap(
          momentum_swap=nuts_obj.init_momentum_state_memory(momentum),
          state_swap=nuts_obj.init_momentum_state_memory(state))
    

    results = nuts_hnn_obj._loop_build_sub_tree(
        directions_expanded,
        integrator,
        integrator,
        current_step_meta_info,
        iter_,
        energy_diff_sum_previous,
        momentum_cumsum_previous,
        leapfrogs_taken,
        tree_start_states,
        initial_state_candidate,
        continue_tree,
        not_divergence,
        momentum_state_memory,
        seed=[0, 1],
        num_grad_flag=tf.zeros([], dtype=tf.bool),
        total_num_grad_steps_count=tf.zeros([], dtype=tf.int32)
    )

    expected = nuts_obj._loop_build_sub_tree(
        directions_expanded_base,
        integrator_base,
        current_step_meta_info_base,
        iter_,
        energy_diff_sum_previous,
        momentum_cumsum_previous,
        leapfrogs_taken,
        tree_start_states_base,
        initial_state_candidate_base,
        continue_tree,
        not_divergence,
        momentum_state_memory_base,
        seed=[0, 1]
    )
    
    assert results[0] == expected[0] # iter_ + 1
    assert tf.reduce_all(tf.equal(results[1], expected[1])) # next_seed
    assert tf.experimental.numpy.allclose(results[2], expected[2]) # energy_diff_sum
    assert tf.experimental.numpy.allclose(results[3], expected[3]) # momentum_cumsum
    assert tf.reduce_all(tf.equal(results[4], expected[4])) # leapfrogs_taken
    for x, y in zip(results[5], expected[5]): # next_tree_state
        assert tf.experimental.numpy.allclose(x, y)
    for x, y in zip(results[6][:2], expected[6][:2]): # next_candidate_tree_state
        assert tf.experimental.numpy.allclose(x, y)
    for x, y in zip(results[6][2:], expected[6][3:]): # next_candidate_tree_state
        assert  tf.experimental.numpy.allclose(x, y)
    
    assert tf.reduce_all(tf.equal(results[7], expected[7])) #continue_tree_next
    assert tf.reduce_all(tf.equal(results[8], expected[8])) # not_divergent_previous & not_divergent_tokeep

@pytest.fixture
def mock_while_loop():
    def _mock_while_loop(*args, **kwargs):
        args = mock_args()
        direction = tf.cast(tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=0), dtype=tf.bool)
        initial_state = nuts_hnn.TreeDoublingState(
          momentum=[tf.ones(shape=[int(args.input_dim//2)]) * 0.5],
          state=[tf.ones(shape=[int(args.input_dim//2)])],
          target=tf.ones(shape=[]) * 0.3,)
        initial_step_state = tf.nest.map_structure(_copy, initial_state)
        
        tree_start_states = tf.nest.map_structure(
            lambda v: nuts_hnn.bu.where_left_justified_mask(direction, v[1], v[0]),
            initial_step_state)
        
        initial_state_candidate = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
    
        nuts_hnn_obj = nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
        
        momentum_state_memory = nuts_hnn.MomentumStateSwap(
          momentum_swap=nuts_hnn_obj.init_momentum_state_memory(initial_state.momentum),
          state_swap=nuts_hnn_obj.init_momentum_state_memory(initial_state.state))
        return [None,
                None,
                tf.ones(shape=[], dtype=tf.float32) * 0.5,
                tf.ones(shape=[], dtype=tf.float32) * 0.6,
                tf.ones(shape=[], dtype=tf.int32),
                tree_start_states,
                initial_state_candidate,
                tf.ones(shape=[], dtype=tf.bool),
                tf.ones(shape=[], dtype=tf.bool),
                momentum_state_memory,
                tf.ones(shape=[], dtype=tf.bool),
                tf.ones(shape=[], dtype=tf.int32)]
    return _mock_while_loop

@pytest.fixture
def mock_while_loop_base():
    def _mock_while_loop(*args, **kwargs):
        args = mock_args()
        direction = tf.cast(tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=0), dtype=tf.bool)
        initial_state = nuts.TreeDoublingState(
          momentum=[tf.ones(shape=[int(args.input_dim//2)]) * 0.5],
          state=[tf.ones(shape=[int(args.input_dim//2)])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[int(args.input_dim//2)]) * 0.2])
        initial_step_state_base = tf.nest.map_structure(_copy, initial_state)
        
        tree_start_states_base = tf.nest.map_structure(
            lambda v: nuts.bu.where_left_justified_mask(direction, v[1], v[0]),
            initial_step_state_base)
        
        initial_state_candidate_base = nuts.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          target_grad_parts=initial_state.target_grad_parts,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
        
        nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)
        
        momentum_state_memory = nuts.MomentumStateSwap(
          momentum_swap=nuts_obj.init_momentum_state_memory(initial_state.momentum),
          state_swap=nuts_obj.init_momentum_state_memory(initial_state.state))
        return [None,
                None,
                tf.ones(shape=[], dtype=tf.float32) * 0.5,
                tf.ones(shape=[], dtype=tf.float32) * 0.6,
                tf.ones(shape=[], dtype=tf.int32),
                tree_start_states_base,
                initial_state_candidate_base,
                tf.ones(shape=[], dtype=tf.bool),
                tf.ones(shape=[], dtype=tf.bool),
                momentum_state_memory,
                ]
    return _mock_while_loop


def test_build_sub_tree(mocker, mock_uniform, mock_while_loop, mock_while_loop_base):
    '''
    test _build_sub_tree function
    '''
    mocker.patch('pseudo_marginal.pm_nuts_hnn.samplers.uniform', mock_uniform)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.samplers.uniform', mock_uniform)
    mocker.patch('pseudo_marginal.pm_nuts_hnn.tf.while_loop', mock_while_loop)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.tf.while_loop', mock_while_loop_base)

    # initialize args function
    args = mock_args()

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)
    
    # prepare the inputs for the samplers
    ## itegrator
    integrator = nuts_hnn.strang_impl_hnn.StrangIntegrator(mock_target_func, [0.01], 1, args.target_dim, args.aux_dim, args.rho_precision_mat)
    def integrator_base(momentum, state, target, grads):
        a, b, c = integrator(momentum, state, target)
        return a, b, c, grads

    # common settings
    init_energy = tf.constant(100, dtype=tf.float32)
    momentum = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target_prob = tf.constant(120, dtype=tf.float32)
    target_grad_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    log_slice_sample = tf.random.uniform(shape=[], maxval=1.0)

    iter_ = tf.zeros([], dtype=tf.int32, name='iter')

    direction = tf.cast(
        tf.random.uniform(shape=[], maxval=2, dtype=tf.int32),
          dtype=tf.bool)
    
    continue_tree=tf.ones_like(init_energy, dtype=tf.bool)
    not_divergence=tf.ones_like(init_energy, dtype=tf.bool)

    # for nuts_hnn_obj
    write_instruction = tf.TensorArray(
          tf.int32,
          size=len(nuts_hnn_obj._write_instruction),
          clear_after_read=False).unstack(nuts_hnn_obj._write_instruction)
    read_instruction = tf.TensorArray(
          tf.int32,
          size=len(nuts_hnn_obj._read_instruction),
          clear_after_read=False).unstack(nuts_hnn_obj._read_instruction)

    current_step_meta_info = nuts_hnn.OneStepMetaInfo(
        log_slice_sample=log_slice_sample,
        init_energy=init_energy,
        write_instruction=write_instruction,
        read_instruction=read_instruction
        )
    
    initial_state = nuts_hnn.TreeDoublingState(
          momentum=momentum,
          state=state,
          target=target_prob,
        )
    initial_step_state = tf.nest.map_structure(_copy, initial_state)
    
    tree_start_states = tf.nest.map_structure(
        lambda v: nuts_hnn.bu.where_left_justified_mask(direction, v[1], v[0]),
        initial_step_state)

    directions_expanded = [
        nuts_hnn.bu.left_justified_expand_dims_like(direction, state)
        for state in tree_start_states.state
    ]
    
    momentum_state_memory = nuts_hnn.MomentumStateSwap(
          momentum_swap=nuts_hnn_obj.init_momentum_state_memory(momentum),
          state_swap=nuts_hnn_obj.init_momentum_state_memory(state))
    
    # for nuts_obj
    write_instruction_base = tf.TensorArray(
          tf.int32,
          size=len(nuts_obj._write_instruction),
          clear_after_read=False).unstack(nuts_obj._write_instruction)
    read_instruction_base = tf.TensorArray(
          tf.int32,
          size=len(nuts_obj._read_instruction),
          clear_after_read=False).unstack(nuts_obj._read_instruction)

    current_step_meta_info_base = nuts.OneStepMetaInfo(
        log_slice_sample=log_slice_sample,
        init_energy=init_energy,
        write_instruction=write_instruction_base,
        read_instruction=read_instruction_base
        )
    
    initial_state_base = nuts.TreeDoublingState(
          momentum=momentum,
          state=state,
          target=target_prob,
          target_grad_parts=target_grad_parts)
    initial_step_state_base = tf.nest.map_structure(_copy, initial_state_base)
    
    tree_start_states_base = tf.nest.map_structure(
        lambda v: nuts.bu.where_left_justified_mask(direction, v[1], v[0]),
        initial_step_state_base)

    directions_expanded_base = [
        nuts.bu.left_justified_expand_dims_like(direction, state)
        for state in tree_start_states_base.state
    ]
    
    momentum_state_memory_base = nuts.MomentumStateSwap(
          momentum_swap=nuts_obj.init_momentum_state_memory(momentum),
          state_swap=nuts_obj.init_momentum_state_memory(state))
    
    results = nuts_hnn_obj._build_sub_tree(
        directions_expanded, 
        integrator, 
        integrator,
        current_step_meta_info,
        tf.bitwise.left_shift(1, iter_),
        tree_start_states,
        continue_tree,
        not_divergence,
        momentum_state_memory,
        seed=[0, 1],
        num_grad_flag = tf.zeros([], dtype=tf.bool)
    )

    expected = nuts_obj._build_sub_tree(
        directions_expanded_base, 
        integrator_base, 
        current_step_meta_info_base,
        1,
        tree_start_states_base,
        continue_tree,
        not_divergence,
        momentum_state_memory_base,
        seed=[0, 1]
    )

    # candidate_tree_state
    assert tf.reduce_all(tf.equal(results[0].state[0], expected[0].state[0]))
    assert tf.reduce_all(tf.equal(results[0].target, expected[0].target))
    assert tf.reduce_all(tf.equal(results[0].energy, expected[0].energy))
    assert tf.reduce_all(tf.equal(results[0].weight, expected[0].weight))
    # final_state
    assert tf.reduce_all(tf.equal(results[1].state[0], expected[1].state[0]))
    assert tf.reduce_all(tf.equal(results[1].target, expected[1].target))
    assert tf.reduce_all(tf.equal(results[1].momentum[0], expected[1].momentum[0]))
    
    assert tf.reduce_all(tf.equal(results[2], expected[2])) # final_not_divergence
    assert tf.reduce_all(tf.equal(results[3], expected[3])) # final_continue_tree
    assert tf.reduce_all(tf.equal(results[4], expected[4])) # energy_diff_tree)sum
    assert tf.reduce_all(tf.equal(results[5], expected[5])) # momentum_cumsum
    assert tf.reduce_all(tf.equal(results[6], expected[6])) # leapfrogs_taken

@pytest.fixture
def mock_build_sub_tree():
    def _mock_build_sub_tree(*args, **kwargs):
        args = mock_args()
        direction = tf.cast(tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=0), dtype=tf.bool)
        initial_state = nuts_hnn.TreeDoublingState(
          momentum=[tf.ones(shape=[int(args.input_dim//2)]) * 0.5],
          state=[tf.ones(shape=[int(args.input_dim//2)])],
          target=tf.ones(shape=[]) * 0.3,
        )
        initial_step_state = tf.nest.map_structure(_copy, initial_state)

        init_energy = tf.constant(100.0, dtype=tf.float32)
        energy_diff_tree_sum = tf.zeros_like(init_energy,
                                      name='energy_diff_sum')
        momentum_subtree_cumsum = [tf.zeros(shape=[int(args.input_dim//2)], dtype=tf.float32)]
        leapfrogs_taken = tf.zeros(shape=[], dtype=tf.int32)
        
        # tree_final_states
        tree_final_states = tf.nest.map_structure(
            lambda v: nuts_hnn.bu.where_left_justified_mask(direction, v[1], v[0]),
            initial_step_state)
        
        # candidate_tree_state
        candidate_tree_state = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
        
        # continue_tree_final
        continue_tree_final=tf.ones_like(init_energy, dtype=tf.bool)
        # final_not_divergence
        final_not_divergence=tf.ones_like(init_energy, dtype=tf.bool)

        return [
            candidate_tree_state,
            tree_final_states,
            final_not_divergence,
            continue_tree_final,
            energy_diff_tree_sum,
            momentum_subtree_cumsum,
            leapfrogs_taken,
            tf.ones(shape=[], dtype=tf.bool), 
            tf.ones(shape=[], dtype=tf.int32) * 2
        ]
    return _mock_build_sub_tree

@pytest.fixture
def mock_build_sub_tree_base():
    def _mock_build_sub_tree(*args, **kwargs):
        args = mock_args()
        direction = tf.cast(tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=0), dtype=tf.bool)
        initial_state = nuts.TreeDoublingState(
          momentum=[tf.ones(shape=[int(args.input_dim//2)]) * 0.5],
          state=[tf.ones(shape=[int(args.input_dim//2)])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[int(args.input_dim//2)]) * 0.2])
        initial_step_state = tf.nest.map_structure(_copy, initial_state)

        init_energy = tf.constant(100.0, dtype=tf.float32)
        energy_diff_tree_sum = tf.zeros_like(init_energy,
                                      name='energy_diff_sum')
        momentum_subtree_cumsum = [tf.zeros(shape=[int(args.input_dim//2)], dtype=tf.float32)]
        leapfrogs_taken = tf.zeros(shape=[], dtype=tf.int32)
        
        # tree_final_states
        tree_final_states = tf.nest.map_structure(
            lambda v: nuts.bu.where_left_justified_mask(direction, v[1], v[0]),
            initial_step_state)
        
        # candidate_tree_state
        candidate_tree_state = nuts.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          target_grad_parts=initial_state.target_grad_parts,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
        
        # continue_tree_final
        continue_tree_final=tf.ones_like(init_energy, dtype=tf.bool)
        # final_not_divergence
        final_not_divergence=tf.ones_like(init_energy, dtype=tf.bool)

        
        return [
            candidate_tree_state,
            tree_final_states,
            final_not_divergence,
            continue_tree_final,
            energy_diff_tree_sum,
            momentum_subtree_cumsum,
            leapfrogs_taken,
        ]
    return _mock_build_sub_tree

def test_loop_tree_doubling(mocker, mock_uniform, mock_build_sub_tree, mock_build_sub_tree_base):
    '''
    test for _loop_tree_doubling
    '''
    mocker.patch('pseudo_marginal.pm_nuts_hnn.samplers.uniform', mock_uniform)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.samplers.uniform', mock_uniform)
    mocker.patch('pseudo_marginal.pm_nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN._build_sub_tree', mock_build_sub_tree)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.NoUTurnSampler._build_sub_tree', mock_build_sub_tree_base)

    # initialize args function
    args = mock_args()

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)
    
    # prepare the inputs for the samplers
    # common settings
    init_energy = tf.constant(100, dtype=tf.float32)
    momentum = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    state = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    target_prob = tf.constant(120, dtype=tf.float32)
    target_grad_parts = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    log_slice_sample = tf.random.uniform(shape=[], maxval=1.0)

    iter_ = tf.zeros([], dtype=tf.int32, name='iter')

    # for nuts_hnn_obj
    write_instruction = tf.TensorArray(
          tf.int32,
          size=len(nuts_hnn_obj._write_instruction),
          clear_after_read=False).unstack(nuts_hnn_obj._write_instruction)
    read_instruction = tf.TensorArray(
          tf.int32,
          size=len(nuts_hnn_obj._read_instruction),
          clear_after_read=False).unstack(nuts_hnn_obj._read_instruction)

    ## current_step_meta_info
    current_step_meta_info = nuts_hnn.OneStepMetaInfo(
        log_slice_sample=log_slice_sample,
        init_energy=init_energy,
        write_instruction=write_instruction,
        read_instruction=read_instruction
        )
    
    initial_state = nuts_hnn.TreeDoublingState(
          momentum=momentum,
          state=state,
          target=target_prob,
        )
    ## initial_step_state
    initial_step_state = tf.nest.map_structure(_copy, initial_state)


    initial_state_candidate = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
    
    ## initial_step_metastate
    initial_step_metastate = nuts_hnn.TreeDoublingMetaState(
          candidate_state=initial_state_candidate,
          is_accepted=tf.zeros_like(init_energy, dtype=tf.bool),
          momentum_sum=momentum,
          energy_diff_sum=tf.zeros_like(init_energy),
          leapfrog_count=tf.zeros_like(init_energy, dtype=tf.int32),
          continue_tree=tf.ones_like(init_energy, dtype=tf.bool),
          not_divergence=tf.ones_like(init_energy, dtype=tf.bool),
          num_grad_flag=tf.zeros_like(init_energy, dtype=tf.bool),
          total_num_grad_steps_count=tf.zeros_like(init_energy, dtype=tf.int32))
    
    
    ## momentum_state_memory
    momentum_state_memory = nuts_hnn.MomentumStateSwap(
          momentum_swap=nuts_hnn_obj.init_momentum_state_memory(momentum),
          state_swap=nuts_hnn_obj.init_momentum_state_memory(state))
    
    # for nuts_obj
    write_instruction_base = tf.TensorArray(
          tf.int32,
          size=len(nuts_obj._write_instruction),
          clear_after_read=False).unstack(nuts_obj._write_instruction)
    read_instruction_base = tf.TensorArray(
          tf.int32,
          size=len(nuts_obj._read_instruction),
          clear_after_read=False).unstack(nuts_obj._read_instruction)

    current_step_meta_info_base = nuts.OneStepMetaInfo(
        log_slice_sample=log_slice_sample,
        init_energy=init_energy,
        write_instruction=write_instruction_base,
        read_instruction=read_instruction_base
        )
    
    initial_state_base = nuts.TreeDoublingState(
          momentum=momentum,
          state=state,
          target=target_prob,
          target_grad_parts=target_grad_parts)
    initial_step_state_base = tf.nest.map_structure(_copy, initial_state_base)
    

    initial_state_candidate_base = nuts.TreeDoublingStateCandidate(
          state=initial_state_base.state,
          target=initial_state_base.target,
          energy=initial_state_base.target,
          target_grad_parts=initial_state_base.target_grad_parts,
          weight=tf.zeros([], dtype=tf.int32))
    
     ## initial_step_metastate
    initial_step_metastate_base = nuts.TreeDoublingMetaState(
          candidate_state=initial_state_candidate_base,
          is_accepted=tf.zeros_like(init_energy, dtype=tf.bool),
          momentum_sum=momentum,
          energy_diff_sum=tf.zeros_like(init_energy),
          leapfrog_count=tf.zeros_like(init_energy, dtype=tf.int32),
          continue_tree=tf.ones_like(init_energy, dtype=tf.bool),
          not_divergence=tf.ones_like(init_energy, dtype=tf.bool),)
    

    momentum_state_memory_base = nuts.MomentumStateSwap(
          momentum_swap=nuts_obj.init_momentum_state_memory(momentum),
          state_swap=nuts_obj.init_momentum_state_memory(state))
    
    results = nuts_hnn_obj._loop_tree_doubling(
       [0.01], 
       momentum_state_memory,
       current_step_meta_info, 
       iter_, 
       initial_step_state,
       initial_step_metastate, 
       seed = [0, 1]
    )

    expected = nuts_obj._loop_tree_doubling(
       [0.01], 
       momentum_state_memory_base,
       current_step_meta_info_base, 
       iter_, 
       initial_step_state_base,
       initial_step_metastate_base, 
       seed = [0, 1]
    )

    assert tf.reduce_all(tf.equal(results[0], expected[0])) # iter_ + 1
    for x, y in zip(results[2][0], expected[2][0]): # new_step_state
        assert tf.reduce_all(tf.equal(x, y))
    for x, y in zip(results[2][1:], expected[2][1:]): # new_step_state
        assert tf.reduce_all(tf.equal(x, y))
    
    # new_step_metastate 
    ## candidate_state
    assert tf.reduce_all(tf.equal(results[3][0].state[0], expected[3][0].state[0]))
    assert tf.reduce_all(tf.equal(results[3][0].target, expected[3][0].target))
    assert tf.reduce_all(tf.equal(results[3][0].energy, expected[3][0].energy))
    assert tf.reduce_all(tf.equal(results[3][0].weight, expected[3][0].weight))
    for x, y in zip(results[3][1:7], expected[3][1:7]): # new_step_metastate
        assert tf.reduce_all(tf.equal(x, y))

def test_start_trajectory_batched():
    # initialize args function
    args = mock_args()
    for x in [1.0, tf.random.normal(shape=[args.target_dim]).numpy().tolist()]:
        args.rho_var = x

        # initialize the sampler
        nuts_hnn_obj = nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
        nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)

        # initialize inputs
        state = [tf.random.normal(shape=[args.target_dim + args.aux_dim])]
        target_log_prob = tf.random.normal(shape=[])
        seed = [0, 1]

        momentum, init_energy, log_slice_sample = nuts_hnn_obj._start_trajectory_batched(state, target_log_prob, seed)
        momentum_expected, init_energy_expected, log_slice_sample_expected = nuts_obj._start_trajectory_batched(state, target_log_prob, seed)

        assert momentum[0].dtype == momentum_expected[0].dtype
        assert momentum[0].shape == momentum_expected[0].shape
        assert init_energy.shape == init_energy_expected.shape
        assert log_slice_sample.shape == log_slice_sample_expected.shape

def test_bootstrap_results(mocker, mock_hnn_func):
    '''
    test for bootstrap_results
    '''
    # initialize args function
    args = mock_args()

    # hnn_model
    mocker.patch('pseudo_marginal.pm_nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN._load_hnn_model', mock_hnn_func)

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)

    init_state = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    results = nuts_hnn_obj.bootstrap_results(init_state)
    expected = nuts_obj.bootstrap_results(init_state)

    assert tf.equal(results[0], expected[0])
    for x, y in zip(results[1:-3], expected[2:]):
        print('x', x)
        print('y', y)
        if isinstance(x, list):
            assert tf.reduce_all(tf.equal(x[0], y[0]))
        else:
            assert tf.reduce_all(tf.equal(x, y))
    
    assert tf.equal(results[-3], tf.zeros([], dtype=tf.bool)) # num_grad_flag
    assert tf.equal(results[-2], tf.zeros([], dtype=tf.int32)) # num_grad_steps_count
    assert tf.equal(results[-1], tf.zeros([], dtype=tf.int32)) # total_num_grad_steps_count

@pytest.fixture
def mock_while_loop_one_step():
    def _mock_while_loop(*args, **kwargs):
        args = mock_args()
        initial_state = nuts_hnn.TreeDoublingState(
          momentum=[tf.ones(shape=[int(args.input_dim//2)]) * 0.5],
          state=[tf.ones(shape=[int(args.input_dim//2)])],
          target=tf.ones(shape=[]) * 0.3,
        )

        init_energy = tf.constant(100.0, dtype=tf.float32)
        initial_state_candidate = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
    
        ## initial_step_metastate
        initial_step_metastate = nuts_hnn.TreeDoublingMetaState(
            candidate_state=initial_state_candidate,
            is_accepted=tf.zeros_like(init_energy, dtype=tf.bool),
            momentum_sum=initial_state.momentum,
            energy_diff_sum=tf.zeros_like(init_energy),
            leapfrog_count=tf.zeros_like(init_energy, dtype=tf.int32),
            continue_tree=tf.ones_like(init_energy, dtype=tf.bool),
            not_divergence=tf.ones_like(init_energy, dtype=tf.bool),
            num_grad_flag=tf.zeros_like(init_energy, dtype=tf.bool),
            total_num_grad_steps_count=tf.zeros_like(init_energy, dtype=tf.int32))
        return [None, None, None, initial_step_metastate]
    return _mock_while_loop

@pytest.fixture
def mock_while_loop_one_step_base():
    def _mock_while_loop(*args, **kwargs):
        args = mock_args()
        initial_state = nuts.TreeDoublingState(
          momentum=[tf.ones(shape=[int(args.input_dim//2)]) * 0.5],
          state=[tf.ones(shape=[int(args.input_dim//2)])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[int(args.input_dim//2)]) * 0.2])

        init_energy = tf.constant(100.0, dtype=tf.float32)
        initial_state_candidate = nuts.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          target_grad_parts=initial_state.target_grad_parts,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
    
        ## initial_step_metastate
        initial_step_metastate = nuts.TreeDoublingMetaState(
            candidate_state=initial_state_candidate,
            is_accepted=tf.zeros_like(init_energy, dtype=tf.bool),
            momentum_sum=initial_state.momentum,
            energy_diff_sum=tf.zeros_like(init_energy),
            leapfrog_count=tf.zeros_like(init_energy, dtype=tf.int32),
            continue_tree=tf.ones_like(init_energy, dtype=tf.bool),
            not_divergence=tf.ones_like(init_energy, dtype=tf.bool),
        )
        return [None, None, None, initial_step_metastate]
    return _mock_while_loop

def test_one_step(mocker, mock_hnn_func, mock_while_loop_one_step, mock_while_loop_one_step_base):
    # mock
    mocker.patch('pseudo_marginal.pm_nuts_hnn.tf.while_loop', mock_while_loop_one_step)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.tf.while_loop', mock_while_loop_one_step_base)

    # initialize args function
    args = mock_args()

    # hnn_model
    mocker.patch('pseudo_marginal.pm_nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN._load_hnn_model', mock_hnn_func)

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)

    # initialize states
    init_state = [tf.random.normal(shape=[int(args.input_dim//2)])]
    current_state = [tf.random.normal(shape=[int(args.input_dim//2)])]

    pkr = nuts_hnn_obj.bootstrap_results(init_state)
    pkr_base = nuts_obj.bootstrap_results(init_state)

    # get results
    _, results = nuts_hnn_obj.one_step(current_state, pkr, seed=[0,1])
    _, expected = nuts_obj.one_step(current_state, pkr_base, seed=[0,1])

    # comparison
    assert tf.equal(results.target_log_prob, expected.target_log_prob)
    assert tf.equal(results.step_size, expected.step_size)
    assert tf.equal(results.leapfrogs_taken, expected.leapfrogs_taken)
    assert tf.equal(results.is_accepted, expected.is_accepted)
    assert tf.equal(results.reach_max_depth, expected.reach_max_depth)
    assert tf.equal(results.has_divergence, expected.has_divergence)
    assert tf.equal(results.energy, expected.energy)
    assert tf.equal(results.num_grad_flag, tf.zeros_like(results.energy, dtype=tf.bool))
    assert tf.equal(results.num_grad_steps_count, tf.zeros([], dtype=tf.int32))
    assert tf.equal(results.total_num_grad_steps_count,  tf.zeros([], dtype=tf.int32))

def test_load_hnn_model(mocker, mock_hnn_func, capsys):
    # initialize target function
    args = mock_args()

    # mock
    mocker.patch('pseudo_marginal.pm_nuts_hnn.HNN', mock_hnn_func)
    mocker.patch('pseudo_marginal.pm_nuts_hnn.MLP')
    mocker.patch('pseudo_marginal.pm_nuts_hnn.os.path.exists', return_value=False)

    # path
    path = '{}/ckp/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_{}'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, args.nn_model_name)

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.PseudoMarginal_NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    model = nuts_hnn_obj._load_hnn_model()
    captured = capsys.readouterr()
    assert 'The checkpoint does not exist at {}. We just use a random initialization!\n'.format(path) in captured.out
    
    mocker.patch('pseudo_marginal.pm_nuts_hnn.os.path.exists', return_value=True)
    model = nuts_hnn_obj._load_hnn_model()
    captured = capsys.readouterr()
    assert 'Load the HNN checkpoint!' in captured.out