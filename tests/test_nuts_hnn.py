import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from tf_version import nuts_hnn
from tf_version.functions_tf import dist_func
from tensorflow_probability.python.mcmc import nuts
nuts.GENERALIZED_UTURN = False
nuts.MULTINOMIAL_SAMPLE = False

class mock_args():
    def __init__(self):
        self.input_dim = 4
        self.dist_name = '2D_Gauss_mix'
        self.lf_threshold = 1000.0
        self.hnn_threshold = 10.0
        self.hidden_dim = 100
        self.nonlinearity = 'sine'
        self.num_layers = 3
    
    def __getattr__(self, name):
        return None

class mock_hnn(tf.Module):
    def __init__(self, input_dim, *args, ham_func=None, assume_canonical_coords=True, name=None, **kwargs):
        super().__init__(name)
        self.input_dim = input_dim
        self.ham_func = ham_func
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(int(input_dim)) # Levi-Civita permutation tensor
        
    def __call__(self, x):
        return x
    
    def load_weights(self, x):
        pass

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

@pytest.fixture
def mock_hnn_func():
    args_tmp = mock_args()
    dist_func_obj = dist_func(args_tmp)
    ham_func = dist_func_obj.get_Hamiltonian
    def _mock_hnn_func(*args, **kwargs):
        hnn_model = mock_hnn(args_tmp.input_dim, *args, ham_func=ham_func, **kwargs)
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
    '''
    mocker.patch('tf_version.nuts_hnn.samplers.uniform', mock_uniform)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.samplers.uniform', mock_uniform)
    
    # initialize target function
    args = mock_args()
    dist_func_obj = dist_func(args)
    ham_func = dist_func_obj.get_Hamiltonian
    mock_target_func = dist_func_obj.get_target_log_prob_func

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)
    log_slice_sample = tf.random.uniform(shape=[], maxval=1.0)

    # prepare the inputs to the function

    ## integrator
    integrator = nuts.leapfrog_impl.SimpleLeapfrogIntegrator(mock_target_func, [0.01], 1)

    # common settings
    init_energy = tf.constant(100, dtype=tf.float32)
    momentum = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]
    state = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]
    target_prob = tf.constant(120, dtype=tf.float32)
    target_grad_parts = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]

    ### iter_
    iter_ = tf.zeros([], dtype=tf.int32, name='iter')
    ### energy_diff_sum_previous
    energy_diff_sum_previous = tf.zeros_like(init_energy,
                                      name='energy_diff_sum')
    ### momentum_cumsum_previous
    momentum_cumsum_previous = [tf.zeros(shape=[args.input_dim], dtype=tf.float32)]
    
    ### leapfrogs_taken
    leapfrogs_taken = tf.zeros(shape=[], dtype=tf.int32)

    direction = tf.cast(
        tf.random.uniform(shape=[], maxval=2, dtype=tf.int32),
          dtype=tf.bool)
    
    ### continue_tree_previous
    continue_tree=tf.ones_like(init_energy, dtype=tf.bool)
    ### not_divergent_previous
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

    ### current_step_meta_info
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
          target_grad_parts=target_grad_parts)
    initial_step_state = tf.nest.map_structure(_copy, initial_state)
    
    ### prev_tree_state
    tree_start_states = tf.nest.map_structure(
        lambda v: nuts_hnn.bu.where_left_justified_mask(direction, v[1], v[0]),
        initial_step_state)

    ### directions
    directions_expanded = [
        nuts_hnn.bu.left_justified_expand_dims_like(direction, state)
        for state in tree_start_states.state
    ]

    ### candidate_tree_state
    initial_state_candidate = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          target_grad_parts=target_grad_parts,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
    
    ### momentum_state_memory
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

    ### current_step_meta_info
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
    
    ### prev_tree_state
    tree_start_states_base = tf.nest.map_structure(
        lambda v: nuts.bu.where_left_justified_mask(direction, v[1], v[0]),
        initial_step_state_base)

    ### directions
    directions_expanded_base = [
        nuts.bu.left_justified_expand_dims_like(direction, state)
        for state in tree_start_states_base.state
    ]

    ### candidate_tree_state
    initial_state_candidate_base = nuts.TreeDoublingStateCandidate(
          state=initial_state_base.state,
          target=initial_state_base.target,
          energy=initial_state_base.target,
          target_grad_parts=initial_state_base.target_grad_parts,
          weight=tf.zeros([], dtype=tf.int32))
    
    ### momentum_state_memory
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
        num_grad_flag = tf.zeros(shape=[], dtype=tf.bool),
        total_num_grad_steps_count = tf.zeros(shape=[], dtype=tf.int32)
    )
    

    expected = nuts_obj._loop_build_sub_tree(
        directions_expanded_base,
        integrator,
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
    for x, y in zip(results[6], expected[6]): # next_tree_state
        assert tf.experimental.numpy.allclose(x, y)
    assert tf.reduce_all(tf.equal(results[7], expected[7])) #continue_tree_next
    assert tf.reduce_all(tf.equal(results[8], expected[8])) # not_divergent_previous & not_divergent_tokeep

@pytest.fixture
def mock_while_loop():
    def _mock_while_loop(*args, **kwargs):
        args = mock_args()
        direction = tf.cast(tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=0), dtype=tf.bool)
        initial_state = nuts_hnn.TreeDoublingState(
          momentum=[tf.ones(shape=[args.input_dim]) * 0.5],
          state=[tf.ones(shape=[args.input_dim])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[args.input_dim]) * 0.2])
        initial_step_state = tf.nest.map_structure(_copy, initial_state)
        
        tree_start_states = tf.nest.map_structure(
            lambda v: nuts_hnn.bu.where_left_justified_mask(direction, v[1], v[0]),
            initial_step_state)
        
        initial_state_candidate = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          target_grad_parts=initial_state.target_grad_parts,
          energy=initial_state.target,
          weight=tf.zeros([], dtype=tf.int32))
        
        args = mock_args()
        dist_func_obj = dist_func(args)
        mock_target_func = dist_func_obj.get_target_log_prob_func
        nuts_hnn_obj = nuts_hnn.NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
        
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
          momentum=[tf.ones(shape=[args.input_dim]) * 0.5],
          state=[tf.ones(shape=[args.input_dim])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[args.input_dim]) * 0.2])
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
        
        args = mock_args()
        dist_func_obj = dist_func(args)
        mock_target_func = dist_func_obj.get_target_log_prob_func
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
    mocker.patch('tf_version.nuts_hnn.samplers.uniform', mock_uniform)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.samplers.uniform', mock_uniform)
    mocker.patch('tf_version.nuts_hnn.tf.while_loop', mock_while_loop)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.tf.while_loop', mock_while_loop_base)

    # initialize target function
    args = mock_args()
    dist_func_obj = dist_func(args)
    mock_target_func = dist_func_obj.get_target_log_prob_func

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)
    
    # prepare the inputs for the samplers
    
    ## integrator
    integrator = nuts.leapfrog_impl.SimpleLeapfrogIntegrator(mock_target_func, [0.01], 1)

    # common settings
    init_energy = tf.constant(100, dtype=tf.float32)
    momentum = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]
    state = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]
    target_prob = tf.constant(120, dtype=tf.float32)
    target_grad_parts = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]
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
          target_grad_parts=target_grad_parts)
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
        integrator, 
        current_step_meta_info_base,
        1,
        tree_start_states_base,
        continue_tree,
        not_divergence,
        momentum_state_memory_base,
        seed=[0, 1]
    )

    for x, y in zip(results[0], expected[0]): # candidate_tree_state
        assert tf.reduce_all(tf.equal(x, y))
    for x, y in zip(results[1], expected[1]): # final_state
        assert tf.reduce_all(tf.equal(x, y))
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
          momentum=[tf.ones(shape=[args.input_dim]) * 0.5],
          state=[tf.ones(shape=[args.input_dim])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[args.input_dim]) * 0.2])
        initial_step_state = tf.nest.map_structure(_copy, initial_state)

        init_energy = tf.constant(100.0, dtype=tf.float32)
        energy_diff_tree_sum = tf.zeros_like(init_energy,
                                      name='energy_diff_sum')
        momentum_subtree_cumsum = [tf.zeros(shape=[args.input_dim], dtype=tf.float32)]
        leapfrogs_taken = tf.zeros(shape=[], dtype=tf.int32)
        
        # tree_final_states
        tree_final_states = tf.nest.map_structure(
            lambda v: nuts_hnn.bu.where_left_justified_mask(direction, v[1], v[0]),
            initial_step_state)
        
        # candidate_tree_state
        candidate_tree_state = nuts_hnn.TreeDoublingStateCandidate(
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
          momentum=[tf.ones(shape=[args.input_dim]) * 0.5],
          state=[tf.ones(shape=[args.input_dim])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[args.input_dim]) * 0.2])
        initial_step_state = tf.nest.map_structure(_copy, initial_state)

        init_energy = tf.constant(100.0, dtype=tf.float32)
        energy_diff_tree_sum = tf.zeros_like(init_energy,
                                      name='energy_diff_sum')
        momentum_subtree_cumsum = [tf.zeros(shape=[args.input_dim], dtype=tf.float32)]
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
    mocker.patch('tf_version.nuts_hnn.samplers.uniform', mock_uniform)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.samplers.uniform', mock_uniform)
    mocker.patch('tf_version.nuts_hnn.NoUTurnSampler_HNN._build_sub_tree', mock_build_sub_tree)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.NoUTurnSampler._build_sub_tree', mock_build_sub_tree_base)

    # initialize target function
    args = mock_args()
    dist_func_obj = dist_func(args)
    mock_target_func = dist_func_obj.get_target_log_prob_func

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)
    
    # prepare the inputs for the samplers
    
    # common settings
    init_energy = tf.constant(100, dtype=tf.float32)
    momentum = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]
    state = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]
    target_prob = tf.constant(120, dtype=tf.float32)
    target_grad_parts = [tf.random.normal(shape=[args.input_dim], dtype=tf.float32)]
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
          target_grad_parts=target_grad_parts)
    ## initial_step_state
    initial_step_state = tf.nest.map_structure(_copy, initial_state)


    initial_state_candidate = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          target_grad_parts=target_grad_parts,
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
    for x, y in zip(results[3][0], expected[3][0]): # new_step_metastate
        assert tf.reduce_all(tf.equal(x, y))
    for x, y in zip(results[3][1:7], expected[3][1:7]): # new_step_metastate
        assert tf.reduce_all(tf.equal(x, y))

def test_bootstrap_results(mocker, mock_hnn_func):
    '''
    test for bootstrap_results
    '''
    # initialize target function
    args = mock_args()
    dist_func_obj = dist_func(args)
    mock_target_func = dist_func_obj.get_target_log_prob_func

    # hnn_model
    mocker.patch('tf_version.nuts_hnn.NoUTurnSampler_HNN._load_hnn_model', mock_hnn_func)

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    nuts_obj = nuts.NoUTurnSampler(mock_target_func, step_size=0.01)

    init_state = [tf.random.normal(shape=[int(args.input_dim//2)], dtype=tf.float32)]
    results = nuts_hnn_obj.bootstrap_results(init_state)
    expected = nuts_obj.bootstrap_results(init_state)

    for x, y in zip(results[:-3], expected):
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
          momentum=[tf.ones(shape=[args.input_dim]) * 0.5],
          state=[tf.ones(shape=[args.input_dim])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[args.input_dim]) * 0.2])

        init_energy = tf.constant(100.0, dtype=tf.float32)
        initial_state_candidate = nuts_hnn.TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          target_grad_parts=initial_state.target_grad_parts,
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
          momentum=[tf.ones(shape=[args.input_dim]) * 0.5],
          state=[tf.ones(shape=[args.input_dim])],
          target=tf.ones(shape=[]) * 0.3,
          target_grad_parts=[tf.ones(shape=[args.input_dim]) * 0.2])

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
    mocker.patch('tf_version.nuts_hnn.tf.while_loop', mock_while_loop_one_step)
    mocker.patch('tensorflow_probability.python.mcmc.nuts.tf.while_loop', mock_while_loop_one_step_base)

    # initialize target function
    args = mock_args()
    dist_func_obj = dist_func(args)
    mock_target_func = dist_func_obj.get_target_log_prob_func

    # hnn_model
    mocker.patch('tf_version.nuts_hnn.NoUTurnSampler_HNN._load_hnn_model', mock_hnn_func)

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
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
    assert tf.reduce_all(tf.equal(results.grads_target_log_prob[0], expected.grads_target_log_prob[0]))
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
    dist_func_obj = dist_func(args)
    mock_target_func = dist_func_obj.get_target_log_prob_func

    # mock
    mocker.patch('tf_version.nuts_hnn.HNN', mock_hnn_func)
    mocker.patch('tf_version.nuts_hnn.MLP')
    mocker.patch('tf_version.nuts_hnn.os.path.exists', return_value=False)

    # path
    path = '{}/ckp/{}_d{}_n{}_l{}_t{}_{}/{}_d{}_n{}_l{}_t{}_{}.ckpt'.format(args.save_dir, args.dist_name, args.input_dim, args.num_samples, 
                                                      args.len_sample, args.total_steps, args.grad_type, args.dist_name, args.input_dim, 
                                                      args.num_samples, args.len_sample, args.total_steps, args.grad_type)

    # initialize the sampler
    nuts_hnn_obj = nuts_hnn.NoUTurnSampler_HNN(mock_target_func, step_size=0.01, hnn_model_args=args)
    model = nuts_hnn_obj._load_hnn_model()
    captured = capsys.readouterr()
    assert 'The checkpoint does not exist at {}. We just use a random initialization!\n'.format(path) in captured.out
    
    mocker.patch('tf_version.nuts_hnn.os.path.exists', return_value=True)
    model = nuts_hnn_obj._load_hnn_model()
    captured = capsys.readouterr()
    assert 'Load the HNN checkpoint!' in captured.out