# Copyright 2019 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""No U-Turn Sampler.

The implementation closely follows [1; Algorithm 3], with Multinomial sampling
on the tree (instead of slice sampling) and a generalized No-U-Turn termination
criterion [2; Appendix A].

Achieves batch execution across chains by precomputing the recursive tree
doubling data access patterns and then executes this "unrolled" data pattern via
a `tf.while_loop`.

#### References

[1]: Matthew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively
     Setting Path Lengths in Hamiltonian Monte Carlo.
     In _Journal of Machine Learning Research_, 15(1):1593-1623, 2014.
     http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf

[2]: Michael Betancourt. A Conceptual Introduction to Hamiltonian Monte Carlo.
     _arXiv preprint arXiv:1701.02434_, 2018. https://arxiv.org/abs/1701.02434
"""

import collections
import numpy as np
import math

import tensorflow as tf

from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.mcmc import kernel
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

import tensorflow_probability as tfp
tfd = tfp.distributions

import os,sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
import pseudo_marginal.strang_integrator_hnn as strang_impl_hnn
from pseudo_marginal.hnn import HNN
from pseudo_marginal.nn_models import *


JAX_MODE = False

##############################################################
### BEGIN STATIC CONFIGURATION ###############################
##############################################################
TREE_COUNT_DTYPE = tf.int32           # Default: tf.int32

# Whether to use slice sampling (original NUTS implementation in [1]) or
# multinomial sampling (implementation in [2]) from the tree trajectory.
MULTINOMIAL_SAMPLE = False             # Default: True

# Whether to use U turn criteria in [1] or generalized U turn criteria in [2]
# to check the tree trajectory.
GENERALIZED_UTURN = False              # Default: True
##############################################################
### END STATIC CONFIGURATION #################################
##############################################################

__all__ = [
    'PseudoMarginal_NoUTurnSampler_HNN',
]


class NUTSKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'NUTSKernelResults',
        [
            'target_log_prob',
            'step_size',
            'log_accept_ratio',
            'leapfrogs_taken',  # How many leapfrogs each chain took this step.
            'is_accepted',
            'reach_max_depth',
            'has_divergence',
            'energy',
            'seed',
            'num_grad_flag', # 1_lf
            'num_grad_steps_count', # Count how many num_grad steps are taken (n_lf)
            'total_num_grad_steps_count' # total cumulation of numerical gradients steps
        ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class MomentumStateSwap(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('MomentumStateSwap',
                           ['momentum_swap', 'state_swap'])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class OneStepMetaInfo(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('OneStepMetaInfo',
                           ['log_slice_sample',
                            'init_energy',
                            'write_instruction',
                            'read_instruction',
                           ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class TreeDoublingState(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('TreeDoublingState',
                           ['momentum',
                            'state',
                            'target',
                            ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class TreeDoublingStateCandidate(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'TreeDoublingStateCandidate',
        [
            'state',
            'target',
            'energy',
            'weight',
        ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class TreeDoublingMetaState(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'TreeDoublingMetaState',
        [
            'candidate_state',  # A namedtuple of TreeDoublingStateCandidate.
            'is_accepted',
            'momentum_sum',     # Sum of momentum of the current tree for
                                # generalized U turn criteria.
            'energy_diff_sum',  # Sum over all states explored within the
                                # subtree of Metropolis acceptance probabilities
                                # exp(min(0, H' - H0)), where H0 is the negative
                                # energy of the initial state and H' is the
                                # negative energy of a state explored in the
                                # subtree.
                                # TODO(b/150152798): Do sum in log-space.
            'leapfrog_count',   # How many leapfrogs each chain has taken.
            'continue_tree',
            'not_divergence',
            'num_grad_flag',     # whether to use the numerical gradients: 1_lf
            'total_num_grad_steps_count' # total number of numerical gradient counts
        ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class PseudoMarginal_NoUTurnSampler_HNN(kernel.TransitionKernel):
  """Runs one step of the No U-Turn Sampler.

  The No U-Turn Sampler (NUTS) is an adaptive variant of the Hamiltonian Monte
  Carlo (HMC) method for MCMC. NUTS adapts the distance traveled in response to
  the curvature of the target density. Conceptually, one proposal consists of
  reversibly evolving a trajectory through the sample space, continuing until
  that trajectory turns back on itself (hence the name, 'No U-Turn'). This class
  implements one random NUTS step from a given `current_state`.
  Mathematical details and derivations can be found in
  [Hoffman, Gelman (2011)][1] and [Betancourt (2018)][2].

  The `one_step` function can update multiple chains in parallel. It assumes
  that a prefix of leftmost dimensions of `current_state` index independent
  chain states (and are therefore updated independently).  The output of
  `target_log_prob_fn(*current_state)` should sum log-probabilities across all
  event dimensions.  Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0][0, ...]` could have a
  different target distribution from `current_state[0][1, ...]`.  These
  semantics are governed by `target_log_prob_fn(*current_state)`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

  #### References

  [1]: Matthew D. Hoffman, Andrew Gelman.  The No-U-Turn Sampler: Adaptively
  Setting Path Lengths in Hamiltonian Monte Carlo.  2011.
  https://arxiv.org/pdf/1111.4246.pdf.

  [2]: Michael Betancourt. A Conceptual Introduction to Hamiltonian Monte Carlo.
  _arXiv preprint arXiv:1701.02434_, 2018. https://arxiv.org/abs/1701.02434
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               max_tree_depth=10,
               max_energy_diff=1000.,
               unrolled_leapfrog_steps=1,
               parallel_iterations=10,
               experimental_shard_axis_names=None,
               hnn_model_args=None,
               name=None):
    """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      max_tree_depth: Maximum depth of the tree implicitly built by NUTS. The
        maximum number of leapfrog steps is bounded by `2**max_tree_depth` i.e.
        the number of nodes in a binary tree `max_tree_depth` nodes deep. The
        default setting of 10 takes up to 1024 leapfrog steps.
      max_energy_diff: Scaler threshold of energy differences at each leapfrog,
        divergence samples are defined as leapfrog steps that exceed this
        threshold. Default to 1000.
      unrolled_leapfrog_steps: The number of leapfrogs to unroll per tree
        expansion step. Applies a direct linear multipler to the maximum
        trajectory length implied by max_tree_depth. Defaults to 1.
      parallel_iterations: The number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'NoUTurnSampler').
    """
    with tf.name_scope(name or 'PseudoMarginal_NoUTurnSampler_HNN') as name:
      # Process `max_tree_depth` argument.
      max_tree_depth = tf.get_static_value(max_tree_depth)
      if max_tree_depth is None or max_tree_depth < 1:
        raise ValueError(
            'max_tree_depth must be known statically and >= 1 but was '
            '{}'.format(max_tree_depth))
      self._max_tree_depth = max_tree_depth

      # Compute parameters derived from `max_tree_depth`.
      instruction_array = build_tree_uturn_instruction(
          max_tree_depth, init_memory=-1)
      [
          write_instruction_numpy,
          read_instruction_numpy
      ] = generate_efficient_write_read_instruction(instruction_array)

      # TensorArray version of the read/write instruction need to be created
      # within the function call to be compatible with XLA. Here we store the
      # numpy version of the instruction and convert it to TensorArray later.
      self._write_instruction = write_instruction_numpy
      self._read_instruction = read_instruction_numpy

      # Process all other arguments.
      self._target_log_prob_fn = target_log_prob_fn
      self._step_size = step_size

      self._parallel_iterations = parallel_iterations
      self._unrolled_leapfrog_steps = unrolled_leapfrog_steps
      self._name = name
      self._max_energy_diff = max_energy_diff
      self._hnn_model_args = hnn_model_args
      # load hnn model
      hnn_model = self._load_hnn_model()
      self._hnn_model = hnn_model
      # prepare for the M^{-1} case
      if isinstance(hnn_model_args.rho_var, float):
        rho_cov_mat = tf.eye(hnn_model_args.target_dim, dtype=tf.float32) * hnn_model_args.rho_var
        multivariate_normal_dist = None
      else:
        rho_cov_mat = tf.linalg.diag(tf.constant(hnn_model_args.rho_var, dtype=tf.float32))
        assert len(hnn_model_args.rho_var) == hnn_model_args.target_dim, 'if rho_var is a list, it must have length {}'.format(hnn_model_args.target_dim)
        multivariate_normal_dist = tfd.MultivariateNormalDiag(scale_diag=hnn_model_args.rho_var)
      rho_precision_mat = tf.linalg.inv(rho_cov_mat)
      self._target_momentum_precision_mat = rho_precision_mat
      self._target_momentum_cov_mat = rho_cov_mat
      self._multivariate_normal_dist = multivariate_normal_dist

      self._parameters = dict(
          target_log_prob_fn=target_log_prob_fn,
          step_size=step_size,
          max_tree_depth=max_tree_depth,
          max_energy_diff=max_energy_diff,
          unrolled_leapfrog_steps=unrolled_leapfrog_steps,
          parallel_iterations=parallel_iterations,
          experimental_shard_axis_names=experimental_shard_axis_names,
          name=name,
          hnn_model_args=hnn_model_args,
      )   

  @property
  def target_log_prob_fn(self):
    return self._target_log_prob_fn

  @property
  def step_size(self):
    return self._step_size

  @property
  def max_tree_depth(self):
    return self._max_tree_depth

  @property
  def max_energy_diff(self):
    return self._max_energy_diff

  @property
  def unrolled_leapfrog_steps(self):
    return self._unrolled_leapfrog_steps

  @property
  def name(self):
    return self._name
  
  @property
  def hnn_model(self):
    return self._hnn_model
  
  @property
  def hnn_model_args(self):
    return self._hnn_model_args
  
  @property
  def target_momentum_precision_mat(self):
    return self._target_momentum_precision_mat
  
  @property
  def target_momentum_cov_mat(self):
    return self._target_momentum_cov_mat
  
  @property
  def multivariate_normal_dist(self):
    return self._multivariate_normal_dist

  @property
  def parallel_iterations(self):
    return self._parallel_iterations

  @property
  def write_instruction(self):
    return self._write_instruction

  @property
  def read_instruction(self):
    return self._read_instruction

  @property
  def parameters(self):
    return self._parameters

  @property
  def is_calibrated(self):
    return True
  
  def _load_hnn_model(self, baseline=False):
    args = self.hnn_model_args
    args.input_dim = 2 * (args.target_dim + args.aux_dim)

    if args.nn_model_name == 'mlp':
        nn_model = MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                    num_layers=args.num_layers)
    elif args.nn_model_name == 'cnn':
        nn_model = CNN_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                    num_layers=args.num_layers)
    elif args.nn_model_name == 'info':
        nn_model = Info_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                    num_layers=args.num_layers)
    elif args.nn_model_name == 'infocnn':
        nn_model = Info_CNN_MLP(args.input_dim, args.hidden_dim, args.nn_out_dim, args.nonlinearity, 
                    num_layers=args.num_layers)
    else:
        raise NotImplementedError
    
    model = HNN(args, differentiable_model=nn_model, grad_type=args.grad_type)

    folder = '{}/ckp/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_{}'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, args.nn_model_name)
    if os.path.exists(folder):
      print('Load the HNN checkpoint!')
      model.load_weights(os.path.join(folder, 'best.ckpt'))
    else:
      print('The checkpoint does not exist at {}. We just use a random initialization!'.format(folder))
    model(tf.random.normal(shape=[1, args.input_dim]))
    return model

  def one_step(self, current_state, previous_kernel_results, seed=None):
    seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
    start_trajectory_seed, loop_seed = samplers.split_seed(seed)

    with tf.name_scope(self.name + '.one_step'):
      state_structure = current_state
      current_state = tf.nest.flatten(current_state)
      if (tf.nest.is_nested(state_structure)
          and (not mcmc_util.is_list_like(state_structure)
               or len(current_state) != len(state_structure))):
        # TODO(b/170865194): Support dictionaries and other non-list-like state.
        raise TypeError('NUTS does not currently support nested or '
                        'non-list-like state structures (saw: {}).'.format(
                            state_structure))

      current_target_log_prob = previous_kernel_results.target_log_prob
      [
          init_momentum,
          init_energy,
          log_slice_sample
      ] = self._start_trajectory_batched(current_state, current_target_log_prob,
                                         seed=start_trajectory_seed)

      # can be understood as copy the tensor and stack them together
      def _copy(v):
        return v * ps.ones(
            ps.pad(
                [2], paddings=[[0, ps.rank(v)]], constant_values=1),
            dtype=v.dtype)

      initial_state = TreeDoublingState(
          momentum=init_momentum,
          state=current_state,
          target=current_target_log_prob,)
      # kind of repeats? for multiple trajectories
      initial_step_state = tf.nest.map_structure(_copy, initial_state)

      if MULTINOMIAL_SAMPLE:
        init_weight = tf.zeros_like(init_energy)  # log(exp(H0 - H0))
      else:
        init_weight = tf.ones_like(init_energy, dtype=TREE_COUNT_DTYPE)

      candidate_state = TreeDoublingStateCandidate(
          state=current_state,
          target=current_target_log_prob,
          energy=init_energy,
          weight=init_weight)

      ##################### newly added for online error monitoring ###############
      ## if 1_lf = 1 then n_lf = n_lf + 1
      num_grad_steps_count = tf.where(previous_kernel_results.num_grad_flag,
                                      previous_kernel_results.num_grad_steps_count + 1,
                                      previous_kernel_results.num_grad_steps_count)
      ## if n_lf = N_lf then 1_lf = 0
      num_grad_flag = tf.where(num_grad_steps_count == self.hnn_model_args.num_cool_down,
                               tf.zeros_like(previous_kernel_results.num_grad_flag, dtype=tf.bool),
                               previous_kernel_results.num_grad_flag)
      ## if n_lf = N_lf then n_lf = 0
      num_grad_steps_count = tf.where(num_grad_steps_count == self.hnn_model_args.num_cool_down,
                                      tf.zeros_like(previous_kernel_results.num_grad_steps_count, dtype=TREE_COUNT_DTYPE),
                                      num_grad_steps_count)
      #############################################################################

      initial_step_metastate = TreeDoublingMetaState(
          candidate_state=candidate_state,
          is_accepted=tf.zeros_like(init_energy, dtype=tf.bool),
          momentum_sum=init_momentum,
          energy_diff_sum=tf.zeros_like(init_energy),
          leapfrog_count=tf.zeros_like(init_energy, dtype=TREE_COUNT_DTYPE),
          continue_tree=tf.ones_like(init_energy, dtype=tf.bool),
          not_divergence=tf.ones_like(init_energy, dtype=tf.bool),
          num_grad_flag=num_grad_flag,
          total_num_grad_steps_count=tf.zeros_like(init_energy, dtype=TREE_COUNT_DTYPE))

      # Convert the write/read instruction into TensorArray so that it is
      # compatible with XLA.
      write_instruction = tf.TensorArray(
          TREE_COUNT_DTYPE,
          size=len(self._write_instruction),
          clear_after_read=False).unstack(self._write_instruction)
      read_instruction = tf.TensorArray(
          tf.int32,
          size=len(self._read_instruction),
          clear_after_read=False).unstack(self._read_instruction)

      current_step_meta_info = OneStepMetaInfo(
          log_slice_sample=log_slice_sample,
          init_energy=init_energy,
          write_instruction=write_instruction,
          read_instruction=read_instruction
          )

      momentum_state_memory = MomentumStateSwap(
          momentum_swap=self.init_momentum_state_memory(init_momentum),
          state_swap=self.init_momentum_state_memory(current_state))

      step_size = _prepare_step_size(
          previous_kernel_results.step_size,
          current_target_log_prob.dtype,
          len(current_state))
      
      ##########################################################################################################
      ## FLAG: main loop in NUTS: while s=1 do ...
      ## FLAG: iter_ here stands for j which is the depth of the tree
      _, _, _, new_step_metastate = tf.while_loop(
          cond=lambda iter_, seed, state, metastate: (  # pylint: disable=g-long-lambda
              (iter_ < self.max_tree_depth) &
              tf.reduce_any(metastate.continue_tree)),
          body=lambda iter_, seed, state, metastate: self._loop_tree_doubling(  # pylint: disable=g-long-lambda
              step_size,
              momentum_state_memory,
              current_step_meta_info,
              iter_,
              state,
              metastate,
              seed),
          loop_vars=(
              tf.zeros([], dtype=tf.int32, name='iter'),
              loop_seed,
              initial_step_state,
              initial_step_metastate),
          parallel_iterations=self.parallel_iterations,
      )
      ##########################################################################################################
      
      kernel_results = NUTSKernelResults(
          target_log_prob=new_step_metastate.candidate_state.target,
          step_size=previous_kernel_results.step_size,
          log_accept_ratio=tf.math.log(
              new_step_metastate.energy_diff_sum /
              tf.cast(new_step_metastate.leapfrog_count,
                      dtype=new_step_metastate.energy_diff_sum.dtype)),
          leapfrogs_taken=(
              new_step_metastate.leapfrog_count * self.unrolled_leapfrog_steps
          ),
          is_accepted=new_step_metastate.is_accepted,
          reach_max_depth=new_step_metastate.continue_tree,
          has_divergence=~new_step_metastate.not_divergence,
          energy=new_step_metastate.candidate_state.energy,
          seed=seed,
          num_grad_flag=new_step_metastate.num_grad_flag,
          num_grad_steps_count=num_grad_steps_count,
          total_num_grad_steps_count=new_step_metastate.total_num_grad_steps_count
      )

      result_state = tf.nest.pack_sequence_as(
          state_structure, new_step_metastate.candidate_state.state)
      return result_state, kernel_results

  def init_momentum_state_memory(self, input_tensors):
    """Allocate TensorArray for storing state and momentum."""
    shape_and_dtype = [(ps.shape(x_), x_.dtype) for x_ in input_tensors]
    return [  # pylint: disable=g-complex-comprehension
        ps.zeros(
            ps.concat([[max(self._write_instruction) + 1], s], axis=0),
            dtype=d) for (s, d) in shape_and_dtype
    ]

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    with tf.name_scope(self.name + '.bootstrap_results'):
      if not tf.nest.is_nested(init_state):
        init_state = [init_state]
      dummy_momentum = [tf.ones_like(state) for state in init_state]
      [
          _,
          _,
          current_target_log_prob,
      ] = strang_impl_hnn.process_args_hnn(self.target_log_prob_fn, dummy_momentum,
                                     init_state, hnn_model=self.hnn_model, target_dim=self.hnn_model_args.target_dim,
                                     aux_dim=self.hnn_model_args.aux_dim)
      # Confirm that the step size is compatible with the state parts.
      _ = _prepare_step_size(
          self.step_size, current_target_log_prob.dtype, len(init_state))

      return NUTSKernelResults(
          target_log_prob=current_target_log_prob,
          step_size=tf.nest.map_structure(
              lambda x: tf.convert_to_tensor(  # pylint: disable=g-long-lambda
                  x,
                  dtype=current_target_log_prob.dtype,
                  name='step_size'),
              self.step_size),
          log_accept_ratio=tf.zeros_like(current_target_log_prob,
                                         name='log_accept_ratio'),
          leapfrogs_taken=tf.zeros_like(current_target_log_prob,
                                        dtype=TREE_COUNT_DTYPE,
                                        name='leapfrogs_taken'),
          is_accepted=tf.zeros_like(current_target_log_prob,
                                    dtype=tf.bool,
                                    name='is_accepted'),
          reach_max_depth=tf.zeros_like(current_target_log_prob,
                                        dtype=tf.bool,
                                        name='reach_max_depth'),
          has_divergence=tf.zeros_like(current_target_log_prob,
                                       dtype=tf.bool,
                                       name='has_divergence'),
          energy=compute_hamiltonian(
              current_target_log_prob, dummy_momentum,
              shard_axis_names=self.experimental_shard_axis_names,
              precision_mat=None if self.hnn_model_args.rho_var == 1.0 else self.target_momentum_precision_mat,
              target_dim=self.hnn_model_args.target_dim,
              aux_dim=self.hnn_model_args.aux_dim),
          # Allow room for one_step's seed.
          seed=samplers.zeros_seed(),
          num_grad_flag = tf.zeros_like(current_target_log_prob,
                                        dtype=tf.bool,
                                        name='num_grad_flag'),
          num_grad_steps_count=tf.zeros_like(current_target_log_prob,
                                             dtype=TREE_COUNT_DTYPE,
                                             name='num_grad_steps_count'),
          total_num_grad_steps_count=tf.zeros_like(current_target_log_prob,
                                             dtype=TREE_COUNT_DTYPE,
                                             name='total_num_grad_steps_count')
                                             
      )

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(experimental_shard_axis_names=shard_axis_names)

  def _start_trajectory_batched(self, state, target_log_prob, seed):
    """Computations needed to start a trajectory."""
    '''
    This method
        - samples the momentum variables from the normal distrbution
        - computes the initial Hamiltonian
        - samples the u-slice sampler
    '''
    with tf.name_scope('start_trajectory_batched'):
      seeds = list(samplers.split_seed(seed, n=len(state) + 1))
      momentum_seeds = distribute_lib.fold_in_axis_index(
          seeds[:-1], self.experimental_shard_axis_names)
      ## FLAG: sample momentum variables r~N(0, I)
      ## FLAG: accommodate rho ~ N(0, M)
      if isinstance(self.hnn_model_args.rho_var, float):
        momentum = [
          tf.concat([
            samplers.normal(
              shape=[self.hnn_model_args.target_dim],
              stddev=1.0 if self.hnn_model_args.rho_var == 1.0 else math.sqrt(self.hnn_model_args.rho_var),
              dtype=x.dtype,
              seed=momentum_seeds[i]
            ),
            samplers.normal(
              shape=[self.hnn_model_args.aux_dim],
              dtype=x.dtype,
              seed=momentum_seeds[i]
            )
          ], axis=-1) for (i, x) in enumerate(state)
        ]
      else:
        momentum = [
          tf.concat([
            tf.cast(self.multivariate_normal_dist.sample(seed=momentum_seeds[i]), x.dtype),
            samplers.normal(
              shape=[self.hnn_model_args.aux_dim],
              dtype=x.dtype,
              seed=momentum_seeds[i]
            )
          ], axis=-1) for (i, x) in enumerate(state)
        ]

      init_energy = compute_hamiltonian(
          target_log_prob, momentum,
          shard_axis_names=self.experimental_shard_axis_names,
          precision_mat=None if self.hnn_model_args.rho_var == 1.0 else self.target_momentum_precision_mat,
          target_dim=self.hnn_model_args.target_dim,
          aux_dim=self.hnn_model_args.aux_dim)

      if MULTINOMIAL_SAMPLE:
        return momentum, init_energy, None

      # Draw a slice variable u ~ Uniform(0, p(initial state, initial
      # momentum)) and compute log u. For numerical stability, we perform this
      # in log space where log u = log (u' * p(...)) = log u' + log
      # p(...) and u' ~ Uniform(0, 1).
      # log(1-u)?? -- still a log uniform random variable
      
      ## FLAG: u' = log_slice_sample ~ U(0, 1)
      log_slice_sample = tf.math.log1p(-samplers.uniform(
          shape=ps.shape(init_energy),
          dtype=init_energy.dtype,
          seed=seeds[len(state)]))
      return momentum, init_energy, log_slice_sample

  def _loop_tree_doubling(self, step_size, momentum_state_memory,
                          current_step_meta_info, iter_, initial_step_state,
                          initial_step_metastate, seed):
    """Main loop for tree doubling."""
    ## FLAG: should at a specific j
    with tf.name_scope('loop_tree_doubling'):
      (direction_seed,
       subtree_seed,
       acceptance_seed,
       next_seed) = samplers.split_seed(seed, n=4)
      batch_shape = ps.shape(current_step_meta_info.init_energy)

      ## FLAG: within the while s=1 do... generate direction
      direction = tf.cast(
          samplers.uniform(
              shape=batch_shape,
              minval=0,
              maxval=2,
              dtype=tf.int32,
              seed=direction_seed),
          dtype=tf.bool)

      tree_start_states = tf.nest.map_structure(
          lambda v: bu.where_left_justified_mask(direction, v[1], v[0]),
          initial_step_state)

      directions_expanded = [
          bu.left_justified_expand_dims_like(direction, state)
          for state in tree_start_states.state
      ]

      # initialize the integrator
      # numerical gradients version
      integrator_num = strang_impl_hnn.StrangIntegrator(
          self.target_log_prob_fn,
          step_sizes=[
              # an additional step to generate direction
              tf.where(d, ss, -ss)
              for d, ss in zip(directions_expanded, step_size)
          ],
          num_steps=self.unrolled_leapfrog_steps,
          target_dim=self.hnn_model_args.target_dim,
          aux_dim=self.hnn_model_args.aux_dim,
          target_momentum_precision_mat=self.target_momentum_precision_mat) # the default value is 1
      
      # hnn version
      integrator_hnn = strang_impl_hnn.StrangIntegrator_HNN(
          self.target_log_prob_fn,
          step_sizes=[
              # an additional step to generate direction
              tf.where(d, ss, -ss)
              for d, ss in zip(directions_expanded, step_size)
          ],
          num_steps=self.unrolled_leapfrog_steps, # the default value is 1
          target_dim=self.hnn_model_args.target_dim,
          aux_dim=self.hnn_model_args.aux_dim,
          target_momentum_precision_mat=self.target_momentum_precision_mat,
          hnn_model=self.hnn_model,
          ) 

      ## FLAG: BuildTree(theta^-, r^-, u, vj, j, epsilon) or BuildTree(theta^+, r^+, u, vj, j, epsilon)
      [
          candidate_tree_state,
          tree_final_states,
          final_not_divergence,
          continue_tree_final,
          energy_diff_tree_sum,
          momentum_subtree_cumsum,
          leapfrogs_taken,
          num_grad_flag, # 1_lf returned by BuildTree in the while s = 1 loop
          total_num_grad_steps_count
      ] = self._build_sub_tree(
          directions_expanded,
          integrator_num,
          integrator_hnn,
          current_step_meta_info,
          # num_steps_at_this_depth = 2**iter_ = 1 << iter_
          tf.bitwise.left_shift(1, iter_), # 1 / 2 / 4 / 8 / 16 / ... ## nsteps
          tree_start_states,
          initial_step_metastate.continue_tree,
          initial_step_metastate.not_divergence,
          momentum_state_memory,
          seed=subtree_seed,
          num_grad_flag=initial_step_metastate.num_grad_flag)
      
      ## FLAG: initial states: theta^m = theta^{m-1}
      last_candidate_state = initial_step_metastate.candidate_state

      energy_diff_sum = (
          energy_diff_tree_sum + initial_step_metastate.energy_diff_sum)
      if MULTINOMIAL_SAMPLE:
        tree_weight = tf.where(
            continue_tree_final,
            candidate_tree_state.weight,
            tf.constant(-np.inf, dtype=candidate_tree_state.weight.dtype))
        weight_sum = generic.log_add_exp(
            tree_weight, last_candidate_state.weight)
        log_accept_thresh = tree_weight - last_candidate_state.weight
      else:
        ## FLAG: candidate_tree_state.weight should be the weight for the current sample
        ## In the explanation, it corresponds to b in MH([previous_x, current x], weight=[a / b])
        tree_weight = tf.where(
            continue_tree_final,
            candidate_tree_state.weight,
            tf.zeros([], dtype=TREE_COUNT_DTYPE))
        weight_sum = tree_weight + last_candidate_state.weight
        # since log_slice_sampler is also in log u form so here the threshold is also in log form
        log_accept_thresh = tf.math.log(
            tf.cast(tree_weight, tf.float32) /
            tf.cast(last_candidate_state.weight, tf.float32))
      log_accept_thresh = tf.where(
          tf.math.is_nan(log_accept_thresh),
          tf.zeros([], log_accept_thresh.dtype),
          log_accept_thresh)
      u = tf.math.log1p(-samplers.uniform(
          shape=batch_shape,
          dtype=log_accept_thresh.dtype,
          seed=acceptance_seed))
      
      ## FLAG: this is the MH step after each tree doubling
      ## FLAG: with probability min{1, nprime(tree weight) / n(last_cadidate_state_weight weight)}
      ## logu <= log(tree_weight / last_candidate_state.weight)
      is_sample_accepted = u <= log_accept_thresh

      ## FLAG: need to satisfy two conditions: the probability and continue_tree_final
      choose_new_state = is_sample_accepted & continue_tree_final

      ## FLAG: update samples: theta^m <- thetaprime
      new_candidate_state = TreeDoublingStateCandidate(
          state=[
              bu.where_left_justified_mask(choose_new_state, s0, s1)
              for s0, s1 in zip(candidate_tree_state.state,
                                last_candidate_state.state)
          ],
          target=bu.where_left_justified_mask(
              choose_new_state,
              candidate_tree_state.target,
              last_candidate_state.target),
          energy=bu.where_left_justified_mask(
              choose_new_state,
              candidate_tree_state.energy,
              last_candidate_state.energy),
          weight=weight_sum)

      for new_candidate_state_temp, old_candidate_state_temp in zip(
          new_candidate_state.state, last_candidate_state.state):
        tensorshape_util.set_shape(new_candidate_state_temp,
                                   old_candidate_state_temp.shape)

      # Update left right information of the trajectory, and check trajectory
      # level U turn
      tree_otherend_states = tf.nest.map_structure(
          lambda v: bu.where_left_justified_mask(direction, v[0], v[1]),
          initial_step_state)

      ## FLAG: should be updating the boundary state (the leftmost and the rightmost)
      new_step_state = tf.nest.pack_sequence_as(initial_step_state, [
          tf.stack([  # pylint: disable=g-complex-comprehension
              bu.where_left_justified_mask(direction, right, left),
              bu.where_left_justified_mask(direction, left, right),
          ], axis=0)
          for left, right in zip(tf.nest.flatten(tree_final_states),
                                 tf.nest.flatten(tree_otherend_states))
      ])

      momentum_tree_cumsum = []
      for p0, p1 in zip(
          initial_step_metastate.momentum_sum, momentum_subtree_cumsum):
        momentum_part_temp = p0 + p1
        tensorshape_util.set_shape(momentum_part_temp, p0.shape)
        momentum_tree_cumsum.append(momentum_part_temp)

      for new_state_temp, old_state_temp in zip(
          tf.nest.flatten(new_step_state),
          tf.nest.flatten(initial_step_state)):
        tensorshape_util.set_shape(new_state_temp, old_state_temp.shape)

      if GENERALIZED_UTURN:
        state_diff = momentum_tree_cumsum
      else:
        ## FLAG: calculate theta^+ - theta^-
        state_diff = [s[1] - s[0] for s in new_step_state.state]

      ## FLAG: check whether there is u-turn
      no_u_turns_trajectory = has_not_u_turn(
          state_diff,
          [m[0] for m in new_step_state.momentum],  ## FLAG: r^-
          [m[1] for m in new_step_state.momentum],  ## FLAG: r^+
          log_prob_rank=ps.rank_from_shape(batch_shape),
          shard_axis_names=self.experimental_shard_axis_names)

      new_step_metastate = TreeDoublingMetaState(
          candidate_state=new_candidate_state,
          is_accepted=choose_new_state | initial_step_metastate.is_accepted, ## FLAG: whether accept the new sample
          momentum_sum=momentum_tree_cumsum,
          energy_diff_sum=energy_diff_sum,
          continue_tree=continue_tree_final & no_u_turns_trajectory, ## FLAG: sprime
          not_divergence=final_not_divergence,
          leapfrog_count=(initial_step_metastate.leapfrog_count +
                          leapfrogs_taken),
          num_grad_flag=num_grad_flag,
          total_num_grad_steps_count=(initial_step_metastate.total_num_grad_steps_count + total_num_grad_steps_count))

      return iter_ + 1, next_seed, new_step_state, new_step_metastate ## FLAG: j+=1

  ## FLAG: build tree wrapper - for a specific j
  def _build_sub_tree(self,
                      directions,
                      integrator_num,
                      integrator_hnn,
                      current_step_meta_info,
                      nsteps,
                      initial_state,
                      continue_tree,
                      not_divergence,
                      momentum_state_memory,
                      seed,
                      num_grad_flag,
                      name=None):
    with tf.name_scope('build_sub_tree'):
      batch_shape = ps.shape(current_step_meta_info.init_energy)
      # We never want to select the initial state
      if MULTINOMIAL_SAMPLE:
        init_weight = tf.fill(
            batch_shape,
            tf.constant(-np.inf,
                        dtype=current_step_meta_info.init_energy.dtype))
      else:
        init_weight = tf.zeros(batch_shape, dtype=TREE_COUNT_DTYPE)

      init_momentum_cumsum = [tf.zeros_like(x) for x in initial_state.momentum]
      ## FLAG: this should be initial states
      ## FLAG: this should be the so-called x_ defined in the explaination (weight = 0)
      initial_state_candidate = TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          energy=initial_state.target,
          weight=init_weight)
      energy_diff_sum = tf.zeros_like(current_step_meta_info.init_energy,
                                      name='energy_diff_sum')
      
      #######################################################################################################
      ## FLAG: Here is unrolling NUTS instead of the recursive version, that is why it uses while loop
      [
          _,
          _,
          energy_diff_tree_sum,
          momentum_tree_cumsum,
          leapfrogs_taken,
          final_state,
          candidate_tree_state,
          final_continue_tree,
          final_not_divergence,
          momentum_state_memory,
          num_grad_flag,
          total_num_grad_steps_count
      ] = tf.while_loop(
          cond=lambda iter_, seed, energy_diff_sum, init_momentum_cumsum,  # pylint: disable=g-long-lambda
                      leapfrogs_taken, state, state_c, continue_tree,
                      not_divergence, momentum_state_memory, num_grad_flag, total_num_grad_steps_count: (
                          (iter_ < nsteps) & tf.reduce_any(continue_tree)),
          body=lambda iter_, seed, energy_diff_sum, init_momentum_cumsum,  # pylint: disable=g-long-lambda
                      leapfrogs_taken, state, state_c, continue_tree,
                      not_divergence, momentum_state_memory, num_grad_flag, total_num_grad_steps_count: (
                          self._loop_build_sub_tree(
                              directions, integrator_num, integrator_hnn, current_step_meta_info,
                              iter_, energy_diff_sum, init_momentum_cumsum,
                              leapfrogs_taken, state, state_c, continue_tree,
                              not_divergence, momentum_state_memory, seed, num_grad_flag, total_num_grad_steps_count)),
          loop_vars=(
              tf.zeros([], dtype=tf.int32, name='iter'),
              seed,
              energy_diff_sum,
              init_momentum_cumsum,
              tf.zeros(batch_shape, dtype=TREE_COUNT_DTYPE),
              initial_state,
              initial_state_candidate,
              continue_tree,
              not_divergence,
              momentum_state_memory,
              num_grad_flag,
              tf.zeros(batch_shape, dtype=TREE_COUNT_DTYPE) # total_num_grad_steps_count
          ),
          parallel_iterations=self.parallel_iterations
      )
      #######################################################################################################

    return (
        candidate_tree_state,
        final_state,
        final_not_divergence,
        final_continue_tree,
        energy_diff_tree_sum,
        momentum_tree_cumsum,
        leapfrogs_taken,
        num_grad_flag,
        total_num_grad_steps_count
    )

  ## FLAG: build tree in each step (one leapfrog step)
  def _loop_build_sub_tree(self,
                           directions,
                           integrator_num,
                           integrator_hnn,
                           current_step_meta_info,
                           iter_,
                           energy_diff_sum_previous,
                           momentum_cumsum_previous,
                           leapfrogs_taken,
                           prev_tree_state,
                           candidate_tree_state,
                           continue_tree_previous,
                           not_divergent_previous,
                           momentum_state_memory,
                           seed,
                           num_grad_flag,
                           total_num_grad_steps_count):
    """Base case in tree doubling."""
    acceptance_seed, next_seed = samplers.split_seed(seed)
    with tf.name_scope('loop_build_sub_tree'):
      ##################################################################################
      # FLAG: Take one leapfrog step in the direction v and check divergence
      # Questions: how to calculate target grad parts -- should we modify it?
      [
          next_momentum_parts,
          next_state_parts,
          next_target
      ] = integrator_hnn(prev_tree_state.momentum,
                         prev_tree_state.state,
                         prev_tree_state.target)
      ##################################################################################
      
      ####################### newly added for online monitor ###########################
      energy = compute_hamiltonian(
          next_target, next_momentum_parts,
          shard_axis_names=self.experimental_shard_axis_names,
          precision_mat=None if self.hnn_model_args.rho_var == 1.0 else self.target_momentum_precision_mat,
          target_dim=self.hnn_model_args.target_dim,
          aux_dim=self.hnn_model_args.aux_dim)
      current_energy = tf.where(tf.math.is_nan(energy),
                                tf.constant(-np.inf, dtype=energy.dtype),
                                energy)
      # FLAG: -joint + joint0
      energy_diff = current_energy - current_step_meta_info.init_energy

      if MULTINOMIAL_SAMPLE:
        not_divergent_hnn = -energy_diff < self.hnn_model_args.hnn_threshold
      else:
        log_slice_sample = current_step_meta_info.log_slice_sample
        # FLAG: logu + joint - joint0 < delta --> logu - joint0 - delta < -joint -- this is sprime
        not_divergent_hnn = log_slice_sample - energy_diff < self.hnn_model_args.hnn_threshold

      num_grad_flag = num_grad_flag | tf.logical_not(not_divergent_hnn)
  
      if tf.reduce_any(num_grad_flag):
        # print('next_momentum_parts before', next_momentum_parts)
        # print('next_state_parts before', next_state_parts)
        # print('next_target before', next_target)
        # print('next_target_grad_parts before', next_target_grad_parts)
        # print('num_grad_flag before', num_grad_flag)
        # print('numerical gradient is called')
        [
            next_momentum_parts_num,
            next_state_parts_num,
            next_target_num
        ] = integrator_num(prev_tree_state.momentum,
                          prev_tree_state.state,
                          prev_tree_state.target)
        assert tf.reduce_all(tf.equal(next_momentum_parts[0].shape[:-1], num_grad_flag.shape)), \
               'the batch dimension of states {} does not match the shape of num_grad_flag {}'.format(next_momentum_parts[0].shape[:-1], num_grad_flag.shape)
        next_momentum_parts = [tf.where(tf.expand_dims(num_grad_flag, axis=-1), momentum_num, momentum)
                               for momentum_num, momentum in zip(next_momentum_parts_num, next_momentum_parts)]
        next_state_parts = [tf.where(tf.expand_dims(num_grad_flag, axis=-1), state_num, state)
                            for state_num, state in zip(next_state_parts_num, next_state_parts)]
        next_target = tf.where(num_grad_flag, next_target_num, next_target)
        
        # print('next_momentum_parts after', next_momentum_parts)
        # print('next_state_parts after', next_state_parts)
        # print('next_target after', next_target)
        # print('next_target_grad_parts after', next_target_grad_parts)

        total_num_grad_steps_count += tf.reduce_sum(tf.cast(tf.where(continue_tree_previous, num_grad_flag, 
                                                        tf.zeros_like(num_grad_flag, dtype=tf.bool)), dtype=TREE_COUNT_DTYPE))
      ##################################################################################

      next_tree_state = TreeDoublingState(
          momentum=next_momentum_parts,
          state=next_state_parts,
          target=next_target)
      momentum_cumsum = [p0 + p1 for p0, p1 in zip(momentum_cumsum_previous,
                                                   next_momentum_parts)]
      # If the tree have not yet terminated previously, we count this leapfrog.
      leapfrogs_taken = tf.where(
          continue_tree_previous, leapfrogs_taken + 1, leapfrogs_taken)

      write_instruction = current_step_meta_info.write_instruction
      read_instruction = current_step_meta_info.read_instruction
      init_energy = current_step_meta_info.init_energy

      if GENERALIZED_UTURN:
        state_to_write = momentum_cumsum_previous
        state_to_check = momentum_cumsum
      else:
        state_to_write = next_state_parts
        state_to_check = next_state_parts

      batch_shape = ps.shape(next_target)
      has_not_u_turn_init = ps.ones(batch_shape, dtype=tf.bool)

      read_index = read_instruction.gather([iter_])[0]
      no_u_turns_within_tree = has_not_u_turn_at_all_index(  # pylint: disable=g-long-lambda
          read_index,
          directions,
          momentum_state_memory,
          next_momentum_parts,
          state_to_check,
          has_not_u_turn_init,
          log_prob_rank=ps.rank(next_target),
          shard_axis_names=self.experimental_shard_axis_names)

      # Get index to write state into memory swap
      write_index = write_instruction.gather([iter_])
      momentum_state_memory = MomentumStateSwap(
          momentum_swap=[
              _safe_tensor_scatter_nd_update(old, [write_index], [new])
              for old, new in zip(momentum_state_memory.momentum_swap,
                                  next_momentum_parts)
          ],
          state_swap=[
              _safe_tensor_scatter_nd_update(old, [write_index], [new])
              for old, new in zip(momentum_state_memory.state_swap,
                                  state_to_write)
          ])

      energy = compute_hamiltonian(
          next_target, next_momentum_parts,
          shard_axis_names=self.experimental_shard_axis_names,
          precision_mat=None if self.hnn_model_args.rho_var == 1.0 else self.target_momentum_precision_mat,
          target_dim=self.hnn_model_args.target_dim,
          aux_dim=self.hnn_model_args.aux_dim)
      current_energy = tf.where(tf.math.is_nan(energy),
                                tf.constant(-np.inf, dtype=energy.dtype),
                                energy)
      # FLAG: -joint + joint0
      energy_diff = current_energy - init_energy

      if MULTINOMIAL_SAMPLE:
        not_divergent = -energy_diff < self.max_energy_diff
        weight_sum = generic.log_add_exp(
            candidate_tree_state.weight, energy_diff)
        log_accept_thresh = energy_diff - weight_sum
      else:
        log_slice_sample = current_step_meta_info.log_slice_sample
        # FLAG: logu + joint - joint0 < delta --> logu - joint0 - delta < -joint -- this is sprime
        # FLAG: self.max_energy_diff == self.hnn_model_args.lf_threshold
        # TODO: handle max_energy_diff = lf_threshold / hnn_threshold
        not_divergent = log_slice_sample - energy_diff < self.max_energy_diff
        # Uniform sampling on the trajectory within the subtree across valid
        # samples.
        # FLAG: logu <= -joint + joint0 -> logu - joint0 <= -joint -- this is nprime
        is_valid = log_slice_sample <= energy_diff
        weight_sum = tf.where(is_valid,
                              candidate_tree_state.weight + 1,
                              candidate_tree_state.weight)
        log_accept_thresh = tf.where(
            is_valid,
            -tf.math.log(tf.cast(weight_sum, dtype=tf.float32)),
            tf.constant(-np.inf, dtype=tf.float32))
      u = tf.math.log1p(-samplers.uniform(
          shape=batch_shape,
          dtype=log_accept_thresh.dtype,
          seed=acceptance_seed))
      
      # FLAG: this is the uniform sampling performed after each leapfrog operation
      # The Uniform sampling is performed as U([x_previous, current_x], weight=[weight_sum-1, 1(if valid else 0)]
      # Hence with probability 1 / weightsum, will choose the new one --> sample logu < log(weightsum^{-1})
      is_sample_accepted = u <= log_accept_thresh

      next_candidate_tree_state = TreeDoublingStateCandidate(
          state=[
              bu.where_left_justified_mask(is_sample_accepted, s0, s1)
              for s0, s1 in zip(next_state_parts, candidate_tree_state.state)
          ],
          target=bu.where_left_justified_mask(
              is_sample_accepted, next_target, candidate_tree_state.target),
          energy=bu.where_left_justified_mask(
              is_sample_accepted,
              current_energy,
              candidate_tree_state.energy),
          weight=weight_sum)

      ## FLAG: sprime = sprime x previous sprime
      continue_tree = not_divergent & continue_tree_previous
      ## FLAG: sprime = sprime x has no u-turn check
      continue_tree_next = no_u_turns_within_tree & continue_tree

      not_divergent_tokeep = tf.where(
          continue_tree_previous,
          not_divergent,
          ps.ones(batch_shape, dtype=tf.bool))

      # min(1., exp(energy_diff)).
      # energy_diff = -joint + joint0
      exp_energy_diff = tf.math.exp(tf.minimum(energy_diff, 0.))
      energy_diff_sum = tf.where(continue_tree,
                                 energy_diff_sum_previous + exp_energy_diff,
                                 energy_diff_sum_previous)

      return (
          iter_ + 1,
          next_seed,
          energy_diff_sum,
          momentum_cumsum,
          leapfrogs_taken,
          next_tree_state,
          next_candidate_tree_state,
          continue_tree_next,
          not_divergent_previous & not_divergent_tokeep,
          momentum_state_memory,
          num_grad_flag,
          total_num_grad_steps_count
      )


def has_not_u_turn_at_all_index(read_indexes, direction, momentum_state_memory,
                                momentum_right, state_right,
                                no_u_turns_within_tree, log_prob_rank,
                                shard_axis_names=None):
  """Check u turn for early stopping."""

  def _get_left_state_and_check_u_turn(left_current_index, no_u_turns_last):
    """Check U turn on a single index."""
    momentum_left = [
        tf.gather(x, left_current_index, axis=0)
        for x in momentum_state_memory.momentum_swap
    ]
    state_left = [
        tf.gather(x, left_current_index, axis=0)
        for x in momentum_state_memory.state_swap
    ]
    # Note that in generalized u turn, state_diff is actually the cumulated sum
    # of the momentum.
    state_diff = [s1 - s2 for s1, s2 in zip(state_right, state_left)]
    if not GENERALIZED_UTURN:
      state_diff = [tf.where(d, m, -m) for d, m in zip(direction, state_diff)]

    no_u_turns_current = has_not_u_turn(
        state_diff,
        momentum_left,
        momentum_right,
        log_prob_rank,
        shard_axis_names=shard_axis_names)
    return left_current_index + 1, no_u_turns_current & no_u_turns_last

  _, no_u_turns_within_tree = tf.while_loop(
      cond=lambda i, no_u_turn: ((i < read_indexes[1]) &  # pylint: disable=g-long-lambda
                                 tf.reduce_any(no_u_turn)),
      body=_get_left_state_and_check_u_turn,
      loop_vars=(read_indexes[0], no_u_turns_within_tree))
  return no_u_turns_within_tree

## FLAG to check whether there is u-turn; return True if it has no u-turn.
def has_not_u_turn(state_diff,
                   momentum_left,
                   momentum_right,
                   log_prob_rank,
                   shard_axis_names=None):
  """If the trajectory does not exhibit a U-turn pattern."""
  shard_axis_names = (shard_axis_names or ([None] * len(state_diff)))
  def reduce_sum(x, m, shard_axes):
    out = tf.reduce_sum(x, axis=ps.range(log_prob_rank, ps.rank(m)))
    if shard_axes is not None:
      out = distribute_lib.psum(out, shard_axes)
    return out
  with tf.name_scope('has_not_u_turn'):
    batch_dot_product_left = sum(
        reduce_sum(s_diff * m, m, axes)
        for s_diff, m, axes in zip(state_diff, momentum_left,
                                   shard_axis_names)
    )
    batch_dot_product_right = sum(
        reduce_sum(s_diff * m, m, axes)
        for s_diff, m, axes in zip(state_diff, momentum_right,
                                   shard_axis_names)
    )
    return (batch_dot_product_left >= 0) & (batch_dot_product_right >= 0)


def build_tree_uturn_instruction(max_depth, init_memory=0):
  """Run build tree and output the u turn checking input instruction."""

  def _buildtree(address, depth):
    if depth == 0:
      address += 1
      return address, address
    else:
      address_left, address_right = _buildtree(address, depth - 1)
      _, address_right = _buildtree(address_right, depth - 1)
      instruction.append((address_left, address_right))
      return address_left, address_right

  instruction = []
  _, _ = _buildtree(init_memory, max_depth)
  # return unique pairs of addresses (indices) representing the tree structure
  # Question: why it only contains one direction?
  return np.unique(np.array(instruction, dtype=np.int32), axis=0)


def generate_efficient_write_read_instruction(instruction_array):
  """Statically generate a memory efficient write/read instruction."""
  nsteps_within_tree = np.max(instruction_array) + 1
  instruction_mat = np.zeros((nsteps_within_tree, nsteps_within_tree))
  for previous_step, current_step in instruction_array:
    instruction_mat[previous_step, current_step] = 1

  # Generate a sparse matrix that represents the memory footprint:
  #   -1 : no need to save to memory (these are odd steps)
  #    1 : needed for check u turn (either already in memory or will be saved)
  #    0 : still in memory but not needed for check u turn
  for i in range(nsteps_within_tree):
    temp = instruction_mat[i]
    endpoint = np.where(temp == 1)[0]
    if endpoint.size > 0:
      temp[:i] = -1
      temp[endpoint[-1]+1:] = -1
      instruction_mat[i] = temp
    else:
      instruction_mat[i] = -1

  # In the classical U-turn check, the writing is only at odd step and the
  # instruction follows squence A000120 (https://oeis.org/A000120)
  to_write_temp = np.sum(instruction_mat > -1, axis=0)
  write_instruction = to_write_temp - 1
  write_instruction[np.diag(instruction_mat) == -1] = max(to_write_temp)

  read_instruction = []
  for i in range(nsteps_within_tree):
    temp_instruction = instruction_mat[:, i]
    if np.sum(temp_instruction == 1) > 0:
      r = np.where(temp_instruction[temp_instruction >= 0] == 1)[0][0]
      read_instruction.append([r, r + np.sum(temp_instruction == 1)])
    else:
      # If there is no instruction to do U turn check (e.g., odd step in the
      # original U turn check scheme), we append a pair of 0s as instruction.
      # In the inner most while loop of build tree, we loop through the read
      # instruction to check U turn - looping from 0 to 0 works with the
      # existing code while no computation happens.
      read_instruction.append([0, 0])
  return write_instruction, np.asarray(read_instruction)


def _prepare_step_size(step_size, dtype, n_state_parts):
  step_sizes, _ = mcmc_util.prepare_state_parts(
      step_size, dtype=dtype, name='step_size')
  if len(step_sizes) == 1:
    step_sizes *= n_state_parts
  if n_state_parts != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  return step_sizes

def compute_hamiltonian(target_log_prob, momentum_parts, precision_mat=None, shard_axis_names=None, target_dim=None, aux_dim=None):
    """
    Compute the Hamiltonian of the current system, adapting to non-identity covariance for momentum.

    Args:
        target_log_prob: The log probability of the target distribution (potential energy).
        momentum_parts: A list of tensors representing the momentum variables.
        precision_mat: A precision matrix for the momentum parts rho. If None, assumes identity.
        shard_axis_names: (Optional) A list of axis names for sharding.

    Returns:
        The Hamiltonian of the system.
    """
    shard_axis_names = shard_axis_names or ([None] * len(momentum_parts))
    independent_chain_ndims = ps.rank(target_log_prob)

    def compute_quadratic_form(v, precision_mat, shard_axes):
        """
        Compute the quadratic form v^T cov^{-1} v.
        """
        if precision_mat is None:
            # If no covariance matrix is provided, assume identity (sum of squares)
            sum_sq = tf.reduce_sum(v ** 2., axis=ps.range(independent_chain_ndims, ps.rank(v)))
        else:
            # Compute v^T cov^{-1} v
            v_flat = tf.reshape(v, [-1, tf.shape(v)[-1]])  # Flatten for matrix multiplication
            quadratic_form = tf.reduce_sum(v_flat * tf.matmul(v_flat, precision_mat), axis=-1)
            sum_sq = tf.reshape(quadratic_form, tf.shape(v)[:-1])  # Reshape back to original shape

        if shard_axes is not None:
            sum_sq = distribute_lib.psum(sum_sq, shard_axes)
        return sum_sq

    rho_parts = []
    p_parts = []
    for m in momentum_parts:
      x, y = tf.split(m, [target_dim, aux_dim])
      rho_parts.append(x)
      p_parts.append(y)

    # Compute the kinetic energy terms for each momentum part
    kinetic_energy_rho_parts = (
        tf.cast(  # Cast to the same dtype as target_log_prob
            compute_quadratic_form(m, precision_mat, axes),
            dtype=target_log_prob.dtype)
        for m, axes in zip(rho_parts, shard_axis_names))
    
    # Compute the kinetic energy terms for each momentum part
    kinetic_energy_p_parts = (
        tf.cast(  # Cast to the same dtype as target_log_prob
            compute_quadratic_form(m, None, axes),
            dtype=target_log_prob.dtype)
        for m, axes in zip(p_parts, shard_axis_names))

    # Hamiltonian is the sum of potential energy (target_log_prob) and kinetic energy
    return target_log_prob - 0.5 * sum(kinetic_energy_rho_parts) - 0.5 * sum(kinetic_energy_p_parts)


def _safe_tensor_scatter_nd_update(tensor, indices, updates):
  if tensorshape_util.num_elements(tensor.shape) == 0:
    return tensor
  return tf.tensor_scatter_nd_update(tensor, indices, updates)