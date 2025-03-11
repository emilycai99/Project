# Copyright 2018 The TensorFlow Probability Authors.
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
"""Defines the LeapfrogIntegrator class."""
import abc
import six

import tensorflow as tf

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'LeapfrogIntegrator',
    'SimpleLeapfrogIntegrator_HNN',
    'process_args',
]


@six.add_metaclass(abc.ABCMeta)
class LeapfrogIntegrator(object):
  """Base class for all leapfrog integrators.

  [Leapfrog integrators](https://en.wikipedia.org/wiki/Leapfrog_integration)
  numerically integrate differential equations of the form:

  ```none
  v' = dv/dt = F(x)
  x' = dx/dt = v
  ```

  This class defines minimal requirements for leapfrog integration calculations.
  """

  @abc.abstractmethod
  def __call__(self, momentum_parts, state_parts, target=None,
               target_grad_parts=None, kinetic_energy_fn=None, name=None):
    """Computes the integration.

    Args:
      momentum_parts: Python `list` of `Tensor`s representing momentum for each
        state part.
      state_parts: Python `list` of `Tensor`s which collectively representing
        the state.
      target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `state_parts`.
      target_grad_parts: Python `list` of `Tensor`s representing the gradient of
        `target` with respect to each of `state_parts`.
      kinetic_energy_fn: Python callable that can evaluate the kinetic energy
        of the given momentum.
      name: Python `str` used to group ops created by this function.

    Returns:
      next_momentum_parts: Python `list` of `Tensor`s representing new momentum.
      next_state_parts: Python `list` of `Tensor`s which collectively
        representing the new state.
      next_target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `next_state_parts`.
      next_target_grad_parts: Python `list` of `Tensor`s representing the
        gradient of `next_target` with respect to each of `next_state_parts`.
    """
    raise NotImplementedError('Integrate logic not implemented.')


class SimpleLeapfrogIntegrator_HNN(LeapfrogIntegrator):
  # pylint: disable=line-too-long
  """Simple leapfrog integrator.

  Calling this functor is conceptually equivalent to:

  ```none
  def leapfrog(x, v, eps, L, f, M):
    g = lambda x: gradient(f, x)
    v[0] = v + eps/2 g(x)
    for l = 1...L:
      x[l] = x[l-1] + eps * inv(M) @ v[l-1]
      v[l] = v[l-1] + eps * g(x[l])
    v = v[L] - eps/2 * g(x[L])
    return x[L], v
  ```

  where `M = eye(dims(x))`.
  (In the future we may support arbitrary covariance `M`.)

  #### Examples:

  ```python
  import matplotlib.pyplot as plt
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl
  dims = 10
  dtype = tf.float32

  target_fn = tfp.distributions.MultivariateNormalDiag(
      loc=tf.zeros(dims, dtype)).log_prob

  integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
      target_fn,
      step_sizes=[0.1],
      num_steps=3)

  momentum = [tf.random.normal([dims], dtype=dtype)]
  position = [tf.random.normal([dims], dtype=dtype)]
  target = None
  target_grad_parts = None

  num_iter = int(1e3)
  positions = tf.zeros([num_iter, dims], dtype)
  for i in range(num_iter):
    [momentum, position, target, target_grad_parts] = integrator(
        momentum, position, target, target_grad_parts)
    positions = tf.tensor_scatter_nd_update(positions, [[i]], position)

  plt.plot(positions[:, 0]);  # Sinusoidal.
  ```

  """
  # pylint: enable=line-too-long

  def __init__(self, target_fn, step_sizes, num_steps, hnn_model, input_dim=2):
    """Constructs the LeapfrogIntegrator.

    Assumes a simple quadratic kinetic energy function: `0.5 ||momentum||**2`.

    Args:
      target_fn: Python callable which takes an argument like `*state_parts` and
        returns its (possibly unnormalized) log-density under the target
        distribution.
      step_sizes: Python `list` of `Tensor`s representing the step size for the
        leapfrog integrator. Must broadcast with the shape of
        `current_state_parts`.  Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_steps: `int` `Tensor` representing  number of steps to run
        the leapfrog integration. Total progress is roughly proportional to
        `step_size * num_steps`.
    """
    # Note on per-variable step sizes:
    #
    # Using per-variable step sizes is equivalent to using the same step
    # size for all variables and adding a diagonal mass matrix in the
    # kinetic energy term of the Hamiltonian being integrated. This is
    # hinted at by Neal (2011) but not derived in detail there.
    #
    # Let x and v be position and momentum variables respectively.
    # Let g(x) be the gradient of `target_fn(x)`.
    # Let S be a diagonal matrix of per-variable step sizes.
    # Let the Hamiltonian H(x, v) = -target_fn(x) + 0.5 * ||v||**2.
    #
    # Using per-variable step sizes gives the updates:
    #
    #   v' = v0 + 0.5 * S @ g(x0)
    #   x1 = x0 + S @ v'
    #   v1 = v' + 0.5 * S @ g(x1)
    #
    # Let,
    #
    #   u = inv(S) @ v
    #
    # for "u'", "u0", and "u1". Multiplying v by inv(S) in the updates above
    # gives the transformed dynamics:
    #
    #   u' = inv(S) @ v'
    #      = inv(S) @ v0 + 0.5 * g(x)
    #      = u0 + 0.5 * g(x)
    #
    #   x1 = x0 + S @ v'
    #      = x0 + S @ S @ u'
    #
    #   u1 = inv(S) @ v1
    #      = inv(S) @ v' + 0.5 * g(x1)
    #      = u' + 0.5 * g(x1)
    #
    # These are exactly the leapfrog updates for the Hamiltonian
    #
    #   H'(x, u) = -target_fn(x) + 0.5 * (S @ u).T @ (S @ u)
    #            = -target_fn(x) + 0.5 * ||v||**2
    #            = H(x, v).
    #
    # To summarize:
    #
    # * Using per-variable step sizes implicitly simulates the dynamics
    #   of the Hamiltonian H' (which are energy-conserving in H'). We
    #   keep track of v instead of u, but the underlying dynamics are
    #   the same if we transform back.
    # * The value of the Hamiltonian H'(x, u) is the same as the value
    #   of the original Hamiltonian H(x, v) after we transform back from
    #   u to v.
    # * Sampling v ~ N(0, I) is equivalent to sampling u ~ N(0, S**-2).
    #
    # So using per-variable step sizes in HMC will give results that are
    # exactly identical to explicitly using a diagonal mass matrix.
    self._target_fn = target_fn
    self._step_sizes = step_sizes
    self._num_steps = num_steps
    self._hnn_model = hnn_model
    self._input_dim = input_dim

  @property
  def target_fn(self):
    return self._target_fn

  @property
  def step_sizes(self):
    return self._step_sizes

  @property
  def num_steps(self):
    return self._num_steps
  
  @property
  def hnn_model(self):
    return self._hnn_model
  
  @property
  def input_dim(self):
    return self._input_dim

  def __call__(self,
               momentum_parts,
               state_parts,
               target=None,
               target_grad_parts=None,
               kinetic_energy_fn=None,
               name=None):
    """Applies `num_steps` of the leapfrog integrator.

    Args:
      momentum_parts: Python `list` of `Tensor`s representing momentum for each
        state part.
      state_parts: Python `list` of `Tensor`s which collectively representing
        the state.
      target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `state_parts`.
      target_grad_parts: Python `list` of `Tensor`s representing the gradient of
        `target` with respect to each of `state_parts`.
      kinetic_energy_fn: Python callable that can evaluate the kinetic energy
        of the given momentum. This is typically the negative log probability of
        the distribution over the momentum.
      name: Python `str` used to group ops created by this function.

    Returns:
      next_momentum_parts: Python `list` of `Tensor`s representing new momentum.
      next_state_parts: Python `list` of `Tensor`s which collectively
        representing the new state.
      next_target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `next_state_parts`.
      next_target_grad_parts: Python `list` of `Tensor`s representing the
        gradient of `next_target` with respect to each of `next_state_parts`.
    """
    with tf.name_scope(name or 'leapfrog_integrate'):
      [
          momentum_parts,
          state_parts,
          target,
          target_grad_parts,
      ] = process_args(
          self.target_fn,
          momentum_parts,
          state_parts,
          target,
          target_grad_parts,
          hnn_model=self.hnn_model,
          input_dim=self.input_dim)

      if kinetic_energy_fn is None:
        # Avoid adding ops and taking grads, when the implied kinetic energy
        # is just 0.5 * ||x||^2, so the gradient is x
        get_velocity_parts = lambda x: x
      else:
        def get_velocity_parts(half_next_momentum_parts):
          _, velocity_parts = mcmc_util.maybe_call_fn_and_grads(
              kinetic_energy_fn, half_next_momentum_parts)
          return velocity_parts

      # See Algorithm 1 of "Faster Hamiltonian Monte Carlo by Learning Leapfrog
      # Scale", https://arxiv.org/abs/1810.04449.
      
      ## FLAG: target_grad_parts should be the gradient of L with respect to theta
      # print('target_grad_parts', target_grad_parts)
      half_next_momentum_parts = [
          v + _multiply(0.5 * eps, g, dtype=v.dtype)
          for v, eps, g
          in zip(momentum_parts, self.step_sizes, target_grad_parts)]
      [
          _,
          next_half_next_momentum_parts,
          next_state_parts,
          next_target,
          next_target_grad_parts,
      ] = tf.while_loop(
          cond=lambda i, *_: i < self.num_steps,
          body=lambda i, *args: [i + 1] + list(_one_step(  # pylint: disable=no-value-for-parameter,g-long-lambda
              self.target_fn, self.step_sizes, get_velocity_parts, *args, hnn_model=self.hnn_model, input_dim=self.input_dim)),
          loop_vars=[
              tf.zeros_like(self.num_steps, name='iter'),
              half_next_momentum_parts,
              state_parts,
              target,
              target_grad_parts,
          ])
      
      ## FLAG: the third equation, next_target_grad_parts is the gradient evaluated at next-step theta
      ## Questions: why here is the minus sign??? Solved because it first + eps * g in the loop
      next_momentum_parts = [
          v - _multiply(0.5 * eps, g, dtype=v.dtype)  # pylint: disable=g-complex-comprehension
          for v, eps, g
          in zip(next_half_next_momentum_parts,
                 self.step_sizes,
                 next_target_grad_parts)
      ]

      return (
          next_momentum_parts,
          next_state_parts,
          next_target,
          next_target_grad_parts,
      )


def _one_step(
    target_fn,
    step_sizes,
    get_velocity_parts,
    half_next_momentum_parts,
    state_parts,
    target,
    target_grad_parts,
    hnn_model=None,
    input_dim=2):
  """Body of integrator while loop."""
  with tf.name_scope('leapfrog_integrate_one_step'):
    
    # gradients for the momentum variables
    velocity_parts = get_velocity_parts(half_next_momentum_parts)
    next_state_parts = []
    for state_part, eps, velocity_part in zip(
        state_parts, step_sizes, velocity_parts):
      next_state_parts.append(
          state_part + _multiply(eps, velocity_part, dtype=state_part.dtype))

    [next_target, next_target_grad_parts] = hnn_fn_and_grads(
        target_fn, next_state_parts, half_next_momentum_parts, hnn_model, input_dim=input_dim)
    # print('next_target', next_target)
    # print('next_target_grad_parts', next_target_grad_parts)
    if any(g is None for g in next_target_grad_parts):
      raise ValueError(
          'Encountered `None` gradient.\n'
          '  state_parts: {}\n'
          '  next_state_parts: {}\n'
          '  next_target_grad_parts: {}'.format(
              state_parts,
              next_state_parts,
              next_target_grad_parts))

    tensorshape_util.set_shape(next_target, target.shape)
    for ng, g in zip(next_target_grad_parts, target_grad_parts):
      tensorshape_util.set_shape(ng, g.shape)

    # outside the loop, it will minus half
    next_half_next_momentum_parts = [
        v + _multiply(eps, g, dtype=v.dtype)  # pylint: disable=g-complex-comprehension
        for v, eps, g
        in zip(half_next_momentum_parts, step_sizes, next_target_grad_parts)]

    return [
        next_half_next_momentum_parts,
        next_state_parts,
        next_target,
        next_target_grad_parts,
    ]


def process_args(target_fn, momentum_parts, state_parts,
                 target=None, target_grad_parts=None, hnn_model=None, input_dim=2):
  """Sanitize inputs to `__call__`."""
  with tf.name_scope('process_args'):
    momentum_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='momentum_parts')
        for v in momentum_parts]
    state_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='state_parts')
        for v in state_parts]
    if target is None or target_grad_parts is None:
      # print('state_parts in process_args', state_parts)
      [target, target_grad_parts] = hnn_fn_and_grads(
          target_fn, state_parts, momentum_parts, hnn_model, input_dim=input_dim)
      # print('target_grad_parts in process_args', target_grad_parts)
    else:
      target = tf.convert_to_tensor(
          target, dtype_hint=tf.float32, name='target')
      target_grad_parts = [
          tf.convert_to_tensor(
              g, dtype_hint=tf.float32, name='target_grad_part')
          for g in target_grad_parts]
    return momentum_parts, state_parts, target, target_grad_parts


def _multiply(tensor, state_sized_tensor, dtype):
  """Multiply `tensor` by a "state sized" tensor and preserve shape."""
  # User should be using a step size that does not alter the state size. This
  # will fail noisily if that is not the case.
  result = tf.cast(tensor, dtype) * tf.cast(state_sized_tensor, dtype)
  # print('result.shape in _multiply', result.shape)
  # print('state_sized_tensor.shape in _multiply', state_sized_tensor.shape)
  tensorshape_util.set_shape(result, state_sized_tensor.shape)
  return result

def is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))

JAX_MODE = False

def _hnn_value_and_gradients(fn, fn_arg_list, momentum_parts, hnn_model, input_dim=2, result=None, grads=None, name=None):
  """Helper to `maybe_call_fn_and_grads`."""
  with tf.name_scope(name or '_hnn_value_and_gradients'):

    def _convert_to_tensor(x, name):
      ctt = lambda x_: None if x_ is None else tf.convert_to_tensor(  # pylint: disable=g-long-lambda
          x_, name=name)
      return [ctt(x_) for x_ in x] if is_list_like(x) else ctt(x)

    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    fn_arg_list = _convert_to_tensor(fn_arg_list, 'fn_arg')
    '''
    input_dim // 2 == 1:
      - rank = 0: (this could only happen for input_dim // 2 == 1 case) input is a scalar
      - rank >= 1: could think everything is batch size
    input_dim // 2 > 1:
      - rank = 1: batch size = 1
      - rank > 1: except the last dimension, all others are batch size
    '''
    if input_dim // 2 == 1:
      tmp_input = [tf.concat([tf.reshape(target, shape=[-1, input_dim // 2]),
                              tf.reshape(momentum, shape=[-1, input_dim // 2])], axis=-1)
                              for target, momentum in zip(fn_arg_list, momentum_parts)]
    else:
      if tf.rank(fn_arg_list[0]) == 1:
        assert fn_arg_list[0].shape == input_dim // 2, 'The input dimension does not match with the input_dim!'
        tmp_input = [tf.expand_dims(tf.concat([target, momentum], axis=0), axis=0) 
                   for target, momentum in zip(fn_arg_list, momentum_parts)]
      elif tf.rank(fn_arg_list[0]) > 1:
        assert fn_arg_list[0].shape[-1] == input_dim // 2, 'The last input dimension does not match with the input_dim!'
        tmp_input = [tf.concat([tf.reshape(target, shape=[-1, input_dim // 2]),
                              tf.reshape(momentum, shape=[-1, input_dim // 2])], axis=-1)
                              for target, momentum in zip(fn_arg_list, momentum_parts)]
      else:
        raise NotImplementedError
    
    # double check whether the input shape to hnn_model is right
    for _tmp_input in tmp_input:
      assert _tmp_input.shape[-1] == input_dim, \
      'hnn_model.time_derivative requires input[-1] to be {} not {}'.format(input_dim, _tmp_input.shape[-1])
      assert tf.rank(_tmp_input) == 2, 'hnn_model.time_derivative requires the input of rank 2'
    
    if result is None and grads is None and (JAX_MODE or
                                             not tf.executing_eagerly()):
      # Currently, computing gradient is not working well with caching in
      # tensorflow eager mode (see below), so we will handle that case
      # separately.
      # TODO: how to handle the parallel case? 
      ## -- I think it can already be handled by the function itself as it allows inputs of different shapes
      result = fn(*fn_arg_list)
      # TODO: here *fn_arg_list should be modified -- directly ignore the packed case?
      # TODO: double check that we should get the later half of the derivative and times -1 as well -- no need?
      _, grads = tf.split(hnn_model.time_derivative(tf.Variable(*tmp_input)), 2, axis=-1) # batch size x (input_dim // 2)
      grads = tf.reshape(grads, shape=fn_arg_list[0].shape)
      return result, [grads]

    if result is None:
      result = fn(*fn_arg_list)
      if grads is None:
        assert tf.executing_eagerly()
        # Ensure we disable bijector cacheing in eager mode.
        # TODO(b/72831017): Remove this once bijector cacheing is fixed for
        # eager mode.
        fn_arg_list = [0 + x for x in fn_arg_list]

    result = _convert_to_tensor(result, 'fn_result')

    if grads is not None:
      grads = _convert_to_tensor(grads, 'fn_grad')
      return result, grads

    # print('*tmp_input.shape', tf.Variable(*tmp_input).shape)
    # print('hnn_model.time_derivative(tf.Variable(*tmp_input))', hnn_model.time_derivative(tf.Variable(*tmp_input)))
    _, grads = tf.split(hnn_model.time_derivative(tf.Variable(*tmp_input)), 2, axis=-1) # batch size x (input_dim // 2)
    # print('grads.shape', grads.shape)
    # print('fn_arg_list[0].shape', fn_arg_list[0].shape)
    grads = tf.reshape(grads, shape=fn_arg_list[0].shape)
    return result, [grads]

def hnn_fn_and_grads(fn,
                     fn_arg_list,
                     momentum_parts,
                     hnn_model,
                     input_dim=2,
                     result=None,
                     grads=None,
                     check_non_none_grads=True,
                     name=None):
  """Calls `fn` and computes the gradient of the result wrt `args_list`."""
  with tf.name_scope(name or 'hnn_fn_and_grads'):
    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    result, grads = _hnn_value_and_gradients(fn, fn_arg_list, momentum_parts, hnn_model, input_dim=input_dim,
                                             result=result, grads=grads)
    
    if not all(dtype_util.is_floating(r.dtype)
               for r in (result if is_list_like(result) else [result])):  # pylint: disable=superfluous-parens
      raise TypeError('Function result must be a `Tensor` with `float` '
                      '`dtype`.')
    if len(fn_arg_list) != len(grads):
      raise ValueError('Function args must be in one-to-one correspondence '
                       'with grads.')
    if check_non_none_grads and any(g is None for g in grads):
      raise ValueError('Encountered `None` gradient.\n'
                       '  fn_arg_list: {}\n'
                       '  grads: {}'.format(fn_arg_list, grads))
    return result, grads