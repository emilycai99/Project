import abc
import six

import tensorflow as tf

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import dtype_util

import sys, os
from pseudo_marginal.grad import calculate_grad

__all__ = [
    'StrangIntegrator',
    'process_args',
    'StrangIntegrator_HNN',
    'process_args_hnn',
    'StrangIntegrator_grad',
    'process_args_grad'
]

class StrangIntegrator(object):
    def __init__(self, target_fn, step_sizes, num_steps, target_dim, aux_dim, target_momentum_precision_mat):
        """Constructs the StrangIntegrator

        Assumes a kinetic energy function: `0.5*rho^T@M^{-1}@rho + 0.5*p^Tp`.

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
        target_dim: `int` dimension of target variables
        aux_dim: `int` dimension of auxiliary variables
        target_momentum_precision_mat: `Tensor`, M^{-1}
        """
        
        self._target_fn = target_fn
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._target_dim = target_dim
        self._aux_dim = aux_dim
        self._target_momentum_precision_mat = target_momentum_precision_mat

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
    def target_dim(self):
        return self._target_dim

    @property
    def aux_dim(self):
        return self._aux_dim
    
    @property
    def target_momentum_precision_mat(self):
        return self._target_momentum_precision_mat

    def __call__(self,
               momentum_parts,
               state_parts,
               target=None,
               name=None):
        """Applies `num_steps` of the Strang integrator

        Args:
            momentum_parts: Python `list` of `Tensor`s representing momentum for each
                state part. 
                - [(rho, p)].
            state_parts: Python `list` of `Tensor`s which collectively representing
                the state. 
                - [(theta, u)]
            target: Batch of scalar `Tensor` representing the target (i.e.,
                unnormalized log prob) evaluated at `state_parts`. 
                - logp(theta) + logphat(y|theta, u) + 0.5*u^Tu
            name: Python `str` used to group ops created by this function.
        
         Returns:
            next_momentum_parts: Python `list` of `Tensor`s representing new momentum.
            next_state_parts: Python `list` of `Tensor`s which collectively
                representing the new state.
            next_target: Batch of scalar `Tensor` representing the target (i.e.,
                unnormalized log prob) evaluated at `next_state_parts`.
        """
        with tf.name_scope(name or 'strang_integrate'):
            [
                momentum_parts,
                state_parts,
                target
            ] = process_args(
                self.target_fn,
                momentum_parts,
                state_parts,
                target
            )

            [   
                _, 
                next_momentum_parts,
                next_state_parts,
                next_target
            ] = tf.while_loop(
                cond=lambda i, *_: i < self.num_steps,
                body=lambda i, *args: [i+1] + list(_one_step(
                    self.target_fn, self.step_sizes, *args, 
                    target_momentum_precision_mat=self.target_momentum_precision_mat, 
                    target_dim=self.target_dim
                )),
                loop_vars=[
                    tf.zeros_like(self.num_steps, name='iter'),
                    momentum_parts,
                    state_parts,
                    target,
                ]
            )
            return (
                next_momentum_parts,
                next_state_parts,
                next_target
            )

def _one_step(
    target_fn,
    step_sizes,
    momentum_parts,
    state_parts,
    target,
    target_momentum_precision_mat=None,
    target_dim=None, 
    ):
    """ Body of integrator while loop.
    target_fn: logp(theta) + logphat(y|theta, u) + 0.5 * u^Tu
    """
    with tf.name_scope('strang_integrate_one_step'):
        # gradients for the momentum variables
        theta_parts = []
        rho_parts = []
        u_parts = []
        p_parts = []
        for x, y in zip(state_parts, momentum_parts):
            theta, u = tf.split(x, [target_dim, x.shape[0] - target_dim])
            rho, p = tf.split(y, [target_dim, y.shape[0] - target_dim])
            theta_parts.append(theta)
            rho_parts.append(rho)
            u_parts.append(u)
            p_parts.append(p)
        
        theta_parts_tmp = []
        for theta, eps, rho in zip(theta_parts, step_sizes, rho_parts):
            theta_parts_tmp.append(
                theta + _multiply(0.5 * eps, tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(rho, [-1, 1])), [-1]), dtype=theta.dtype)
            )
        u_parts_tmp = []
        for u, eps, p in zip(u_parts, step_sizes, p_parts):
            u_parts_tmp.append(
                _multiply(tf.math.sin(0.5 * eps), p, dtype=p.dtype) + _multiply(tf.math.cos(0.5 * eps), u, dtype=u.dtype)
            )
        
        [_, target_grad_parts] = mcmc_util.maybe_call_fn_and_grads(target_fn, tf.unstack(
                                    tf.concat([theta_parts_tmp, u_parts_tmp], axis=1), axis=0))

        next_theta_parts = []
        next_rho_parts = []
        next_u_parts = []
        next_p_parts = []
        for theta, rho, u, p, u_tmp, target_grad, eps in zip(theta_parts, rho_parts, u_parts, p_parts, u_parts_tmp, target_grad_parts, step_sizes):
            next_theta_parts.append(
                theta + _multiply(eps, tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(rho, [-1, 1])), -1), dtype=rho.dtype) \
                      + _multiply(0.5 * (eps**2), tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(target_grad[:target_dim], [-1, 1])), -1), dtype=theta.dtype)
            )
            next_rho_parts.append(
                rho + _multiply(eps, target_grad[:target_dim], dtype=rho.dtype)
            )
            next_u_parts.append(
                _multiply(tf.math.sin(eps), p, dtype=p.dtype) + \
                _multiply(tf.math.cos(eps), u, dtype=u.dtype) + \
                _multiply(tf.math.sin(0.5 * eps) * eps, target_grad[target_dim:] + u_tmp, dtype=u.dtype)
            )
            next_p_parts.append(
                _multiply(tf.math.cos(eps), p, dtype=p.dtype) - \
                _multiply(tf.math.sin(eps), u, dtype=u.dtype) + \
                _multiply(tf.math.cos(0.5 * eps) * eps, target_grad[target_dim:] + u_tmp, dtype=u.dtype)
            )
        next_state_parts = tf.unstack(tf.concat([next_theta_parts, next_u_parts], axis=1), axis=0)
        next_momentum_parts = tf.unstack(tf.concat([next_rho_parts, next_p_parts], axis=1), axis=0)

        [next_target, _] = mcmc_util.maybe_call_fn_and_grads(target_fn, next_state_parts)
       
        tensorshape_util.set_shape(next_target, target.shape)
        return [
            next_momentum_parts,
            next_state_parts,
            next_target,
        ]
   
def process_args(target_fn, momentum_parts, state_parts,
                 target=None):
  """Sanitize inputs to `__call__`."""
  ## convert momentum_parts, state_parts to a list
  with tf.name_scope('process_args'):
    momentum_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='momentum_parts')
        for v in momentum_parts]
    state_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='state_parts')
        for v in state_parts]
    if target is None:
      [target, _] = mcmc_util.maybe_call_fn_and_grads(
          target_fn, state_parts)
    else:
      target = tf.convert_to_tensor(
          target, dtype_hint=tf.float32, name='target')
    return momentum_parts, state_parts, target

class StrangIntegrator_HNN(object):
    def __init__(self, target_fn, step_sizes, num_steps, target_dim, aux_dim, target_momentum_precision_mat, hnn_model):
        """Constructs the StrangIntegrator with HNN

        Assumes a kinetic energy function: `0.5*rho^T@M^{-1}@rho + 0.5*p^Tp`.

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
        target_dim: `int` dimension of target variables
        aux_dim: `int` dimension of auxiliary variables
        target_momentum_precision_mat: `Tensor`, M^{-1}
        """
        
        self._target_fn = target_fn
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._target_dim = target_dim
        self._aux_dim = aux_dim
        self._target_momentum_precision_mat = target_momentum_precision_mat
        self._hnn_model = hnn_model

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
    def target_dim(self):
        return self._target_dim

    @property
    def aux_dim(self):
        return self._aux_dim
    
    @property
    def target_momentum_precision_mat(self):
        return self._target_momentum_precision_mat
    
    @property
    def hnn_model(self):
        return self._hnn_model
    
    def __call__(self,
               momentum_parts,
               state_parts,
               target=None,
               name=None):
        with tf.name_scope(name or 'strang_integrate_hnn'):
                [
                    momentum_parts,
                    state_parts,
                    target
                ] = process_args_hnn(
                    self.target_fn,
                    momentum_parts,
                    state_parts,
                    target
                )

                [   
                    _, 
                    next_momentum_parts,
                    next_state_parts,
                    next_target
                ] = tf.while_loop(
                    cond=lambda i, *_: i < self.num_steps,
                    body=lambda i, *args: [i+1] + list(_one_step_hnn(
                        self.target_fn, self.step_sizes, *args, 
                        target_momentum_precision_mat=self.target_momentum_precision_mat, 
                        target_dim=self.target_dim, aux_dim=self.aux_dim, hnn_model=self.hnn_model
                    )),
                    loop_vars=[
                        tf.zeros_like(self.num_steps, name='iter'),
                        momentum_parts,
                        state_parts,
                        target,
                    ]
                )
                return (
                    next_momentum_parts,
                    next_state_parts,
                    next_target
                )
    
def _one_step_hnn(
    target_fn,
    step_sizes,
    momentum_parts,
    state_parts,
    target,
    target_momentum_precision_mat=None,
    target_dim=None, 
    aux_dim=None,
    hnn_model=None,
    ):
    """ Body of integrator while loop.
    target_fn: logp(theta) + logphat(y|theta, u) + 0.5 * u^Tu
    """
    with tf.name_scope('strang_integrate_one_step_hnn'):
        # gradients for the momentum variables
        theta_parts = []
        rho_parts = []
        u_parts = []
        p_parts = []
        for x, y in zip(state_parts, momentum_parts):
            theta, u = tf.split(x, [target_dim, aux_dim])
            rho, p = tf.split(y, [target_dim, aux_dim])
            theta_parts.append(theta)
            rho_parts.append(rho)
            u_parts.append(u)
            p_parts.append(p)
        
        theta_parts_tmp = []
        for theta, eps, rho in zip(theta_parts, step_sizes, rho_parts):
            theta_parts_tmp.append(
                theta + _multiply(0.5 * eps, tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(rho, [-1, 1])), [-1]), dtype=theta.dtype)
            )
        u_parts_tmp = []
        for u, eps, p in zip(u_parts, step_sizes, p_parts):
            u_parts_tmp.append(
                _multiply(tf.math.sin(0.5 * eps), p, dtype=p.dtype) + _multiply(tf.math.cos(0.5 * eps), u, dtype=u.dtype)
            )
        
        [_, target_grad_parts] = hnn_fn_and_grads(target_fn, tf.unstack(
                                    tf.concat([theta_parts_tmp, u_parts_tmp], axis=1), axis=0),
                                    momentum_parts, hnn_model=hnn_model, target_dim=target_dim, aux_dim=aux_dim)

        next_theta_parts = []
        next_rho_parts = []
        next_u_parts = []
        next_p_parts = []
        for theta, rho, u, p, u_tmp, target_grad, eps in zip(theta_parts, rho_parts, u_parts, p_parts, u_parts_tmp, target_grad_parts, step_sizes):
            next_theta_parts.append(
                theta + _multiply(eps, tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(rho, [-1, 1])), -1), dtype=rho.dtype) \
                      + _multiply(0.5 * (eps**2), tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(target_grad[:target_dim], [-1, 1])), -1), dtype=theta.dtype)
            )
            next_rho_parts.append(
                rho + _multiply(eps, target_grad[:target_dim], dtype=rho.dtype)
            )
            next_u_parts.append(
                _multiply(tf.math.sin(eps), p, dtype=p.dtype) + \
                _multiply(tf.math.cos(eps), u, dtype=u.dtype) + \
                _multiply(tf.math.sin(0.5 * eps) * eps, target_grad[target_dim:] + u_tmp, dtype=u.dtype)
            )
            next_p_parts.append(
                _multiply(tf.math.cos(eps), p, dtype=p.dtype) - \
                _multiply(tf.math.sin(eps), u, dtype=u.dtype) + \
                _multiply(tf.math.cos(0.5 * eps) * eps, target_grad[target_dim:] + u_tmp, dtype=u.dtype)
            )
        next_state_parts = tf.unstack(tf.concat([next_theta_parts, next_u_parts], axis=1), axis=0)
        next_momentum_parts = tf.unstack(tf.concat([next_rho_parts, next_p_parts], axis=1), axis=0)

        [next_target, _] = hnn_fn_and_grads(target_fn, next_state_parts, next_momentum_parts, 
                                            hnn_model=hnn_model, target_dim=target_dim, aux_dim=aux_dim)
       
        tensorshape_util.set_shape(next_target, target.shape)
        return [
            next_momentum_parts,
            next_state_parts,
            next_target,
        ]

def is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))

JAX_MODE = False

def _hnn_value_and_gradients(fn, fn_arg_list, momentum_parts, hnn_model, target_dim=2, aux_dim=2, result=None, grads=None, name=None):
  """Helper to `maybe_call_fn_and_grads`."""
  with tf.name_scope(name or '_hnn_value_and_gradients'):

    def _convert_to_tensor(x, name):
      ctt = lambda x_: None if x_ is None else tf.convert_to_tensor(  # pylint: disable=g-long-lambda
          x_, name=name)
      return [ctt(x_) for x_ in x] if is_list_like(x) else ctt(x)

    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    fn_arg_list = _convert_to_tensor(fn_arg_list, 'fn_arg')
    
    tmp_input = []
    for x, y in zip(fn_arg_list, momentum_parts):
       theta, u = tf.split(x, [target_dim, aux_dim], axis=0)
       rho, p = tf.split(y, [target_dim, aux_dim], axis=0)
       tmp_input.append(tf.expand_dims(tf.concat([theta, rho, u, p], axis=0), axis=0))
    
    # double check whether the input shape to hnn_model is right
    for _tmp_input in tmp_input:
      assert _tmp_input.shape[-1] == 2 * (target_dim + aux_dim), \
      'hnn_model.time_derivative requires input[-1] to be {} not {}'.format(2 * (target_dim + aux_dim), _tmp_input.shape[-1])
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
      total_grads = hnn_model.time_derivative(tf.Variable(*tmp_input))
      _, grads_theta, _, grads_u = tf.split(total_grads, [target_dim, target_dim, aux_dim, aux_dim], axis=-1)
      grads = tf.reshape(tf.concat([grads_theta, grads_u], axis=-1), shape=fn_arg_list[0].shape)
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

    total_grads = hnn_model.time_derivative(tf.Variable(*tmp_input))
    _, grads_theta, _, grads_u = tf.split(total_grads, [target_dim, target_dim, aux_dim, aux_dim], axis=-1)
    grads = tf.reshape(tf.concat([grads_theta, grads_u], axis=-1), shape=fn_arg_list[0].shape)
    return result, [grads]

def hnn_fn_and_grads(fn,
                     fn_arg_list,
                     momentum_parts,
                     hnn_model,
                     target_dim=None,
                     aux_dim=None,
                     result=None,
                     grads=None,
                     check_non_none_grads=True,
                     name=None):
  """Calls `fn` and computes the gradient of the result wrt `args_list`."""
  with tf.name_scope(name or 'hnn_fn_and_grads'):
    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    result, grads = _hnn_value_and_gradients(fn, fn_arg_list, momentum_parts, hnn_model, target_dim=target_dim, aux_dim=aux_dim,
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

def process_args_hnn(target_fn, momentum_parts, state_parts,
                 target=None, hnn_model=None, target_dim=None, aux_dim=None):
  """Sanitize inputs to `__call__`."""
  ## convert momentum_parts, state_parts to a list
  with tf.name_scope('process_args_hnn'):
    momentum_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='momentum_parts')
        for v in momentum_parts]
    state_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='state_parts')
        for v in state_parts]
    if target is None:
      [target, _] = hnn_fn_and_grads(
          target_fn, state_parts, momentum_parts, hnn_model=hnn_model, target_dim=target_dim, aux_dim=aux_dim)
    else:
      target = tf.convert_to_tensor(
          target, dtype_hint=tf.float32, name='target')
    return momentum_parts, state_parts, target
     

def _multiply(tensor, state_sized_tensor, dtype):
  """Multiply `tensor` by a "state sized" tensor and preserve shape."""
  # User should be using a step size that does not alter the state size. This
  # will fail noisily if that is not the case.
  result = tf.cast(tensor, dtype) * tf.cast(state_sized_tensor, dtype)
  tensorshape_util.set_shape(result, state_sized_tensor.shape)
  return result

class StrangIntegrator_grad(object):
    def __init__(self, target_fn, step_sizes, num_steps, target_dim, aux_dim, target_momentum_precision_mat, pm_nuts_args=None):
        """Constructs the StrangIntegrator with HNN

        Assumes a kinetic energy function: `0.5*rho^T@M^{-1}@rho + 0.5*p^Tp`.

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
        target_dim: `int` dimension of target variables
        aux_dim: `int` dimension of auxiliary variables
        target_momentum_precision_mat: `Tensor`, M^{-1}
        """
        
        self._target_fn = target_fn
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._target_dim = target_dim
        self._aux_dim = aux_dim
        self._target_momentum_precision_mat = target_momentum_precision_mat
        calculate_grad_obj = calculate_grad(pm_nuts_args)
        self._grad_func = calculate_grad_obj.grad_total

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
    def target_dim(self):
        return self._target_dim

    @property
    def aux_dim(self):
        return self._aux_dim
    
    @property
    def target_momentum_precision_mat(self):
        return self._target_momentum_precision_mat

    @property
    def grad_func(self):
        return self._grad_func
    
    def __call__(self,
               momentum_parts,
               state_parts,
               target=None,
               name=None):
        with tf.name_scope(name or 'strang_integrate_grad'):
                [
                    momentum_parts,
                    state_parts,
                    target
                ] = process_args_grad(
                    self.target_fn,
                    momentum_parts,
                    state_parts,
                    target,
                    grad_func=self.grad_func
                )

                [   
                    _, 
                    next_momentum_parts,
                    next_state_parts,
                    next_target
                ] = tf.while_loop(
                    cond=lambda i, *_: i < self.num_steps,
                    body=lambda i, *args: [i+1] + list(_one_step_grad(
                        self.target_fn, self.step_sizes, *args, 
                        target_momentum_precision_mat=self.target_momentum_precision_mat, 
                        target_dim=self.target_dim, aux_dim=self.aux_dim, grad_func=self.grad_func
                    )),
                    loop_vars=[
                        tf.zeros_like(self.num_steps, name='iter'),
                        momentum_parts,
                        state_parts,
                        target,
                    ]
                )
                return (
                    next_momentum_parts,
                    next_state_parts,
                    next_target
                )
    
def _one_step_grad(
    target_fn,
    step_sizes,
    momentum_parts,
    state_parts,
    target,
    target_momentum_precision_mat=None,
    target_dim=None, 
    aux_dim=None,
    grad_func=None
    ):
    """ Body of integrator while loop.
    target_fn: logp(theta) + logphat(y|theta, u) + 0.5 * u^Tu
    """
    with tf.name_scope('strang_integrate_one_step_hnn'):
        # gradients for the momentum variables
        theta_parts = []
        rho_parts = []
        u_parts = []
        p_parts = []
        for x, y in zip(state_parts, momentum_parts):
            theta, u = tf.split(x, [target_dim, aux_dim])
            rho, p = tf.split(y, [target_dim, aux_dim])
            theta_parts.append(theta)
            rho_parts.append(rho)
            u_parts.append(u)
            p_parts.append(p)
        
        theta_parts_tmp = []
        for theta, eps, rho in zip(theta_parts, step_sizes, rho_parts):
            theta_parts_tmp.append(
                theta + _multiply(0.5 * eps, tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(rho, [-1, 1])), [-1]), dtype=theta.dtype)
            )
        u_parts_tmp = []
        for u, eps, p in zip(u_parts, step_sizes, p_parts):
            u_parts_tmp.append(
                _multiply(tf.math.sin(0.5 * eps), p, dtype=p.dtype) + _multiply(tf.math.cos(0.5 * eps), u, dtype=u.dtype)
            )
        
        [_, target_grad_parts] = manual_fn_and_grads(target_fn, tf.unstack(
                                    tf.concat([theta_parts_tmp, u_parts_tmp], axis=1), axis=0),
                                    momentum_parts, target_dim=target_dim, aux_dim=aux_dim, grad_func=grad_func)

        next_theta_parts = []
        next_rho_parts = []
        next_u_parts = []
        next_p_parts = []
        for theta, rho, u, p, u_tmp, target_grad, eps in zip(theta_parts, rho_parts, u_parts, p_parts, u_parts_tmp, target_grad_parts, step_sizes):
            next_theta_parts.append(
                theta + _multiply(eps, tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(rho, [-1, 1])), -1), dtype=rho.dtype) \
                      + _multiply(0.5 * (eps**2), tf.reshape(tf.matmul(target_momentum_precision_mat, tf.reshape(target_grad[:target_dim], [-1, 1])), -1), dtype=theta.dtype)
            )
            next_rho_parts.append(
                rho + _multiply(eps, target_grad[:target_dim], dtype=rho.dtype)
            )
            next_u_parts.append(
                _multiply(tf.math.sin(eps), p, dtype=p.dtype) + \
                _multiply(tf.math.cos(eps), u, dtype=u.dtype) + \
                _multiply(tf.math.sin(0.5 * eps) * eps, target_grad[target_dim:] + u_tmp, dtype=u.dtype)
            )
            next_p_parts.append(
                _multiply(tf.math.cos(eps), p, dtype=p.dtype) - \
                _multiply(tf.math.sin(eps), u, dtype=u.dtype) + \
                _multiply(tf.math.cos(0.5 * eps) * eps, target_grad[target_dim:] + u_tmp, dtype=u.dtype)
            )
        next_state_parts = tf.unstack(tf.concat([next_theta_parts, next_u_parts], axis=1), axis=0)
        next_momentum_parts = tf.unstack(tf.concat([next_rho_parts, next_p_parts], axis=1), axis=0)

        [next_target, _] = manual_fn_and_grads(target_fn, next_state_parts, next_momentum_parts, 
                                            target_dim=target_dim, aux_dim=aux_dim, grad_func=grad_func)
       
        tensorshape_util.set_shape(next_target, target.shape)
        return [
            next_momentum_parts,
            next_state_parts,
            next_target,
        ]
    
def manual_fn_and_grads(fn,
                     fn_arg_list,
                     momentum_parts,
                     target_dim=None,
                     aux_dim=None,
                     grad_func=None,
                     result=None,
                     grads=None,
                     check_non_none_grads=True,
                     name=None):
  """Calls `fn` and computes the gradient of the result wrt `args_list`."""
  with tf.name_scope(name or 'manual_fn_and_grads'):
    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    result, grads = _manual_value_and_gradients(fn, fn_arg_list, momentum_parts, target_dim=target_dim, aux_dim=aux_dim,
                                             result=result, grads=grads, grad_func=grad_func)
    
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
  
def _manual_value_and_gradients(fn, fn_arg_list, momentum_parts, target_dim=2, aux_dim=2, result=None, grads=None, grad_func=None, name=None):
  """Helper to `maybe_call_fn_and_grads`."""
  with tf.name_scope(name or '_manual_value_and_gradients'):

    def _convert_to_tensor(x, name):
      ctt = lambda x_: None if x_ is None else tf.convert_to_tensor(  # pylint: disable=g-long-lambda
          x_, name=name)
      return [ctt(x_) for x_ in x] if is_list_like(x) else ctt(x)

    fn_arg_list = (list(fn_arg_list) if is_list_like(fn_arg_list)
                   else [fn_arg_list])
    fn_arg_list = _convert_to_tensor(fn_arg_list, 'fn_arg')
    
    tmp_input = []
    for x, y in zip(fn_arg_list, momentum_parts):
       theta, u = tf.split(x, [target_dim, aux_dim], axis=0)
       rho, p = tf.split(y, [target_dim, aux_dim], axis=0)
       tmp_input.append(tf.concat([theta, rho, u, p], axis=0))
    
    # double check whether the input shape to hnn_model is right
    for _tmp_input in tmp_input:
      assert _tmp_input.shape[0] == 2 * (target_dim + aux_dim), \
      'calculate_grad.grad_total requires input to be {} not {}'.format(2 * (target_dim + aux_dim), _tmp_input.shape[0])
      assert tf.rank(_tmp_input) == 1, 'calculate_grad.grad_total requires the input of rank 1'
    
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
      total_grads = grad_func(tf.Variable(*tmp_input))
      grads_theta, _, grads_u, _ = tf.split(total_grads, [target_dim, target_dim, aux_dim, aux_dim], axis=-1)
      grads = tf.reshape(tf.concat([grads_theta, grads_u], axis=-1), shape=fn_arg_list[0].shape)
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

    total_grads = grad_func(tf.Variable(*tmp_input))
    grads_theta, _, grads_u, _ = tf.split(total_grads, [target_dim, target_dim, aux_dim, aux_dim], axis=-1)
    grads = tf.reshape(tf.concat([-grads_theta, -grads_u], axis=-1), shape=fn_arg_list[0].shape)
    return result, [grads]

def process_args_grad(target_fn, momentum_parts, state_parts,
                 target=None, target_dim=None, aux_dim=None, grad_func=None):
  """Sanitize inputs to `__call__`."""
  ## convert momentum_parts, state_parts to a list
  with tf.name_scope('process_args_grad'):
    momentum_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='momentum_parts')
        for v in momentum_parts]
    state_parts = [
        tf.convert_to_tensor(
            v, dtype_hint=tf.float32, name='state_parts')
        for v in state_parts]
    if target is None:
      [target, _] = manual_fn_and_grads(
          target_fn, state_parts, momentum_parts, target_dim=target_dim, aux_dim=aux_dim, grad_func=grad_func)
    else:
      target = tf.convert_to_tensor(
          target, dtype_hint=tf.float32, name='target')
    return momentum_parts, state_parts, target