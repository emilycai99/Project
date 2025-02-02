import sys
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import math

def numerical_grad(coords, func, target_dim, aux_dim):
    '''
    Description:
        calculate numerical gradients and then reshape:
        [grad_theta, grad_rho, grad_u, grad_p] -> [grad_rho, -grad_theta, grad_p, -grad_u]
    Args:
        coords: [theta, rho, u, p]
        func: hamiltonian function
        target_dim: theta.shape[0]
        aux_dim: u.shape[0]
    Return:
        out: [grad_rho, -grad_theta, grad_p, -grad_u]
    '''
    assert coords.shape[0] == 2 * (target_dim + aux_dim), 'incorrect specification of target or aux dim'
    # dcoords: gradient of "func" evaluated at "coords"
    dcoords = tfp.math.value_and_gradient(func, coords)[-1]
    c1, c2, c3, c4 = tf.split(dcoords, [target_dim, target_dim, aux_dim, aux_dim])
    out = tf.concat([c2, -c1, c4, -c3], axis=0)
    return out

def integrator_one_step(coords, func, derivs_func, h, target_dim, aux_dim):
    '''
    Description:
        Implement the one-step integrator in Appendix A of PM-HMC
    Args:
        coords: current value of [theta, rho, u, p]
        func: hamiltonian function
        derivs_func: the function to calculate the derivative
        h: step size
        target_dim: theta.shape[0]
        aux_dim: u.shape[0]
    Return:
        coords_new: after one-step integration, same size as coords
    '''
    theta, rho, u, p = tf.split(coords, [target_dim, target_dim, aux_dim, aux_dim])
    
    # calculate derivative of {log(p(theta)) + log(phat(y|theta, u'))}
    # with respect to theta' = theta + h/2 * rho
    # and set u' = p * sin(h/2) + u * cos(h/2)
    # and
    # calculate derivative of {log(phat(y|theta', u))}
    # with respect to u' = p * sin(h/2) + u*cos(h/2)
    # and set theta' = theta + (h/2) * rho
    theta_tmp = theta + 0.5 * h * rho
    u_tmp = p * math.sin(0.5 * h) + u * math.cos(0.5 * h)
    coords_tmp = tf.concat([theta_tmp, rho, u_tmp, p], axis=0)
    dcoords = derivs_func(coords_tmp, func, target_dim, aux_dim)
    
    # calculate new coords
    theta_new = theta + h * rho + 0.5 * (h**2) * dcoords[target_dim:2*target_dim]
    rho_new = rho + h * dcoords[target_dim:2*target_dim]
    u_new = p * math.sin(h) + u * math.cos(h) + math.sin(0.5 * h) * h * (dcoords[-aux_dim:] + u_tmp)
    p_new = p * math.cos(h) - u * math.sin(h) + math.cos(0.5 * h) * h * (dcoords[-aux_dim:] + u_tmp)

    coords_new = tf.concat([theta_new, rho_new, u_new, p_new], axis=0)
    return coords_new

def integrator(coords, func, derivs_func, h, steps, target_dim, aux_dim):
    '''
    Description:
        Implement the multi-step integrator
    Args:
        coords: current value of [theta, rho, u, p]
        func: hamiltonian function
        derivs_func: the function to calculate the derivative
        h: step size
        steps: number of integration steps
        target_dim: theta.shape[0]
        aux_dim: u.shape[0]
    Return:
        out: coords.shape[0] x (steps + 1)
    '''
    out = [coords]
    for i in range(steps):
        new = integrator_one_step(out[-1], func, derivs_func, h, target_dim, aux_dim)
        out.append(new)
    out = tf.stack(out, axis=-1)
    return out

def L2_loss(u, v):
#   return tf.reduce_mean(tf.square(u - v))
    return tf.reduce_mean(tf.reduce_sum(tf.square(u-v), axis=1))

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

class Transcript(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def log_start(filename):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Transcript(filename)

def log_stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal