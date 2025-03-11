import pytest
from pytest_mock import mocker
import tensorflow as tf
from tensorflow_probability import distributions as tfd

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.grad import *
from pseudo_marginal.utils import numerical_grad

def test_stable_log1pexp():
    '''
    unit test for stable_log1pexp
    '''
    # Test for values less than -37.0
    x = tf.constant(-40.0)
    assert tf.math.reduce_all(tf.math.equal(stable_log1pexp(x), tf.exp(x)))

    # Test for values between -37.0 and 18.0
    x = tf.constant(10.0)
    assert tf.math.reduce_all(tf.math.equal(stable_log1pexp(x), tf.math.log(1 + tf.math.exp(x))))

    # Test for values between 18.0 and 33.3
    x = tf.constant(20.0)
    assert tf.math.reduce_all(tf.math.equal(stable_log1pexp(x), x + tf.math.exp(-x)))

    # Test for values greater than 33.3
    x = tf.constant(40.0)
    assert tf.math.reduce_all(tf.math.equal(stable_log1pexp(x), x))

def test_sigmoid():
    '''
    unit test for sigmoid
    '''
    # Test for positive values
    x_positive = tf.constant([1.0, 2.0, 3.0])
    expected_output_positive = tf.constant([0.7310586, 0.8807971, 0.9525741])
    assert tf.reduce_all(tf.experimental.numpy.allclose(sigmoid(x_positive), expected_output_positive))

    # Test for negative values
    x_negative = tf.constant([-1.0, -2.0, -3.0])
    expected_output_negative = tf.constant([0.26894143, 0.11920292, 0.04742587])
    assert tf.reduce_all(tf.experimental.numpy.allclose(sigmoid(x_negative), expected_output_negative))

    # Test for zero
    x_zero = tf.constant(0.0)
    assert tf.reduce_all(tf.experimental.numpy.allclose(sigmoid(x_zero), 0.5))

def mock_log_gauss_mix_pdf(X, coords):

    mu1 = coords[0]
    mu2 = coords[1]
    lambda1 = tf.math.exp(coords[2])
    lambda2 = tf.math.exp(coords[3])
    w = tf.sigmoid(coords[-1])

    x_dist = tfd.Mixture(
            cat=tfd.Categorical(probs=[w, 1.0-w]),
            components=[
                tfd.Normal(loc=mu1, scale=lambda1**(-0.5)),
                tfd.Normal(loc=mu2, scale=lambda2**(-0.5))
            ]
        )
    
    out = x_dist.log_prob(X)
    return out

def test_grad_gauss_mix():
    '''
    unit test for grad_gauss_mix
    '''
    T = 500
    N = 128
    tf.random.set_seed(seed=0)
    X = tf.random.normal(shape=[T, N], dtype=tf.float32)
    coords = tf.random.normal(shape=[5], dtype=tf.float32)
    # coords = tf.constant([10.0, 20.0, 8.0, -5, (1e-10) / (1 - 1e-10)])
    expected_out = tfp.math.value_and_gradient(mock_log_gauss_mix_pdf, X, coords)[-1][-1]
    expected_pdf = mock_log_gauss_mix_pdf(X, coords)
    expected_u = 3.0 * tfp.math.value_and_gradient(mock_log_gauss_mix_pdf, X, coords)[-1][0]
    result_pdf, result_out, result_u = grad_gauss_mix(X, coords[0], coords[1], coords[2], coords[3], coords[-1])
    result_out = tf.reduce_sum(result_out, axis=[0, 1])

    # assert False
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected_out, result_out, rtol=1e-3))
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected_pdf, result_pdf))
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected_u, result_u, atol=1e-4))

def mock_log_bernoulli_pdf(Y, X, Z, beta):
    X_expand = tf.repeat(tf.expand_dims(X, -1), repeats=Y.shape[-1], axis=-1)
    Z_expand = tf.repeat(tf.expand_dims(Z, 1), repeats=X.shape[-1], axis=1)
    Y_expand = tf.repeat(tf.expand_dims(Y, axis=1), repeats=X.shape[-1], axis=1)
    g_dist = tfd.Bernoulli(logits=X_expand+tf.tensordot(Z_expand, beta, axes=[[3], [0]]))
    out = g_dist.log_prob(Y_expand)
    return out

def test_grad_bernoulli():
    '''
    unit test for grad_bernoulli
    '''
    T = 500
    N = 128
    n = 6
    p = 8
    tf.random.set_seed(seed=0)
    Y = tf.random.normal(shape=[T, n], dtype=tf.float32)
    X = tf.random.normal(shape=[T, N], dtype=tf.float32)
    Z = tf.random.normal(shape=[T, n, p], dtype=tf.float32)
    beta = tf.random.normal(shape=[p], dtype=tf.float32)
    result_pdf, result_beta, result_u = grad_bernoulli(Y, Z, X, beta)
    expected_beta = tfp.math.value_and_gradient(mock_log_bernoulli_pdf, Y, X, Z, beta)[-1][-1]
    expected_pdf = mock_log_bernoulli_pdf(Y, X, Z, beta)
    expected_u = 3 * tfp.math.value_and_gradient(mock_log_bernoulli_pdf, Y, X, Z, beta)[-1][1]
    result_beta = tf.reduce_sum(result_beta, axis=[0, 1, 2])
    result_u = tf.reduce_sum(result_u, axis=-1)

    assert tf.reduce_all(tf.experimental.numpy.allclose(expected_beta, result_beta, atol=1e-5))
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected_pdf, result_pdf, atol=1e-5))
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected_u, result_u, atol=1e-5))

def mock_normal(X):
    dist = tfd.Normal(loc=0.0, scale=3.0)
    return dist.log_prob(X)

def test_grad_normal():
    '''
    unit test for grad_normal
    '''
    T = 2
    N = 3
    X = tf.random.normal(shape=[T, N], dtype=tf.float32)
    log_pdf = mock_normal(X)
    result_log_pdf, result_grad_u = grad_normal(X)
    expected_u = 3 * tfp.math.value_and_gradient(mock_normal, X)[-1]

    assert tf.reduce_all(tf.experimental.numpy.allclose(log_pdf, result_log_pdf))
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected_u, result_grad_u))


class mock_args():
    def __init__(self):
        self.T = 3
        self.N = 2
        self.n = 4
        self.p = 2
        # self.T = 500
        # self.N = 128
        # self.n = 6
        # self.p = 8
        self.target_dim = self.p + 5
        self.aux_dim = self.T * self.N
        self.input_dim = 2 * (self.target_dim + self.aux_dim)
        tf.random.set_seed(0)
        self.Y = tf.random.normal(shape=[self.T, self.n], dtype=tf.float32)
        self.Z = tf.random.normal(shape=[self.T, self.n, self.p], dtype=tf.float32)
        self.dist_name = 'gauss_mix'
        self.data_pth = ''
        self.rho_var = [1.0 for i in range(self.target_dim)]

class mock_Hamiltonian_func(tf.Module):
    def __init__(self, args, Y, Z, name=None):
        super().__init__(name)
        self.args = args
        self.Y = Y
        self.Z = Z
    def gauss_mix_func(self, coords):
        # d: length of theta
        d = self.args.p + 5
        # coords = [beta, mu1, mu2, log(lambda1), log(lambda2), logit(w), rho, u, p]
        assert coords.shape[0] == 2*d + 2*(self.args.T * self.args.N), 'incorrect shape of coords'

        # get all the parameters
        theta = coords[:d]
        beta = coords[:self.args.p]
        mu1 = coords[self.args.p]
        mu2 = coords[self.args.p+1]
        lambda1 = tf.math.exp(coords[self.args.p+2])
        lambda2 = tf.math.exp(coords[self.args.p+3])
        w = tf.math.sigmoid(coords[self.args.p+4])
        rho = coords[d:2*d]
        u, p = tf.split(coords[-2*(self.args.T * self.args.N):], 2)

        assert rho.shape[0] == d, 'incorrect shape of rho'
        assert u.shape[0] == self.args.T * self.args.N, 'incorrect shape of u'
        assert p.shape[0] == self.args.T * self.args.N, 'incorrect shape of p'

        # X contains Xil for i = 1 to T and l = 1 to N
        X = tf.reshape(3.0 * u, shape=[self.args.T, self.args.N])

        # define distributions
        ## define f(Xil): gaussian mixture
        x_dist = tfd.Mixture(
            cat=tfd.Categorical(probs=[w, 1.0-w]),
            components=[
                tfd.Normal(loc=mu1, scale=lambda1**(-0.5)),
                tfd.Normal(loc=mu2, scale=lambda2**(-0.5))
            ]
        )
        ## define q(Xil)
        q_dist = tfd.Normal(loc=0.0, scale=3.0)

        ## define g(Yij|Xil): bernoulli
        ### X_expand.shape = [T, N, n]; X.shape = [T, N]
        X_expand = tf.repeat(tf.expand_dims(X, -1), repeats=self.args.n, axis=-1)
        ### Z_expand.shape = [T, N, n, p]; Z.shape = [T, n, p]
        Z_expand = tf.repeat(tf.expand_dims(self.Z, 1), repeats=self.args.N, axis=1)
        g_dist = tfd.Bernoulli(logits=X_expand+tf.tensordot(Z_expand, beta, axes=[[3], [0]]))

        ## N(0,100) prior for each component of theta
        theta_prior = tfd.MultivariateNormalDiag(loc=[0.0 for _ in range(d)], scale_diag=[10.0 for _ in range(d)])

        # calculate f(Xil): gaussian mixture, shape = [T, N] -> [T, N, n]
        log_f_val = x_dist.log_prob(X_expand)
        # calculate q(Xil): shape = [T, N] -> [T, N, n]
        log_q_val = q_dist.log_prob(X_expand)
        # calculate g(Yij|Xil): shape = [T, N, n] (Yilj)
        Y_expand = tf.repeat(tf.expand_dims(self.Y, axis=1), repeats=self.args.N, axis=1)
        log_g_val = g_dist.log_prob(Y_expand)

        # shape = [T, N, n]
        log_w_ilj = log_g_val + log_f_val - log_q_val
        max_log_w_ilj = tf.math.reduce_max(log_w_ilj, axis=1, keepdims=True)
        w_ilj = tf.math.exp(tf.math.subtract(log_w_ilj, max_log_w_ilj))
        w_ij = tf.math.log(tf.reduce_sum(w_ilj, axis=1)) + tf.squeeze(max_log_w_ilj, axis=1) - tf.math.log(tf.constant([self.args.N], dtype=tf.float32))
        # calculate the log(phat(y|theta, u))
        log_phat = tf.reduce_sum(w_ij)

        # calculate logp(theta)
        log_prior = theta_prior.log_prob(theta)

        # calculate the hamiltonian value
        H = -log_prior - log_phat + 0.5*tf.tensordot(rho, rho, axes=1) + 0.5*tf.tensordot(u, u, axes=1) +\
            0.5*tf.tensordot(p, p, axes=1)

        return H
    
    def get_target_log_prob(self, coords):
        # d: length of theta
        d = self.args.p + 5
        # coords = [beta, mu1, mu2, log(lambda1), log(lambda2), logit(w), rho, u, p]
        assert coords.shape[0] == 2*d + 2*(self.args.T * self.args.N), 'incorrect shape of coords'

        # get all the parameters
        theta = coords[:d]
        beta = coords[:self.args.p]
        mu1 = coords[self.args.p]
        mu2 = coords[self.args.p+1]
        lambda1 = tf.math.exp(coords[self.args.p+2])
        lambda2 = tf.math.exp(coords[self.args.p+3])
        w = tf.math.sigmoid(coords[self.args.p+4])
        rho = coords[d:2*d]
        u, p = tf.split(coords[-2*(self.args.T * self.args.N):], 2)

        assert rho.shape[0] == d, 'incorrect shape of rho'
        assert u.shape[0] == self.args.T * self.args.N, 'incorrect shape of u'
        assert p.shape[0] == self.args.T * self.args.N, 'incorrect shape of p'

        # X contains Xil for i = 1 to T and l = 1 to N
        X = tf.reshape(3.0 * u, shape=[self.args.T, self.args.N])

        # define distributions
        ## define f(Xil): gaussian mixture
        x_dist = tfd.Mixture(
            cat=tfd.Categorical(probs=[w, 1.0-w]),
            components=[
                tfd.Normal(loc=mu1, scale=lambda1**(-0.5)),
                tfd.Normal(loc=mu2, scale=lambda2**(-0.5))
            ]
        )
        ## define q(Xil)
        q_dist = tfd.Normal(loc=0.0, scale=3.0)

        ## define g(Yij|Xil): bernoulli
        ### X_expand.shape = [T, N, n]; X.shape = [T, N]
        X_expand = tf.repeat(tf.expand_dims(X, -1), repeats=self.args.n, axis=-1)
        ### Z_expand.shape = [T, N, n, p]; Z.shape = [T, n, p]
        Z_expand = tf.repeat(tf.expand_dims(self.Z, 1), repeats=self.args.N, axis=1)
        g_dist = tfd.Bernoulli(logits=X_expand+tf.tensordot(Z_expand, beta, axes=[[3], [0]]))

        ## N(0,100) prior for each component of theta
        theta_prior = tfd.MultivariateNormalDiag(loc=[0.0 for _ in range(d)], scale_diag=[10.0 for _ in range(d)])

        # calculate f(Xil): gaussian mixture, shape = [T, N] -> [T, N, n]
        log_f_val = x_dist.log_prob(X_expand)
        # calculate q(Xil): shape = [T, N] -> [T, N, n]
        log_q_val = q_dist.log_prob(X_expand)
        # calculate g(Yij|Xil): shape = [T, N, n] (Yilj)
        Y_expand = tf.repeat(tf.expand_dims(self.Y, axis=1), repeats=self.args.N, axis=1)
        log_g_val = g_dist.log_prob(Y_expand)

        # shape = [T, N, n]
        log_w_ilj = log_g_val + log_f_val - log_q_val
        max_log_w_ilj = tf.math.reduce_max(log_w_ilj, axis=1, keepdims=True)
        w_ilj = tf.math.exp(tf.math.subtract(log_w_ilj, max_log_w_ilj))
        w_ij = tf.math.log(tf.reduce_sum(w_ilj, axis=1)) + tf.squeeze(max_log_w_ilj, axis=1) - tf.math.log(tf.constant([self.args.N], dtype=tf.float32))
        # calculate the log(phat(y|theta, u))
        log_phat = tf.reduce_sum(w_ij)

        # calculate logp(theta)
        log_prior = theta_prior.log_prob(theta)

        # calculate the log prob value
        log_prob = log_prior + log_phat - 0.5*tf.tensordot(u, u, axes=1) 

        return log_prob
        
@ pytest.fixture
def mock_from_pickle():
    args = mock_args()
    def _mock_from_pickle(path, T=args.T, n=args.n, p=args.p, Z=args.Z, Y=args.Y):
        if 'Z' in path:
            return Z
        elif 'Y' in path:
            return Y
    return _mock_from_pickle

def test_grad_total(mocker, mock_from_pickle):
    '''
    integration test for calculate_grad.grad_total
    '''
    args = mock_args()
    mocker.patch('pseudo_marginal.grad.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.grad.os.path.exists', return_value=True)
    # coords = tf.constant([10.0, 20.0, 8.0, -0.1, -4.0], dtype=tf.float32)
    # coords = tf.concat([tf.random.normal(shape=[args.p]), coords, tf.random.normal(shape=[args.target_dim+2*args.aux_dim])], axis=0)
    coords = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)
    ham = mock_Hamiltonian_func(args, args.Y, args.Z)
    func = ham.gauss_mix_func
    cal = calculate_grad(args)
    grad_func = cal.grad_total
    result = grad_func(coords)
    expected = tfp.math.value_and_gradient(func, coords)[-1]
    print('expected', expected)
    print('result', result)
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected, result))

def test_calculate_H(mocker, mock_from_pickle):
    '''
    integration test for calculate_grad.calculate_H
    '''
    args = mock_args()
    mocker.patch('pseudo_marginal.grad.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.grad.os.path.exists', return_value=True)
    coords = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)
    cal = calculate_grad(args)
    func = cal.calculate_H
    result = func(coords)
    ham = mock_Hamiltonian_func(args, args.Y, args.Z)
    func_expected = ham.gauss_mix_func
    expected = func_expected(coords)
    print('result', result)
    print('expected', expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected, result))

def test_get_target_log_prob(mocker, mock_from_pickle):
    '''
    integration test for calculate_grad.get_target_log_prob
    '''
    args = mock_args()
    mocker.patch('pseudo_marginal.grad.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.grad.os.path.exists', return_value=True)
    coords = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)
    cal = calculate_grad(args)
    func = cal.get_target_log_prob
    result = func(tf.concat([coords[:args.target_dim], coords[2*args.target_dim:2*args.target_dim+args.aux_dim]], axis=-1))
    ham = mock_Hamiltonian_func(args, args.Y, args.Z)
    func_expected = ham.get_target_log_prob
    expected = func_expected(coords)
    print('result', result)
    print('expected', expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected, result))

def test_calculate_H_mass(mocker, mock_from_pickle):
    '''
    integration test for calculate_grad_mass.calculate_H
    '''
    args = mock_args()
    mocker.patch('pseudo_marginal.grad.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.grad.os.path.exists', return_value=True)
    coords = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)
    cal = calculate_grad_mass(args)
    func = cal.calculate_H
    result = func(coords)
    ham = mock_Hamiltonian_func(args, args.Y, args.Z)
    func_expected = ham.gauss_mix_func
    expected = func_expected(coords)
    print('result', result)
    print('expected', expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected, result))

def test_grad_total_mass(mocker, mock_from_pickle):
    '''
    integration test for calculate_grad_mass.grad_total
    '''
    args = mock_args()
    mocker.patch('pseudo_marginal.grad.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.grad.os.path.exists', return_value=True)
    # coords = tf.constant([10.0, 20.0, 8.0, -0.1, -4.0], dtype=tf.float32)
    # coords = tf.concat([tf.random.normal(shape=[args.p]), coords, tf.random.normal(shape=[args.target_dim+2*args.aux_dim])], axis=0)
    coords = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)
    ham = mock_Hamiltonian_func(args, args.Y, args.Z)
    func = ham.gauss_mix_func
    cal = calculate_grad_mass(args)
    grad_func = cal.grad_total
    result = grad_func(coords)
    expected = tfp.math.value_and_gradient(func, coords)[-1]
    print('expected', expected)
    print('result', result)
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected, result))

def test_get_target_log_prob_mass(mocker, mock_from_pickle):
    '''
    integration test for calculate_grad_mass.get_target_log_prob
    '''
    args = mock_args()
    mocker.patch('pseudo_marginal.grad.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.grad.os.path.exists', return_value=True)
    coords = tf.random.normal(shape=[2*(args.target_dim+args.aux_dim)], dtype=tf.float32)
    cal = calculate_grad_mass(args)
    func = cal.get_target_log_prob
    result = func(tf.concat([coords[:args.target_dim], coords[2*args.target_dim:2*args.target_dim+args.aux_dim]], axis=-1))
    ham = mock_Hamiltonian_func(args, args.Y, args.Z)
    func_expected = ham.get_target_log_prob
    expected = func_expected(coords)
    print('result', result)
    print('expected', expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(expected, result))

def log_prior(theta):
    return tf.tensordot(theta, theta, axes=1)

def log_phat(theta, u):
    return tf.tensordot(theta, theta, axes=1) + 2.0 * tf.tensordot(u, u, axes=1)

args = mock_args()
def mock_H_func(coords):
    rho_precision_mat = tf.linalg.diag(1.0 / tf.constant(args.rho_var, dtype=tf.float32))
    theta, rho, u, p = tf.split(coords, [args.target_dim, args.target_dim,
                                         args.aux_dim, args.aux_dim], axis=0)
    rho_expand = tf.expand_dims(rho, axis=-1)
    H = -log_prior(theta) -log_phat(theta, u) + 0.5 * tf.squeeze(tf.matmul(tf.matmul(rho_expand, rho_precision_mat, transpose_a=True), rho_expand)) \
        + 0.5 * (tf.tensordot(u, u, axes=1) + tf.tensordot(p, p, axes=1))
    return H

def test_integrator_one_step_mass():
    '''
    integration test for integrator_one_step_mass with numerical_grad
    '''
    args = mock_args()
    coords = tf.random.normal(shape=[args.input_dim], dtype=tf.float32)
    h = 0.05
    result = integrator_one_step_mass(coords, mock_H_func, numerical_grad, h, args.target_dim, args.aux_dim, args)

    # expected: handwritten version of Appendix. A
    theta, rho, u, p = tf.split(coords, [args.target_dim, args.target_dim,
                                         args.aux_dim, args.aux_dim], axis=0)
    new_theta = theta + h * rho + 0.5 * (h**2)  * (tfp.math.value_and_gradient(log_prior, theta + 0.5*h*rho)[-1] + 
                                                   tfp.math.value_and_gradient(log_phat, [theta + 0.5*h*rho, p*math.sin(0.5*h) + u*math.cos(0.5*h)])[-1][0])
    new_rho = rho + h * (tfp.math.value_and_gradient(log_prior, theta + 0.5*h*rho)[-1] + 
                         tfp.math.value_and_gradient(log_phat, [theta + 0.5*h*rho, p*math.sin(0.5*h) + u*math.cos(0.5*h)])[-1][0])
    new_u = p * math.sin(h) + u * math.cos(h) + math.sin(0.5*h) * h * tfp.math.value_and_gradient(log_phat, [theta + 0.5*h*rho, p*math.sin(0.5*h) + u*math.cos(0.5*h)])[-1][-1]
    new_p = p * math.cos(h) - u * math.sin(h) + math.cos(0.5*h) * h * tfp.math.value_and_gradient(log_phat, [theta + 0.5*h*rho, p*math.sin(0.5*h) + u*math.cos(0.5*h)])[-1][-1]
    expected = tf.concat([new_theta, new_rho, new_u, new_p], axis=0)
    print(result)
    print(expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(result, expected))

@pytest.fixture
def mock_integrator_one_step_mass():
    def _mock_integrator_one_step_mass(coords, func, derivs_func, h, target_dim, aux_dim, args):
        return coords * h
    return _mock_integrator_one_step_mass

def test_integrator_mass(mocker, mock_integrator_one_step_mass):
    '''
    unit test for integrator_mass with integrator_one_step_mass mocked
    '''
    mocker.patch('pseudo_marginal.grad.integrator_one_step_mass', mock_integrator_one_step_mass)
    h = 2
    steps = 2
    coords = tf.random.normal(shape=[10], dtype=tf.float32)
    args = mock_args()
    result = integrator_mass(coords, None, None, h, steps, None, None, args)
    expected = tf.stack([coords, coords*h, coords*(h**2)], axis=-1)
    print(result)
    print(expected)
    assert tf.reduce_all(tf.equal(result, expected))