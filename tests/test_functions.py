import pytest
from pytest_mock import mocker
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import math
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.functions import Hamiltonian_func as Hamiltonian_func_tfp 
from pseudo_marginal.functions import Hamiltonian_func_debug as Hamiltonian_func_debug_tfp 
from pseudo_marginal.dist_generation import generate_GLMM

class mock_args():
    def __init__(self):
        self.T = 2
        self.n = 3
        self.N = 4
        self.p = 8
        self.data_pth = ''
        self.dist_name = 'gauss_mix'
        self.target_dim = self.p + 5
        self.aux_dim = self.T * self.N
        tf.random.set_seed(42)
        _, self.Y, self.Z = generate_GLMM(T=self.T, n=self.n, p=self.p, save_flag=False)

@ pytest.fixture
def mock_from_pickle():
    args = mock_args()
    def _mock_from_pickle(path, T=args.T, n=args.n, p=args.p, Z=args.Z, Y=args.Y):
        if 'Z' in path:
            return Z
        elif 'Y' in path:
            return Y
    return _mock_from_pickle

def normal_pdf(x, mu, sigma):
    out = 1 / (sigma * math.sqrt(2.0 * math.pi)) * \
            tf.exp(-0.5 * ((x - mu)/sigma)**2)
    return out

def gauss_mixture_pdf(x, w, mu1, sigma1, mu2, sigma2):
    out = w * normal_pdf(x, mu1, sigma1) + (1-w) * normal_pdf(x, mu2, sigma2)
    return out

def bernoulli_pdf(x, logit):
    p = tf.math.sigmoid(logit)
    return tf.math.pow(p, float(x)) * tf.math.pow(1-p, float(1-x))

def test_Hamiltonian_func_debug_tfp_get_func(mocker, mock_from_pickle):
    'test for Hamitonian function'
    args = mock_args()
    mocker.patch('pseudo_marginal.functions.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.functions.os.path.exists', return_value=True)
    ham = Hamiltonian_func_debug_tfp(args)
    theta = tf.constant([0.0, 0.0, math.log(1.0), math.log(1.0), 1.0], dtype=tf.float32)
    theta = tf.concat([tf.random.normal(shape=[args.p], dtype=tf.float32), theta], axis=0)
    rho = tf.random.normal(shape=[args.target_dim], dtype=tf.float32)
    u = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)
    p = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)

    coords = tf.concat([theta, rho, u, p], axis=0)
    out = ham.get_func(coords)

    p_theta = 1.0
    for i in range(theta.shape[0]):
        p_theta = p_theta * normal_pdf(theta[i], 0.0, 10.0)

    phat = 1.0
    for i in range(args.T):
        for j in range(args.n):
            phat_ij = 0.0
            for l in range(args.N):
                Xil = 3 * u[i*args.N+l]
                logit = Xil + tf.tensordot(args.Z[i, j, :], theta[:args.p], axes=1)
                g_val = bernoulli_pdf(args.Y[i][j], logit)
                f_val = gauss_mixture_pdf(Xil, 0.5, 0.0, 1.0, 0.0, 1.0)
                q_val = normal_pdf(Xil, 0.0, 3.0)
                phat_ij += g_val * f_val / q_val
            phat_ij = phat_ij / args.N
            phat *= phat_ij
    H = -tf.math.log(p_theta) -tf.math.log(phat) + 0.5 * tf.tensordot(rho, rho, axes=1) +\
        0.5 * tf.tensordot(u, u, axes=1) + 0.5 * tf.tensordot(p, p, axes=1)

    print(H, 'H')
    print(out, 'out')
    assert tf.experimental.numpy.allclose(H, out)

def test_Hamiltonian_func_tfp_get_func(mocker, mock_from_pickle):
    'test for Hamitonian function'
    args = mock_args()
    mocker.patch('pseudo_marginal.functions.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.functions.os.path.exists', return_value=True)
    ham = Hamiltonian_func_tfp(args)
    theta = tf.constant([0.0, 0.0, math.log(1.0), math.log(1.0), 1.0], dtype=tf.float32)
    theta = tf.concat([tf.random.normal(shape=[args.p], dtype=tf.float32), theta], axis=0)
    rho = tf.random.normal(shape=[args.target_dim], dtype=tf.float32)
    u = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)
    p = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)

    coords = tf.concat([theta, rho, u, p], axis=0)
    out = ham.get_func(coords)

    p_theta = 1.0
    for i in range(theta.shape[0]):
        p_theta = p_theta * normal_pdf(theta[i], 0.0, 10.0)

    phat = 1.0
    for i in range(args.T):
        for j in range(args.n):
            phat_ij = 0.0
            for l in range(args.N):
                Xil = 3 * u[i*args.N+l]
                logit = Xil + tf.tensordot(args.Z[i, j, :], theta[:args.p], axes=1)
                g_val = bernoulli_pdf(args.Y[i][j], logit)
                f_val = gauss_mixture_pdf(Xil, 0.5, 0.0, 1.0, 0.0, 1.0)
                q_val = normal_pdf(Xil, 0.0, 3.0)
                phat_ij += g_val * f_val / q_val
            phat_ij = phat_ij / args.N
            phat *= phat_ij
    H = -tf.math.log(p_theta) -tf.math.log(phat) + 0.5 * tf.tensordot(rho, rho, axes=1) +\
        0.5 * tf.tensordot(u, u, axes=1) + 0.5 * tf.tensordot(p, p, axes=1)

    print(H, 'H')
    print(out, 'out')
    assert tf.experimental.numpy.allclose(H, out)

def test_Hamiltonian_func_tfp_get_target_log_prob_func(mocker, mock_from_pickle):
    'test for get_target_log_prob_func function in Hamiltonian_func_tfp'
    args = mock_args()
    mocker.patch('pseudo_marginal.functions.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.functions.os.path.exists', return_value=True)
    ham = Hamiltonian_func_tfp(args)
    theta = tf.constant([0.0, 0.0, math.log(1.0), math.log(1.0), 1.0], dtype=tf.float32)
    theta = tf.concat([tf.random.normal(shape=[args.p], dtype=tf.float32), theta], axis=0)
    rho = tf.random.normal(shape=[args.target_dim], dtype=tf.float32)
    u = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)
    p = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)

    out = ham.get_target_log_prob_func(tf.concat([theta, u], axis=0))

    p_theta = 1.0
    for i in range(theta.shape[0]):
        p_theta = p_theta * normal_pdf(theta[i], 0.0, 10.0)

    phat = 1.0
    for i in range(args.T):
        for j in range(args.n):
            phat_ij = 0.0
            for l in range(args.N):
                Xil = 3 * u[i*args.N+l]
                logit = Xil + tf.tensordot(args.Z[i, j, :], theta[:args.p], axes=1)
                g_val = bernoulli_pdf(args.Y[i][j], logit)
                f_val = gauss_mixture_pdf(Xil, 0.5, 0.0, 1.0, 0.0, 1.0)
                q_val = normal_pdf(Xil, 0.0, 3.0)
                phat_ij += g_val * f_val / q_val
            phat_ij = phat_ij / args.N
            phat *= phat_ij
    target_log_prob = tf.math.log(p_theta) + tf.math.log(phat) - 0.5 * tf.tensordot(u, u, axes=1) 

    print(target_log_prob, 'target_log_prob')
    print(out, 'out')
    assert tf.experimental.numpy.allclose(target_log_prob, out)


def test_Hamiltonian_func_debug_tfp_get_target_log_prob_func(mocker, mock_from_pickle):
    'test for get_target_log_prob_func function in Hamiltonian_func_debug_tfp'
    args = mock_args()
    mocker.patch('pseudo_marginal.functions.from_pickle', mock_from_pickle)
    mocker.patch('pseudo_marginal.functions.os.path.exists', return_value=True)
    ham = Hamiltonian_func_debug_tfp(args)
    theta = tf.constant([0.0, 0.0, math.log(1.0), math.log(1.0), 1.0], dtype=tf.float32)
    theta = tf.concat([tf.random.normal(shape=[args.p], dtype=tf.float32), theta], axis=0)
    rho = tf.random.normal(shape=[args.target_dim], dtype=tf.float32)
    u = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)
    p = tf.random.normal(shape=[args.aux_dim], dtype=tf.float32)

    out = ham.get_target_log_prob_func(tf.concat([theta, u], axis=0))

    p_theta = 1.0
    for i in range(theta.shape[0]):
        p_theta = p_theta * normal_pdf(theta[i], 0.0, 10.0)

    phat = 1.0
    for i in range(args.T):
        for j in range(args.n):
            phat_ij = 0.0
            for l in range(args.N):
                Xil = 3 * u[i*args.N+l]
                logit = Xil + tf.tensordot(args.Z[i, j, :], theta[:args.p], axes=1)
                g_val = bernoulli_pdf(args.Y[i][j], logit)
                f_val = gauss_mixture_pdf(Xil, 0.5, 0.0, 1.0, 0.0, 1.0)
                q_val = normal_pdf(Xil, 0.0, 3.0)
                phat_ij += g_val * f_val / q_val
            phat_ij = phat_ij / args.N
            phat *= phat_ij
    target_log_prob = tf.math.log(p_theta) + tf.math.log(phat) - 0.5 * tf.tensordot(u, u, axes=1) 

    print(target_log_prob, 'target_log_prob')
    print(out, 'out')
    assert tf.experimental.numpy.allclose(target_log_prob, out)







        






        





        






        
