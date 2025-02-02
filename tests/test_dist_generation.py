import pytest
from pytest_mock import mocker
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import math
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.functions import generate_GLMM

def test_generate_GLMM():
    '''
    unit test of generate_GLMM
        by see whether the asymptotic mean and variance converge
        to theoretic values
    '''
    
    true_beta = [0.0, 0.0]
    true_mu = [0.0, 0.0]
    true_lambda = [1.0, 1.0]
    true_w = 0.8
    T = 2
    n = 100000
    p = 2
    save_flag=False
    tf.random.set_seed(seed=0)
    X, Y, Z = generate_GLMM(true_beta=true_beta, true_mu=true_mu, true_lambda=true_lambda,
                         true_w=true_w, T=T, n=n, p=p, save_flag=save_flag)
    
    assert tf.shape(X).numpy().tolist() == [T]
    assert tf.shape(Y).numpy().tolist() == [T, n]
    assert tf.shape(Z).numpy().tolist() == [T, n, p]

    # first examine the Y and Z by setting n to a very large value
    for i in range(T):
        assert tf.experimental.numpy.allclose(tf.reduce_mean(tf.cast(Y[i], tf.float32)), tf.math.sigmoid(X[i]), rtol=0.0, atol=1e-2)
        assert tf.experimental.numpy.allclose(tf.math.reduce_variance(tf.cast(Y[i], tf.float32)), 
                                              tf.math.sigmoid(X[i]) * (1-tf.math.sigmoid(X[i])), rtol=0.0, atol=1e-2)

    for i in range(T):
        for j in range(p):
            assert tf.experimental.numpy.allclose(tf.reduce_mean(Z[i, :, j]), 0.0, rtol=0.0, atol=1e-2)
            assert tf.experimental.numpy.allclose(tf.math.reduce_variance(Z[i, :, j]), 1.0, rtol=0.0, atol=1e-2)

    # now change T to a very large value
    n = 2
    T = 100000
    true_mu = [1.0, 2.0]
    true_lambda = [1.0, 2.0]
    X, Y, Z = generate_GLMM(true_beta=true_beta, true_mu=true_mu, true_lambda=true_lambda,
                         true_w=true_w, T=T, n=n, p=p, save_flag=save_flag)
    
    assert tf.experimental.numpy.allclose(tf.reduce_mean(X), true_w * true_mu[0] + (1-true_w) * true_mu[1], rtol=0.0, atol=1e-2)

    print('tf.math.reduce_variance(X)', tf.math.reduce_variance(X))
    print(true_w * true_lambda[0] ** (-1) + (1-true_w) * true_lambda[1] ** (-1) + 
                                          true_w * (1-true_w) * (true_mu[0] - true_mu[1])**2)
   
    assert tf.reduce_max(tf.math.abs(tf.math.reduce_variance(X) - (true_w * true_lambda[0] ** (-1) + (1-true_w) * true_lambda[1] ** (-1) + 
                                          true_w * (1-true_w) * (true_mu[0] - true_mu[1])**2))) < 1e-1
