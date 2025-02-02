import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import torch

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.hnn import HNN

class mock_args():
    def __init__(self):
        self.nn_out_dim = 12
        self.target_dim = 2
        self.aux_dim = 4

@pytest.fixture
def mock_differentiable_model():
    def _mock_differentiable_model(x):
        return x
    return _mock_differentiable_model

def test_call(mock_differentiable_model):
    '''
    unit test for call function in HNN
    '''
    args = mock_args()
    model = HNN(args, mock_differentiable_model)
    input = tf.random.normal(shape=[2, args.nn_out_dim], dtype=tf.float32)
    out1, out2 = model(input)
    assert tf.reduce_all(tf.equal(out1, input[:, :int(args.nn_out_dim/2)]))
    assert tf.reduce_all(tf.equal(out2, input[:, int(args.nn_out_dim/2):])) 

def test_time_derivative(mock_differentiable_model):
    '''
    unit test for time derivative function in HNN
        - compared with using the permutation tensor
    '''
    args = mock_args()
    # check the solenoidal type
    model = HNN(args, mock_differentiable_model)
    input_dim = 2*(args.target_dim + args.aux_dim)
    input = tf.random.normal(shape=[2, input_dim], dtype=tf.float32)
    model(input)
    out = model.time_derivative(input)

    # create the permutation tensor like the old function
    tmp_M = tf.eye(input_dim)
    M = tf.concat([tmp_M[args.target_dim:2*args.target_dim], 
                   -tmp_M[:args.target_dim],
                   tmp_M[-args.aux_dim:],
                   -tmp_M[-2 * args.aux_dim: -args.aux_dim]], axis=0)

    tmp1 = tf.ones(shape=[2, int(input_dim/2)], dtype=tf.float32)
    tmp2 = tf.concat([tf.zeros(shape=[2, int(input_dim/2)], dtype=tf.float32), tmp1], axis=-1)
    expected = tf.matmul(tmp2, tf.transpose(M))
    assert tf.reduce_all(tf.equal(out, expected))

    # check the conservative type
    model = HNN(args, mock_differentiable_model, grad_type='conservative')
    input_dim = 2*(args.target_dim + args.aux_dim)
    input = tf.random.normal(shape=[2, input_dim], dtype=tf.float32)
    model(input)
    out = model.time_derivative(input)

    # create the permutation tensor like the old function
    M = tf.eye(input_dim)
    tmp1 = tf.ones(shape=[2, int(input_dim/2)], dtype=tf.float32)
    tmp2 = tf.concat([tmp1, tf.zeros(shape=[2, int(input_dim/2)], dtype=tf.float32)], axis=-1)
    expected = tf.matmul(tmp2, tf.eye(*M.shape))
    print(out)
    print(expected)
    assert tf.reduce_all(tf.equal(out, expected))

