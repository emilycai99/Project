import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import torch

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from pseudo_marginal.nn_models import MLP, CNN_MLP, Info_MLP, Info_CNN_MLP, choose_nonlinearity

@pytest.fixture
def mock_choose_nonlinearity():
    def _mock_choose_nonlinearity(nonlinearity):
        return tf.sin
    return _mock_choose_nonlinearity

def test_MLP(mocker, mock_choose_nonlinearity):
    '''
    unit test for MLP model by checking the output size
    '''
    input_dim = 10
    hidden_dim = 20
    num_layers = 3
    output_dim = 3
    batch_size = 2
    mocker.patch('pseudo_marginal.nn_models.choose_nonlinearity', mock_choose_nonlinearity)
    mlp = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    input = tf.random.normal(shape=[batch_size, input_dim], dtype=tf.float32)
    out = mlp(input)
    assert (out.shape[0] == batch_size) and (out.shape[1] == output_dim)

def test_CNN_MLP(mocker, mock_choose_nonlinearity):
    '''
    unit test for convolution-based model by checking the output size
    '''
    input_dim = 2 * (500 * 128 + 13)
    hidden_dim = 20
    num_layers = 3
    output_dim = 3
    batch_size = 2
    mocker.patch('pseudo_marginal.nn_models.choose_nonlinearity', mock_choose_nonlinearity)
    mlp = CNN_MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    input = tf.random.normal(shape=[batch_size, input_dim], dtype=tf.float32)
    out = mlp(input)
    assert (out.shape[0] == batch_size) and (out.shape[1] == output_dim)

def test_Info_MLP(mocker, mock_choose_nonlinearity):
    '''
    unit test for MLP model that focuses on learning the target density only by checking the output size
    '''
    input_dim = 2 * (500 * 128 + 13)
    hidden_dim = 20
    num_layers = 3
    output_dim = 3
    batch_size = 2
    mocker.patch('pseudo_marginal.nn_models.choose_nonlinearity', mock_choose_nonlinearity)
    mlp = Info_MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    input = tf.random.normal(shape=[batch_size, input_dim], dtype=tf.float32)
    out = mlp(input)
    assert (out.shape[0] == batch_size) and (out.shape[1] == output_dim)

def test_Info_CNN_MLP(mocker, mock_choose_nonlinearity):
    '''
    unit test for convolution-based model that focuses on learning the target density only by checking the output size
    '''
    input_dim = 2 * (500 * 128 + 13)
    hidden_dim = 20
    num_layers = 3
    output_dim = 3
    batch_size = 2
    mocker.patch('pseudo_marginal.nn_models.choose_nonlinearity', mock_choose_nonlinearity)
    mlp = Info_CNN_MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    input = tf.random.normal(shape=[batch_size, input_dim], dtype=tf.float32)
    out = mlp(input)
    assert (out.shape[0] == batch_size) and (out.shape[1] == output_dim)



