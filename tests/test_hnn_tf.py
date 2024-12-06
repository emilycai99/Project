import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import torch

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.hnn_tf import HNN
from paper_code.hnn import HNN as HNN_pytorch

input_dim = 4
batch_size = 2

@pytest.fixture
def mock_differentiable_model():
    def _mock_differentiable_model(x):
        return x
    return _mock_differentiable_model

def test_call(mock_differentiable_model):
    '''
    unit test for call function in HNN
    '''
    model = HNN(input_dim, mock_differentiable_model)
    x = tf.random.normal(shape=(batch_size, input_dim))
    out1, out2 = model(x)
    assert np.allclose(x.numpy()[:, :int(input_dim/2)], out1)
    assert np.allclose(x.numpy()[:, int(input_dim/2):], out2)

def test_permutation_tensor(mock_differentiable_model):
    '''
    unit test for permutation_tensor function by comparing with pytorch version
    '''
    model = HNN(input_dim, mock_differentiable_model)
    out = model.M
    model_pytorch = HNN_pytorch(input_dim, mock_differentiable_model)
    out_pytorch = model_pytorch.M
    assert np.allclose(out.numpy(), out_pytorch.numpy())

def test_time_derivative(mock_differentiable_model):
    '''
    unit test for time_derivative function by comparing with pytorch version
    '''
    model = HNN(input_dim, mock_differentiable_model)
    input = tf.random.normal(shape=(batch_size, input_dim))
    out = model.time_derivative(input)
    model_pytorch = HNN_pytorch(input_dim, mock_differentiable_model)
    input_pytorch = torch.from_numpy(input.numpy())
    input_pytorch.requires_grad = True
    out_pytorch = model_pytorch.time_derivative(input_pytorch)
    assert np.allclose(out.numpy(), out_pytorch.detach().clone().numpy())
    

