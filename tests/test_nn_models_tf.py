import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import torch

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.nn_models_tf import MLP
from paper_code.nn_models import MLP as MLP_pytorch

@pytest.fixture
def mock_choose_nonlinearity():
    def _mock_choose_nonlinearity(nonlinearity):
        return tf.sin
    return _mock_choose_nonlinearity

def test_MLP(mocker, mock_choose_nonlinearity):
    '''
    unit test MLP
    further test whether the MLP has the same output as the pytorch version
    '''
    mocker.patch('tf_version.nn_models_tf.choose_nonlinearity', mock_choose_nonlinearity)
    batch_size = 2
    input_dim = 4
    hidden_dim = 100
    output_dim = 4
    num_layers = 3
    model = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    input = tf.random.normal(shape=(batch_size, input_dim))
    out = model(input)
    assert (out.shape[0] == batch_size) and (out.shape[1] == input_dim)

    # initialize pytorch 
    model_pytorch = MLP_pytorch(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    # copy the weights
    tf_weights = {x.name: x.numpy() for x in model.trainable_weights}
    keys = list(tf_weights.keys())
    
    for i, (name, x) in enumerate(model_pytorch.named_parameters()):
        if name == 'layers.0.weight':
            x.data = (torch.from_numpy(np.transpose(tf_weights[keys[i]])))
        elif name == 'layers.0.bias':
            x.data = (torch.from_numpy(tf_weights[keys[i]]))
        elif name == 'layers.1.weight':
            x.data = (torch.from_numpy(np.transpose(tf_weights[keys[i]])))
        elif name == 'layers.1.bias':
            x.data = (torch.from_numpy(tf_weights[keys[i]]))
        elif name == 'layers.2.weight':
            x.data = (torch.from_numpy(np.transpose(tf_weights[keys[i]])))
    input_pytorch = torch.from_numpy(input.numpy())
    out_pytorch = model_pytorch(input_pytorch)
    assert np.allclose(out_pytorch.detach().clone().numpy(), out.numpy(), rtol=0, atol=1e-04)



