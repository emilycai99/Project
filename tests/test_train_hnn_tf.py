import pytest
from pytest_mock import mocker
import tensorflow as tf
import keras
import torch
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.train_hnn_tf import train
from tf_version.hnn_tf import HNN
from tf_version.nn_models_tf import MLP
from paper_code.train_hnn import train as train_pytorch
from paper_code.hnn import HNN as HNN_pytorch
from paper_code.nn_models import MLP as MLP_pytorch

class temp_args:
    def __init__(self):
        self.input_dim = 4
        self.hidden_dim = 10
        self.nonlinearity = 'sine'
        self.num_layers = 3
        self.grad_type = 'solenoidal'
        self.learn_rate = 0.01
        self.seed = 0
        self.batch_size_test=2
        self.verbose = True
        self.print_every = 1
        self.total_steps = 0
        self.train_samples = 10
        self.batch_size = self.train_samples
        self.test_samples = 10
        self.dist_name = 'nD_standard_Gaussian'

args = temp_args()
def build_model():
    # initialize tf version model
    nn_model = MLP(args.input_dim, args.hidden_dim, args.input_dim, args.nonlinearity,
                    num_layers=args.num_layers)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                grad_type=args.grad_type, baseline=False)
    model(tf.random.normal(shape=(1, args.input_dim)))

    # initialize pytorch model
    nn_model_pytorch = MLP_pytorch(args.input_dim, args.hidden_dim, args.input_dim, args.nonlinearity,
                    num_layers=args.num_layers)
    # copy the weights
    tf_weights = {x.name: x.numpy() for x in model.trainable_weights}
    keys = list(tf_weights.keys())
    for i, (name, x) in enumerate(nn_model_pytorch.named_parameters()):
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
    model_pytorch = HNN_pytorch(args.input_dim, differentiable_model=nn_model_pytorch,
                grad_type=args.grad_type, baseline=False)
    
    return model, model_pytorch

model1, model_pytorch1 = build_model()

# These mockers are not for unit test but integration test
# ensure the two models have the same weights and parameters
@pytest.fixture
def mock_hnn():
    def _mock_hnn(*args, **kwargs):
        return model1
    return _mock_hnn

@pytest.fixture
def mock_hnn_pytorch():
    def _mock_hnn_pytorch(*args, **kwargs):
        return model_pytorch1
    return _mock_hnn_pytorch

# create some data
data = {'coords': np.random.normal(size=(args.train_samples, args.input_dim)).astype(np.float32),
        'dcoords': np.random.normal(size=(args.train_samples, args.input_dim)).astype(np.float32),
        'test_coords': np.random.normal(size=(args.test_samples, args.input_dim)).astype(np.float32),
        'test_dcoords':np.random.normal(size=(args.test_samples, args.input_dim)).astype(np.float32)}

@pytest.fixture
def mock_get_dataset():
    def _mock_get_dataset(*args, **kwargs):
        tmp = {
            'coords': tf.constant(data['coords']),
            'dcoords': tf.constant(data['dcoords']),
            'test_coords': tf.constant(data['test_coords']),
            'test_dcoords': tf.constant(data['test_dcoords'])
        }
        return tmp
    return _mock_get_dataset

@pytest.fixture
def mock_get_dataset_pytorch():
    def _mock_get_dataset_pytorch(*args, **kwargs):
        return data
    return _mock_get_dataset_pytorch

def test_train(mocker, mock_hnn, mock_hnn_pytorch, mock_get_dataset, mock_get_dataset_pytorch):
    '''
    integration test for train function
    by comparing the result with the pytorch version;
    these mockers just to ensure that there are no discrepancies caused by randomness
    '''
    mocker.patch('tf_version.train_hnn_tf.HNN', mock_hnn)
    mocker.patch('paper_code.train_hnn.HNN', mock_hnn_pytorch)
    mocker.patch('tf_version.train_hnn_tf.get_dataset_tf', mock_get_dataset)
    mocker.patch('paper_code.train_hnn.get_dataset', mock_get_dataset_pytorch)
    mocker.patch('tf_version.functions_tf.args', args)
    mocker.patch('paper_code.functions.args', args)
    
    out_model, out_stats = train(args)
    out_model_pytorch, out_stats_pytorch = train_pytorch(args)

    # test whether the loss are the same
    assert np.allclose(out_stats['train_loss'][0], out_stats_pytorch['train_loss'][0], atol=1e-3)
    assert np.allclose(out_stats['test_loss'][0], out_stats_pytorch['test_loss'][0], atol=1e-3)

    # test whether the weights of the model are the same after update
    tf_weights = {x.name: x.numpy() for x in out_model.trainable_weights}
    keys = list(tf_weights.keys())
    for i, (name, x) in enumerate(out_model_pytorch.named_parameters()):
        if 'weight' in name:
            # print(np.max(x.data.detach().clone().numpy()-np.transpose(tf_weights[keys[i]])))
            assert np.allclose(x.data.detach().clone().numpy(), np.transpose(tf_weights[keys[i]]), atol=1e-1)
        elif 'bias' in name:
            assert np.allclose(x.data.detach().clone().numpy(), tf_weights[keys[i]], atol=1e-1)

    

    

    




