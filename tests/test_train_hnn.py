import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import shutil
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from pseudo_marginal.train_hnn import train, l1_penalty
from pseudo_marginal.nn_models import MLP

class mock_args():
    def __init__(self):
        self.step_size = 0.02
        self.len_sample = 0.04
        self.target_dim = 13
        self.aux_dim = 1000
        self.input_dim = 2 * (self.target_dim + self.aux_dim)
        self.input = tf.random.normal(shape=[self.input_dim], dtype=tf.float32, seed=42)
        self.num_samples = 2
        self.should_load = False
        self.dist_name = 'gauss_mix'
        self.T = 500
        self.n = 6
        self.p = 8
        self.N = 2
        self.test_fraction = 1.0
        self.save_dir = os.path.join(PARENT_DIR, 'pseudo_marginal')
        self.batch_size = 2
        self.batch_size_test = 2
        self.shuffle_buffer_size = 2
        self.nn_model_name = 'mlp'
        self.learn_rate = 1e-4
        self.seed = 0
        self.total_steps = 2
        self.hidden_dim = 10
        self.nn_out_dim = 26
        self.nonlinearity = 'sine'
        self.num_layers = 3
        self.data_pth = os.path.join(PARENT_DIR, 'pseudo_marginal/data')
        self.grad_type = None
        self.penalty_strength = 0.0
        self.verbose = True
        self.print_every = 1
        self.grad_flag = False
        self.retrain = False

@pytest.fixture
def mock_MLP():
    def _mock_MLP(input_dim, hidden_dim, nn_out_dim, nonlinearity, num_layers=3):
        model =  MLP(input_dim, hidden_dim, nn_out_dim, nonlinearity, num_layers)
        for weight in model.trainable_weights:
            weight.assign(tf.ones_like(weight))
        return model
    return _mock_MLP

def test_train(mocker, mock_MLP):
    '''
    integration test for train function
    '''
    args = mock_args()
    save_path = '{}/ckp/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}_lr{}_{}'.format(args.save_dir, args.dist_name,
                                                                           args.T, args.n, args.p, args.N,
                                                                           args.num_samples, args.len_sample, 
                                                                           args.step_size, args.learn_rate, args.nn_model_name)
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    mocker.patch('pseudo_marginal.train_hnn.MLP', mock_MLP)
    model, best_model, stats = train(args, save_path)
    # check whether the model weights have been updated
    for weight in model.trainable_weights:
        assert not tf.reduce_all(tf.equal(weight, tf.ones_like(weight)))
    # check whether the best model is saved
    assert os.path.exists(save_path + '/checkpoint')
    # assert whether the train_loss and test_loss is logged and it is not nan
    assert (len(stats['train_loss']) == args.total_steps + 1) and (len(stats['test_loss']) == args.total_steps + 1)
    assert (not any(np.isnan(x) for x in stats['train_loss'])) and (not any(np.isnan(x) for x in stats['test_loss']))

    data_folder = '{}/data/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, 
                                                                      args.T, args.n, args.p, args.N,
                                                                      args.num_samples, args.len_sample, args.step_size)
    shutil.rmtree(data_folder)
    shutil.rmtree(save_path)

class Args:
    def __init__(self, nn_model_name, target_dim, aux_dim, penalty_strength):
        self.nn_model_name = nn_model_name
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.penalty_strength = penalty_strength

class MockModel:
    def __init__(self, trainable_weights):
        self.trainable_weights = trainable_weights

def test_l1_penalty():
    # Mock Args object
    args_cnn = Args(nn_model_name='cnn_model', target_dim=10, aux_dim=5, penalty_strength=0.1)
    args_info = Args(nn_model_name='info_model', target_dim=10, aux_dim=5, penalty_strength=0.1)
    args_default = Args(nn_model_name='default_model', target_dim=10, aux_dim=5, penalty_strength=0.1)

    # Mock Model object
    trainable_weights = [
        tf.Variable(tf.random.normal((15,))),
        tf.Variable(tf.random.normal((20,))),
        tf.Variable(tf.random.normal((30,)))
    ]
    model = MockModel(trainable_weights)

    assert l1_penalty(args_cnn, model) == 0.0

    assert l1_penalty(args_info, model) == 0.1 * tf.norm(trainable_weights[0], ord=1)

    expected_l1_norm = tf.norm(trainable_weights[2], ord=1)
    expected_result = 0.1 * expected_l1_norm

    assert l1_penalty(args_default, model) == expected_result

