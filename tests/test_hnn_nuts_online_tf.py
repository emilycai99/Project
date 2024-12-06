import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import keras
import torch

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from tf_version.hnn_nuts_online_tf import build_tree_tf, integrate_model_tf, stop_criterion_tf
from tf_version.hnn_tf import HNN
from tf_version.nn_models_tf import MLP
from tf_version.get_args import get_args
from tf_version.hnn_nuts_online_tf import sample
from paper_code.hnn_nuts_online import build_tree, stop_criterion
from paper_code.hnn_nuts_online import sample as sample_pytorch
from paper_code.hnn import HNN as HNN_pytorch
from paper_code.nn_models import MLP as MLP_pytorch

@pytest.fixture
def mock_integrate_model_tf():
    def _mock_integrate_model_tf(model, t_span, y0, n, args):
        return tf.repeat(tf.expand_dims(y0, axis=-1), repeats=n+1, axis=-1)
    return _mock_integrate_model_tf

@pytest.fixture
def mock_integrate_model():
    def _mock_integrate_model(model, t_span, y0, n, args, **kwargs):
        return np.repeat(np.expand_dims(y0, axis=-1), repeats=n+1, axis=-1)
    return _mock_integrate_model

@pytest.fixture
def mock_functions_tf():
    def _mock_functions_tf(x):
        return 5.0
    return _mock_functions_tf

@pytest.fixture
def mock_stop_criterion_tf():
    def _mock_stop_criterion_tf(*args):
        return True
    return _mock_stop_criterion_tf

@pytest.fixture
def mock_random_uniform():
    def _mock_random_uniform(**kwargs):
        return 0.0
    return _mock_random_uniform


def test_build_tree_tf(mocker, mock_integrate_model_tf, mock_functions_tf,
                       mock_stop_criterion_tf, setup_args, mock_random_uniform, mock_integrate_model):
    '''
    unit test for build_tree_tf
    '''
    # mock tf_version.hnn_nuts_online_tf file
    mocker.patch('tf_version.hnn_nuts_online_tf.integrate_model_tf', mock_integrate_model_tf)
    mocker.patch('tf_version.hnn_nuts_online_tf.functions_tf', mock_functions_tf)
    mocker.patch('tf_version.hnn_nuts_online_tf.stop_criterion_tf', mock_stop_criterion_tf)
    # mocker.patch("tf_version.hnn_nuts_online_tf.args", setup_args('2D_Gauss_mix'))
    mocker.patch("tf_version.functions_tf.args", setup_args('2D_Gauss_mix'))
    mocker.patch("tf_version.hnn_nuts_online_tf.tf.random.uniform", mock_random_uniform)

    # mock paper_code.hnn_nuts_online
    mocker.patch('paper_code.hnn_nuts_online.integrate_model', mock_integrate_model)
    mocker.patch('paper_code.hnn_nuts_online.functions', mock_functions_tf)
    mocker.patch('paper_code.hnn_nuts_online.stop_criterion', mock_stop_criterion_tf)
    # mocker.patch("paper_code.hnn_nuts_online.args", setup_args('2D_Gauss_mix'))
    mocker.patch("paper_code.hnn_nuts_online.np.random.uniform", mock_random_uniform)

    theta = tf.random.normal(shape=[2,], dtype=tf.float32)
    r = tf.random.normal(shape=[2,], dtype=tf.float32)
    thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf \
          = build_tree_tf(theta, r, u=0.4, v=1, j=1, epsilon=0.5, joint0=4, call_lf=1, hnn_model=None, args=setup_args('2D_Gauss_mix'))
    thetaminus_np, rminus_np, thetaplus_np, rplus_np, thetaprime_np, rprime_np, nprime_np, sprime_np, alphaprime_np, nalphaprime_np, monitor_np, call_lf_np \
          = build_tree(theta.numpy(), r.numpy(), logu=0.4, v=1, j=1, epsilon=0.5, joint0=4.0, call_lf=1, hnn_model=None, args=setup_args('2D_Gauss_mix'))
    
    assert np.allclose(thetaminus.numpy(), thetaminus_np)
    assert np.allclose(rminus.numpy(), rminus_np)
    assert np.allclose(thetaplus.numpy(), thetaplus_np)
    assert np.allclose(rplus.numpy(), rplus_np)
    assert np.allclose(thetaprime.numpy(), thetaprime_np)
    assert np.allclose(rprime.numpy(), rprime_np)
    assert nprime == nprime_np
    assert sprime == sprime_np
    assert np.allclose(alphaprime.numpy(), alphaprime_np)
    assert nalphaprime == nalphaprime_np
    assert np.allclose(monitor.numpy(), monitor_np)
    assert call_lf == call_lf_np

class temp_hnn(keras.Model):
    def __init__(self, input_dim, *args, **kwargs):
        super(temp_hnn, self).__init__()
        self.input_dim = input_dim
        self.w = self.add_weight(shape=[1], initializer=keras.initializers.Ones(),
                                 trainable=True, name='w')

    def __call__(self, x):
        return x
    
    def time_derivative(self, x):
        return self.w * x 

model = temp_hnn(4)
@pytest.fixture
def mock_hnn():
    def _mock_hnn():
        return model
    return _mock_hnn

@pytest.fixture
def mock_leapfrog_tf():
    def _mock_leapfrog_tf(fun, t_span, y0, n, dim):
        tmp = tf.repeat(tf.expand_dims(fun(None, y0), axis=-1), repeats=n+1, axis=-1)
        return tmp
    return _mock_leapfrog_tf

def test_integrate_model_tf(mocker, mock_hnn, mock_leapfrog_tf, setup_args):
    '''
    unit test to check integrate_model_tf function
    '''
    mocker.patch('tf_version.hnn_nuts_online_tf.leapfrog_tf', mock_leapfrog_tf)
    y0 = tf.random.normal(shape=[4])
    result = integrate_model_tf(mock_hnn(), [0, 1],  y0, 1, args=setup_args('2D_Gauss_mix'))
    expected = tf.repeat(tf.expand_dims(y0, axis=-1), repeats=2, axis=-1)
    assert tf.reduce_all(tf.equal(result, expected)).numpy()

def test_stop_criterion_tf():
    '''
    unit test to check stop_criterion_tf function
    by comparing the output results with those from pytorch version
    '''
    input_dim = 2
    theta_minus = tf.random.normal(shape=[input_dim])
    theta_plus = tf.random.normal(shape=[input_dim])
    r_minus = tf.random.normal(shape=[input_dim])
    r_plus = tf.random.normal(shape=[input_dim])
    results = stop_criterion_tf(theta_minus, theta_plus, r_minus, r_plus)
    expected = stop_criterion(theta_minus.numpy(), theta_plus.numpy(), r_minus.numpy(), r_plus.numpy())
    assert results.numpy() == expected

args = get_args()
args.batch_size = 2
args.hidden_dim = 10
args.num_hmc_samples = 3
args.num_burnin_samples = 0
args.total_steps = 0

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
        if 'weight' in name:
            x.data = (torch.from_numpy(np.transpose(tf_weights[keys[i]])))
        elif 'bias' in name:
            x.data = (torch.from_numpy(tf_weights[keys[i]]))
    model_pytorch = HNN_pytorch(args.input_dim, differentiable_model=nn_model_pytorch,
                grad_type=args.grad_type, baseline=False)
    
    return model, model_pytorch

model1, model_pytorch1 = build_model()

# These mockers are not for unit test but integration test
# ensure the two models have the same weights and parameters
@pytest.fixture
def build_hnn():
    def _build_hnn(*args, **kwargs):
        return model1
    return _build_hnn

@pytest.fixture
def build_hnn_pytorch():
    def _build_hnn_pytorch(*args, **kwargs):
        return model_pytorch1
    return _build_hnn_pytorch


@pytest.fixture()
def mock_tf_random_uniform():
    def _mock_tf_random_uniform(shape, minval=0.0, maxval=1.0):
        return tf.constant([0.5], dtype=tf.float32)
    return _mock_tf_random_uniform

@pytest.fixture()
def mock_np_random_uniform():
    def _mock_np_random_uniform(low=0.0, high=1.0):
        return np.array([0.5])
    return _mock_np_random_uniform

@pytest.fixture()
def mock_tf_random_normal():
    def _mock_tf_random_normal(shape):
        tmp = [0.5 for _ in range(shape[0])]
        return tf.constant(tmp, dtype=tf.float32)
    return _mock_tf_random_normal

class tmp_norm():
    def rvs():
        return np.array([0.5])

@pytest.fixture()
def mock_np_random_normal():
    def _mock_np_random_normal(loc=0.0, scale=1.0):
        return tmp_norm
    return _mock_np_random_normal


def test_integration_hnn_nuts_online(mocker, build_hnn, build_hnn_pytorch, mock_tf_random_uniform,
                                     mock_np_random_uniform, mock_tf_random_normal, mock_np_random_normal):
    '''
    an integration test to check the correctness of hnn_nuts_online_tf
    by comparing the results with pytorch version
    '''

    mocker.patch('tf_version.hnn_nuts_online_tf.log_start')
    mocker.patch('tf_version.hnn_nuts_online_tf.log_stop')
    mocker.patch('tf_version.hnn_nuts_online_tf.np.save')
    mocker.patch('tf_version.hnn_nuts_online_tf.os.makedirs')
    mocker.patch('tf_version.hnn_nuts_online_tf.get_model', build_hnn)
    mocker.patch('tf_version.hnn_nuts_online_tf.tf.random.uniform', mock_tf_random_uniform)
    mocker.patch('tf_version.hnn_nuts_online_tf.tf.random.normal', mock_tf_random_normal)

    mocker.patch('paper_code.hnn_nuts_online.log_start')
    mocker.patch('paper_code.hnn_nuts_online.log_stop')
    mocker.patch('paper_code.hnn_nuts_online.np.save')
    mocker.patch('paper_code.hnn_nuts_online.os.makedirs')
    mocker.patch('paper_code.hnn_nuts_online.get_model', build_hnn_pytorch)
    mocker.patch('paper_code.hnn_nuts_online.np.random.uniform', mock_np_random_uniform)
    mocker.patch('paper_code.hnn_nuts_online.norm', mock_np_random_normal)

    samples, traj_len, alpha_req, H_store, monitor_err, is_lf = sample(args)
    samples_pytorch, traj_len_pytorch, alpha_req_pytorch, H_store_pytorch, monitor_err_pytorch, is_lf_pytorch = sample_pytorch(args)

    print(samples, traj_len, alpha_req, H_store, monitor_err, is_lf)
    assert np.allclose(samples.numpy(), samples_pytorch)
    assert np.array_equal(traj_len.numpy(), traj_len_pytorch)
    assert np.allclose(alpha_req.numpy(), alpha_req_pytorch)
    assert np.allclose(H_store.numpy(), H_store_pytorch)
    assert np.allclose(monitor_err.numpy(), monitor_err_pytorch)
    assert np.array_equal(is_lf.numpy(), is_lf_pytorch)
  
    






