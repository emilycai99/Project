import pytest
import pytest_mock
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import shutil

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.data import get_trajectory_tf, get_trajectory_tf_debug, get_dataset_tf, load_tensor, load_tensor_slices, get_dataset_loader

@pytest.fixture
def mock_integrator():
    def _mock_integrator(coords, func, derivs_func, h, steps, target_dim, aux_dim):
        out = [coords]
        for i in range(steps):
            out.append(coords * (h**(i+1)))
        return tf.stack(out, axis=-1)
    return _mock_integrator

@pytest.fixture
def mock_numerical_grad():
    def _mock_numerical_grad(coords, func, **kwargs):
        return tfp.math.value_and_gradient(func, coords)[-1]
    return _mock_numerical_grad

def mock_func(coords):
    return coords

class mock_args():
    def __init__(self):
        self.step_size = 2
        self.len_sample = 4
        self.target_dim = 3
        self.aux_dim = 2
        self.input_dim = 2 * (self.target_dim + self.aux_dim)
        self.input = tf.random.normal(shape=[self.input_dim], dtype=tf.float32, seed=42)
        self.num_samples = 2
        self.should_load = False
        self.dist_name = 'gauss_mix'
        self.T = None
        self.n = None
        self.p = None
        self.N = None
        self.test_fraction = 0.5
        self.save_dir = None
        self.batch_size = 2
        self.batch_size_test = 2
        self.shuffle_buffer_size = 2
        self.grad_flag = False

def test_get_trajectory_tf(mocker, mock_integrator, mock_numerical_grad):
    '''
    unit test for get_trajector_tf
    '''
    mocker.patch('pseudo_marginal.data.integrator', mock_integrator)
    mocker.patch('pseudo_marginal.data.numerical_grad', mock_numerical_grad)
    args = mock_args()
    input = tf.random.normal(shape=[args.input_dim], dtype=tf.float32)
    dic1, ddic1 = get_trajectory_tf(args, mock_func, y0=input)
    dic1_expected = [tf.stack([tf.expand_dims(input[d], axis=0) * (args.step_size ** i) for i in range(args.len_sample//args.step_size+1)], axis=1) for d in range(args.input_dim)]
    ddic1_expected = [tf.ones(shape=[1, args.len_sample//args.step_size+1], dtype=tf.float32) for _ in range(args.input_dim)]
    for i in range(len(dic1)):
        assert tf.reduce_all(tf.equal(dic1[i], dic1_expected[i]))
        assert tf.reduce_all(tf.equal(ddic1[i], ddic1_expected[i]))

@pytest.fixture
def mock_numerical_grad_debug():
    def _mock_numerical_grad_debug(coords, grad_func, target_dim, aux_dim):
        return grad_func(coords)
    return _mock_numerical_grad_debug

def test_get_trajectory_tf_debug(mocker, mock_integrator, mock_numerical_grad_debug):
    '''
    unit test for get_trajector_tf_debug (with manual gradients)
    '''
    mocker.patch('pseudo_marginal.data.integrator', mock_integrator)
    mocker.patch('pseudo_marginal.data.numerical_grad_debug', mock_numerical_grad_debug)
    args = mock_args()
    args.grad_flag = True
    input = tf.random.normal(shape=[args.input_dim], dtype=tf.float32)
    dic1, ddic1 = get_trajectory_tf_debug(args, mock_func, y0=input)
    dic1_expected = [tf.stack([tf.expand_dims(input[d], axis=0) * (args.step_size ** i) for i in range(args.len_sample//args.step_size+1)], axis=1) for d in range(args.input_dim)]
    ddic1_expected = [tf.stack([tf.expand_dims(input[d], axis=0) * (args.step_size ** i) for i in range(args.len_sample//args.step_size+1)], axis=1) for d in range(args.input_dim)]
    for i in range(len(dic1)):
        assert tf.reduce_all(tf.equal(dic1[i], dic1_expected[i]))
        assert tf.reduce_all(tf.equal(ddic1[i], ddic1_expected[i]))

@pytest.fixture
def mock_get_trajectory_tf():
    def _mock_get_trajectory_tf(args, func, y0=None, **kwargs):
        dic1_expected = [tf.stack([tf.expand_dims(args.input[d], axis=0) * (args.step_size ** i) for i in range(args.len_sample//args.step_size+1)], axis=1) for d in range(args.input_dim)]
        ddic1_expected = [tf.ones(shape=[1, args.len_sample//args.step_size+1], dtype=tf.float32) for _ in range(args.input_dim)]
        return dic1_expected, ddic1_expected
    return _mock_get_trajectory_tf


def test_get_dataset_tf(mocker, mock_get_trajectory_tf):
    '''
    unit test for get_dataset_tf
        - by comparing it with the original version
        - first exclude the data saving part (TODO: also check it later)
    '''
    args = mock_args()
    mocker.patch('pseudo_marginal.data.get_trajectory_tf', mock_get_trajectory_tf)
    mocker.patch('pseudo_marginal.data.to_pickle')
    mocker.patch('pseudo_marginal.data.os.makedirs')
    data = get_dataset_tf(args, func=None)

    # expected (old version)
    xs = []
    dxs = []
    for s in range(args.num_samples):
        dic1 = [tf.stack([tf.expand_dims(args.input[d], axis=0) * (args.step_size ** i) for i in range(args.len_sample//args.step_size+1)], axis=1) for d in range(args.input_dim)]
        ddic1 = [tf.ones(shape=[1, args.len_sample//args.step_size+1], dtype=tf.float32) for _ in range(args.input_dim)]
        dic1_tmp = tf.stack([tf.reshape(dic1[i], -1) for i in range(args.input_dim)], axis=1)
        xs.append(dic1_tmp)
        dxs.append(tf.stack([tf.reshape(ddic1[i], -1) for i in range(args.input_dim)], axis=1))
    coords_expected = tf.concat(xs, axis=0)
    dcoords_expected = tf.squeeze(tf.concat(dxs, axis=0))

    xs = []
    dxs = []
    for s in range(int(args.num_samples * args.test_fraction)):
        dic1 = [tf.stack([tf.expand_dims(args.input[d], axis=0) * (args.step_size ** i) for i in range(args.len_sample//args.step_size+1)], axis=1) for d in range(args.input_dim)]
        ddic1 = [tf.ones(shape=[1, args.len_sample//args.step_size+1], dtype=tf.float32) for _ in range(args.input_dim)]
        dic1_tmp = tf.stack([tf.reshape(dic1[i], -1) for i in range(args.input_dim)], axis=1)
        xs.append(dic1_tmp)
        dxs.append(tf.stack([tf.reshape(ddic1[i], -1) for i in range(args.input_dim)], axis=1))
    test_coords_expected = tf.concat(xs, axis=0)
    test_dcoords_expected = tf.squeeze(tf.concat(dxs, axis=0))

    assert tf.reduce_all(tf.equal(coords_expected, data['coords']))
    assert tf.reduce_all(tf.equal(dcoords_expected, data['dcoords']))
    assert tf.reduce_all(tf.equal(test_coords_expected, data['test_coords']))
    assert tf.reduce_all(tf.equal(test_dcoords_expected, data['test_dcoords']))

def test_load_tensor_slices(mocker):
    '''
    unit test for load_tensor_slices function
    '''
    x = tf.random.normal(shape=[1], dtype=tf.float32)
    mocker.patch('pseudo_marginal.data.from_pickle', return_value=x)
    result = load_tensor_slices(tf.constant('data.pkl'))
    assert np.array_equal(result, x.numpy())

def test_load_tensor():
    '''
    unit test for load_tensor function
    '''
    x = tf.constant([[0, 1], [2, 3]])
    result1, result2 = load_tensor(x)
    assert tf.reduce_all(tf.equal(result1, tf.constant([0, 2])))
    assert tf.reduce_all(tf.equal(result2, tf.constant([1, 3])))

def test_get_dataset_loader():
    '''
    unit test for get_dataset_loader function
    '''
    args = mock_args()
    args.save_dir = THIS_DIR
    args.T = 2
    args.n = 6
    args.p = 8
    args.N = 2
    args.num_samples = 2
    args.len_sample = 0.02
    args.step_size = 0.01
    args.test_fraction = 1.0

    train_set, test_set = get_dataset_loader(args=args, func=mock_func)
    expected_y = tf.ones(shape=[args.input_dim], dtype=tf.float32)
    c1, c2, c3, c4 = tf.split(expected_y, [args.target_dim, args.target_dim, args.aux_dim, args.aux_dim])
    expected_y = tf.concat([c2, -c1, c4, -c3], axis=0)

    for x, y in train_set:
        assert (x.shape[0] == args.batch_size) and (x.shape[1] == args.input_dim)
        assert (y.shape[0] == args.batch_size) and (y.shape[1] == args.input_dim)
        assert tf.reduce_all(tf.equal(expected_y, y))
        break

    for x, y in test_set:
        assert (x.shape[0] == args.batch_size_test) and (x.shape[1] == args.input_dim)
        assert (y.shape[0] == args.batch_size_test) and (y.shape[1] == args.input_dim)
        assert tf.reduce_all(tf.equal(expected_y, y))
        break

    data_folder = '{}/data/{}_T{}_n{}_p{}_N{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, 
                                                                      args.T, args.n, args.p, args.N,
                                                                      args.num_samples, args.len_sample, args.step_size)
    shutil.rmtree(data_folder)


    