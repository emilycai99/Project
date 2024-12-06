import pytest
import pytest_mock
import tensorflow as tf

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.data_tf import dynamics_fn_tf, get_trajectory_tf, get_dataset_tf
from tf_version.get_args import get_args

@pytest.fixture
def tmp_functions_tf():
    '''
    mock functions_tf
    x**2 + y**2
    return (x,y)
    '''
    def _tmp_functions_tf(coords):
        return 0.5 * tf.math.pow(coords[0], 2) + 0.5 * tf.math.pow(coords[1], 2)
    return _tmp_functions_tf

@pytest.fixture
def tmp_leapfrog_tf():
    '''
    mock leapfrog_tf
    return [y0, y0, ...]
    '''
    def _tmp_leapfrog_tf(dydt, tspan, y0, n, dim):
        return tf.repeat(tf.expand_dims(y0, axis=-1), repeats=n, axis=-1)
    return _tmp_leapfrog_tf

@pytest.fixture
def tmp_dynamics_fn_tf():
    '''
    mock dynamics_fn_tf
    return coords
    '''
    def _tmp_dynamics_fn_tf(t, coords):
        return coords
    return _tmp_dynamics_fn_tf

@pytest.fixture
def tmp_get_trajectory_tf():
    '''
    mock get_trajectory_tf
    '''
    def _tmp_get_trajectory_tf(y0=None, **kwargs):
        t_span = [0, 1]
        timescale = 0.5
        dic1 = [tf.ones(shape=[1, int((t_span[1] - t_span[0]) / timescale)]) for _ in range(int(y0.shape[0]))]
        ddic1 = [tf.zeros(shape=[1, int((t_span[1] - t_span[0]) / timescale)]) for _ in range(int(y0.shape[0]))]
        return dic1, ddic1
    return _tmp_get_trajectory_tf

def test_dynamics_fn_tf(mocker, tmp_functions_tf, setup_args):
    '''
    unit test for dynamics_fn_tf
    '''
    x = tf.random.normal(shape=[2])
    mocker.patch('tf_version.data_tf.functions_tf', side_effect=tmp_functions_tf)
    mocker.patch('tf_version.data_tf.args', setup_args('1D_Gauss_mix'))
    results = dynamics_fn_tf(None, x)
    assert bool(tf.reduce_all(tf.math.equal(results, tf.stack([x[-1], -x[0]], axis=0))))

def test_get_trajectory_tf(mocker, tmp_leapfrog_tf, tmp_dynamics_fn_tf, setup_args):
    '''
    unit test for get_trajectory_tf
    '''
    x = tf.random.normal(shape=[2])
    mocker.patch('tf_version.data_tf.leapfrog_tf', side_effect=tmp_leapfrog_tf)
    mocker.patch('tf_version.data_tf.dynamics_fn_tf', side_effect=tmp_dynamics_fn_tf)
    mocker.patch('tf_version.data_tf.args', setup_args('1D_Gauss_mix'))
    dic1, ddic1 = get_trajectory_tf([0, 1], timescale=0.5, y0=x)
    expected1 = [tf.repeat(x[i], repeats=2, axis=0) for i in range(2)]
    expected2 = [tf.repeat(x[i], repeats=2, axis=0) for i in range(2)]
    
    assert bool(tf.reduce_all([tf.reduce_all(tf.equal(dic1[i], expected1[i])) for i in range(len(dic1))]))
    assert bool(tf.reduce_all([tf.reduce_all(tf.equal(ddic1[i], expected2[i])) for i in range(len(ddic1))]))

def test_get_dataset_tf(mocker, tmp_get_trajectory_tf):
    args = get_args()
    mocker.patch('tf_version.data_tf.args', args)
    mocker.patch('tf_version.data_tf.os.makedirs')
    mocker.patch('tf_version.data_tf.to_pickle')
    mocker.patch('tf_version.data_tf.get_trajectory_tf', tmp_get_trajectory_tf)
    results = get_dataset_tf(seed=0, samples=10, y_init=tf.zeros(shape=[args.input_dim]))
    assert 'coords' in results
    assert 'test_coords' in results
    assert 'dcoords' in results
    assert 'test_dcoords' in results
    assert results['coords'].shape[0] == 10 * 2
    assert results['dcoords'].shape[1] == args.input_dim
    assert results['test_coords'].shape[0] == int(10 * args.test_fraction) * 2
    assert results['test_coords'].shape[1] == args.input_dim

