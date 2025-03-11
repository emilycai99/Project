import os
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.nuts_hnn_sample import sample

class mock_args:
    def __init__(self):
        self.save_dir = '/tmp'
        self.dist_name = 'nD_Rosenbrock'
        self.input_dim = 6
        self.num_samples = 40
        self.len_sample = 100
        self.step_size = 0.025
        self.epsilon = 0.025
        self.num_hmc_samples = 100
        self.num_burnin_samples = 50
        self.print_every = 1
        self.grad_type = 'solenoidal'
        self.lf_threshold = 1000
        self.hidden_dim = 100
        self.num_layers = 3
        self.nonlinearity = 'sine'
        self.total_steps = 1000

# @pytest.fixture
# def mock_os_exists(mocker):
#     yield mocker.patch('tf_version.nuts_hnn_sample.os.path.exists')

@pytest.fixture
def mock_os_makedirs(mocker):
    yield mocker.patch('tf_version.nuts_hnn_sample.os.makedirs')

@pytest.fixture
def mock_np(mocker):
    yield mocker.patch('tf_version.nuts_hnn_sample.np.save')

@pytest.fixture
def mock_logging_start(mocker):
    yield mocker.patch('tf_version.nuts_hnn_sample.log_start')

@pytest.fixture
def mock_logging_stop(mocker):
    yield mocker.patch('tf_version.nuts_hnn_sample.log_stop')

def test_sample(mocker, mock_os_makedirs, mock_np, mock_logging_start, mock_logging_stop):
    args = mock_args()

    mocker.patch('tf_version.nuts_hnn_sample.tfp.mcmc.sample_chain', 
                 return_value=(tf.random.normal(shape=[1, args.input_dim], dtype=tf.float32), None))

    # Call the sample function
    result = sample(args)

    # Assert that the result path was created
    result_path = '{}/results/{}_d{}_ns{}_ls{}_ss{}_nhmc{}_{}'.format(args.save_dir, args.dist_name, args.input_dim,
                                                            args.num_samples, args.len_sample, args.step_size, args.num_hmc_samples, 
                                                            args.grad_type)


    # Assert that log_start and log_stop were called
    mock_logging_start.assert_called_once_with(f'{result_path}/log.txt')
    mock_logging_stop.assert_called_once()

    # Assert that the result is as expected
    assert isinstance(result, tf.Tensor)
    assert result.shape == (1, args.input_dim)  # Assuming the mock state has this shape

    # Assert that np.save was called
    assert np.array_equal(mock_np.call_args[0][1], result.numpy())
    assert mock_np.call_args[0][0] == f'{result_path}/samples.npz'

if __name__ == "__main__":
    pytest.main()