import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import pickle

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.utils import *

def func(x):
    return tf.tensordot(x, x, axes=1)

class mock_args:
    def __init__(self):
        self.input_dim = 10
        self.target_dim = 3
        self.aux_dim = 2

def test_numerical_grad():
    '''
    unit test for numerical_grad
        - by checking its output
    '''
    args = mock_args()

    input = tf.random.normal(shape=[args.input_dim], dtype=tf.float32)
    result = numerical_grad(input, func, args.target_dim, args.aux_dim)
    expected = 2.0 * tf.concat([input[args.target_dim:2*args.target_dim],
                                -input[:args.target_dim],
                                input[-args.aux_dim:],
                                -input[-2*args.aux_dim:-args.aux_dim]], axis=0)
    assert tf.reduce_all(tf.equal(expected, result))

def log_prior(theta):
    return tf.tensordot(theta, theta, axes=1)

def log_phat(theta, u):
    return tf.tensordot(theta, theta, axes=1) + 2.0 * tf.tensordot(u, u, axes=1)

args = mock_args()
def mock_H_func(coords):
    theta, rho, u, p = tf.split(coords, [args.target_dim, args.target_dim,
                                         args.aux_dim, args.aux_dim], axis=0)
    H = -log_prior(theta) -log_phat(theta, u) + 0.5 * (tf.tensordot(rho, rho, axes=1)+
                                                        tf.tensordot(u, u, axes=1)+
                                                        tf.tensordot(p, p, axes=1))
    return H

def test_integrator_one_step():
    '''
    integration test with numerical_grad
    '''
    args = mock_args()
    coords = tf.random.normal(shape=[args.input_dim], dtype=tf.float32)
    h = 0.05
    result = integrator_one_step(coords, mock_H_func, numerical_grad, h, args.target_dim, args.aux_dim)

    # expected: handwritten version of Appendix. A
    theta, rho, u, p = tf.split(coords, [args.target_dim, args.target_dim,
                                         args.aux_dim, args.aux_dim], axis=0)
    new_theta = theta + h * rho + 0.5 * (h**2)  * (tfp.math.value_and_gradient(log_prior, theta + 0.5*h*rho)[-1] + 
                                                   tfp.math.value_and_gradient(log_phat, [theta + 0.5*h*rho, p*math.sin(0.5*h) + u*math.cos(0.5*h)])[-1][0])
    new_rho = rho + h * (tfp.math.value_and_gradient(log_prior, theta + 0.5*h*rho)[-1] + 
                         tfp.math.value_and_gradient(log_phat, [theta + 0.5*h*rho, p*math.sin(0.5*h) + u*math.cos(0.5*h)])[-1][0])
    new_u = p * math.sin(h) + u * math.cos(h) + math.sin(0.5*h) * h * tfp.math.value_and_gradient(log_phat, [theta + 0.5*h*rho, p*math.sin(0.5*h) + u*math.cos(0.5*h)])[-1][-1]
    new_p = p * math.cos(h) - u * math.sin(h) + math.cos(0.5*h) * h * tfp.math.value_and_gradient(log_phat, [theta + 0.5*h*rho, p*math.sin(0.5*h) + u*math.cos(0.5*h)])[-1][-1]
    expected = tf.concat([new_theta, new_rho, new_u, new_p], axis=0)
    print(result)
    print(expected)
    assert tf.reduce_all(tf.experimental.numpy.allclose(result, expected))

@pytest.fixture
def mock_integrator_one_step():
    def _mock_integrator_one_step(coords, func, derivs_func, h, target_dim, aux_dim):
        return coords * h
    return _mock_integrator_one_step


def test_integrator(mocker, mock_integrator_one_step):
    '''
    unit test for integrator() with integrator_one_step mocked
    '''
    mocker.patch('pseudo_marginal.utils.integrator_one_step', mock_integrator_one_step)
    h = 2
    steps = 2
    coords = tf.random.normal(shape=[10], dtype=tf.float32)
    result = integrator(coords, None, None, h, steps, None, None)
    expected = tf.stack([coords, coords*h, coords*(h**2)], axis=-1)
    print(result)
    print(expected)
    assert tf.reduce_all(tf.equal(result, expected))

def test_L2_loss():
    '''
    unit test for L2_loss function
    '''
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y = tf.zeros(shape=[2, 2], dtype=tf.float32)
    result = L2_loss(x, y)
    expected = ((1.0 ** 2 + 2.0 ** 2) + (3.0 ** 2 + 4.0 ** 2)) / 2.0
    assert result.numpy() == expected

@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "test.pkl"
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)

def test_to_pickle(temp_file):
    '''
    unit test for to_pickle
    '''
    # Data to be pickled
    data = {'key': 'value'}
    # Call the to_pickle function to save the data to a pickle file
    to_pickle(data, temp_file)
    # Check if the file exists
    assert os.path.isfile(temp_file)
    # Load the pickled data to verify if it was saved correctly
    with open(temp_file, 'rb') as f:
        loaded_data = pickle.load(f)
    # Verify if the loaded data matches the original data
    assert loaded_data == data

@pytest.fixture
def temp_pickle_file(tmp_path):
    data = {'key': 'value'}
    file_path = tmp_path / "test.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)

def test_from_pickle(temp_pickle_file):
    '''
    unit test for from_pickle
    '''
    # Call the from_pickle function to load data from the pickle file
    loaded_data = from_pickle(temp_pickle_file)
    # Define the expected data
    expected_data = {'key': 'value'}
    # Check if the loaded data matches the expected data
    assert loaded_data == expected_data

@pytest.fixture
def temp_log_file(tmp_path):
    '''
    mock log file
    '''
    file_path = tmp_path / "test_log.txt"
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)

def test_transcript_write(temp_log_file, capfd):
    '''
    unit test for transcript_write
    '''
    transcript = Transcript(temp_log_file)

    message = "Testing Transcript class\n"
    transcript.write(message)

    # Check if the message is written to both sys.stdout and the log file
    captured_out, _ = capfd.readouterr()
    assert captured_out == message

    transcript.logfile.seek(0)  # Move the file pointer to the beginning
    content = transcript.logfile.read()
    assert content == message

def test_log_start(temp_log_file):
    '''
    unit test for log_start
    '''
    log_start(temp_log_file)
    # Check if sys.stdout is set to an instance of the Transcript class
    assert isinstance(sys.stdout, Transcript)

def test_log_stop(temp_log_file):
    '''
    unit test for log_stop
    '''
    # Save the original sys.stdout for comparison
    original_stdout = sys.stdout
    # Create a temporary Transcript instance for testing purposes
    sys.stdout = Transcript(temp_log_file)
    # Call the log_stop function
    log_stop()
    # Check if sys.stdout is set back to the original sys.stdout. 
    assert sys.stdout == original_stdout

