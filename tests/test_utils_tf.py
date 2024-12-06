import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import pickle

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from tf_version.utils_tf import leapfrog_tf, L2_loss, to_pickle, from_pickle, Transcript, log_start, log_stop
from paper_code.utils import leapfrog

input_dim = 4
@pytest.fixture
def mock_dydt():
    'mock dynamics fn'
    def _mock_dydt(t, x):
        return tf.range(input_dim, dtype=tf.float32)
    return _mock_dydt

@pytest.fixture
def mock_dydt_np():
    'mock dynamics fn in numpy'
    def _mock_dydt_np(t, x):
        return np.arange(input_dim)
    return _mock_dydt_np

def test_leapfrog_tf(mock_dydt, mock_dydt_np):
    'unit test for leapfrog_tf by comparing with leapfrog in numpy version'
    t_span = [0, 2]
    n = 2
    y0 = tf.random.normal(shape=[input_dim])
    out = leapfrog_tf(mock_dydt, t_span, y0, n, input_dim)
    out_np = leapfrog(mock_dydt_np, t_span, y0.numpy(), n, input_dim)
    print(out)
    print(out_np)
    assert np.allclose(out.numpy(), out_np)

def test_L2_loss():
    'unit test for L2_loss'
    u = tf.random.normal(shape=(2, input_dim))
    v = tf.random.normal(shape=(2, input_dim))
    result = L2_loss(u, v)
    expected = np.mean(np.square(u.numpy() - v.numpy()))
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