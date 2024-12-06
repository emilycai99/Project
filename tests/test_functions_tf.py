import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.functions_tf import functions_tf
from paper_code.functions import functions

@pytest.mark.parametrize('dist_name', ['1D_Gauss_mix', '2D_Gauss_mix', '5D_illconditioned_Gaussian', 'nD_Funnel', 'nD_Rosenbrock', 'nD_standard_Gaussian'])
def test_functions_tf(setup_args, dist_name, mocker):
    mocker.patch("tf_version.functions_tf.args", setup_args(dist_name))
    mocker.patch("paper_code.functions.args", setup_args(dist_name))
    x = tf.random.normal(shape=[setup_args(dist_name).input_dim])
    result = functions_tf(x)
    expected = functions(x.numpy())
    assert np.allclose(result.numpy(), expected)



