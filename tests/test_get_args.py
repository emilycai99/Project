import pytest
import pytest_mock
from argparse import ArgumentTypeError

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from pseudo_marginal.get_args import *

def test_parse_float_list():
    'unit test for parse_float_list function'
    # Test with a single float value
    result = parse_float_list("3.14")
    assert result == 3.14

    # Test with multiple float values
    result = parse_float_list("1.0,2.0,3.0")
    assert result == [1.0, 2.0, 3.0]

    # Test with invalid input (non-float values)
    with pytest.raises(ArgumentTypeError):
        parse_float_list("1.0,2.0,abc")

def test_get_args(mocker):
    'Test with custom command-line arguments for get_args function'
    test_args = [
        '--dist_name', 'gauss_mix',
        '--p', '10',
        '--T', '1000',
        '--N', '256',
        '--n', '12',
        '--input_dim', '20',
        '--target_dim', '15',
        '--aux_dim', '1000',
        '--num_samples', '20',
        '--len_sample', '100.0',
        '--step_size', '0.05',
        '--test_fraction', '0.2',
        '--save_dir', '/custom/save/dir',
        '--load_dir', '/custom/load/dir',
        '--should_load',
        '--load_file_name', 'custom_file.pkl',
        '--nn_out_dim', '30',
        '--hidden_dim', '200',
        '--num_layers', '4',
        '--learn_rate', '1e-3',
        '--batch_size', '512',
        '--batch_size_test', '2000',
        '--nonlinearity', 'relu',
        '--grad_type', 'gradient',
        '--total_steps', '10000',
        '--print_every', '500',
        '--verbose',
        '--seed', '42',
        '--gpu_id', '1',
        '--shuffle_buffer_size', '200',
        '--penalty_strength', '0.1',
        '--nn_model_name', 'custom_model',
        '--decay_rate', '0.9',
        '--retrain',
        '--retrain_lr', '1e-4',
        '--num_hmc_samples', '28000',
        '--num_burnin_samples', '14000',
        '--epsilon', '0.04',
        '--num_cool_down', '40',
        '--hnn_threshold', '20.0',
        '--lf_threshold', '2000.0',
        '--adapt_iter', '10',
        '--delta', '0.7',
        '--grad_flag',
        '--grad_mass_flag',
        '--rho_var', '1.0, 2.0, 3.0',
        '--num_flag',
        '--fff', '2'
    ]
    
    args = get_args(test_args)
    assert args.dist_name == 'gauss_mix'
    assert args.p == 10
    assert args.T == 1000
    assert args.N == 256
    assert args.n == 12
    assert args.input_dim == 20
    assert args.target_dim == 15
    assert args.aux_dim == 1000
    assert args.num_samples == 20
    assert args.len_sample == 100.0
    assert args.step_size == 0.05
    assert args.test_fraction == 0.2
    assert args.save_dir == '/custom/save/dir'
    assert args.load_dir == '/custom/load/dir'
    assert args.should_load is True
    assert args.load_file_name == 'custom_file.pkl'
    assert args.nn_out_dim == 30
    assert args.hidden_dim == 200
    assert args.num_layers == 4
    assert args.learn_rate == 1e-3
    assert args.batch_size == 512
    assert args.batch_size_test == 2000
    assert args.nonlinearity == 'relu'
    assert args.grad_type == 'gradient'
    assert args.total_steps == 10000
    assert args.print_every == 500
    assert args.verbose is True
    assert args.seed == 42
    assert args.gpu_id == 1
    assert args.shuffle_buffer_size == 200
    assert args.penalty_strength == 0.1
    assert args.nn_model_name == 'custom_model'
    assert args.decay_rate == 0.9
    assert args.retrain is True
    assert args.retrain_lr == 1e-4
    assert args.num_hmc_samples == 28000
    assert args.num_burnin_samples == 14000
    assert args.epsilon == 0.04
    assert args.num_cool_down == 40
    assert args.hnn_threshold == 20.0
    assert args.lf_threshold == 2000.0
    assert args.adapt_iter == 10
    assert args.delta == 0.7
    assert args.grad_flag is True
    assert args.grad_mass_flag is True
    assert args.rho_var == [1.0, 2.0, 3.0]
    assert args.num_flag is True
    assert args.fff == "2"
    assert '{}'.format(args.rho_var) == '[1.0, 2.0, 3.0]'