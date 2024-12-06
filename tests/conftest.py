import pytest

class args:
    def __init__(self):
        args.input_dim = None
        args.dist_name = None

@pytest.fixture
def setup_args(request):
    def _setup(dist_name):
        args()
        args.dist_name = dist_name
        args.batch_size = 2
        args.hidden_dim = 10
        args.num_hmc_samples = 3
        args.num_burnin_samples = 0
        args.total_steps = 0
        args.hnn_threshold = 10.0
        args.lf_threshold = 1000.0
        if dist_name == '1D_Gauss_mix':
            args.input_dim = 2
            return args
        elif dist_name == '2D_Gauss_mix':
            args.input_dim = 4
            return args
        elif dist_name == '5D_illconditioned_Gaussian':
            args.input_dim = 10
            return args
        elif dist_name == 'nD_Funnel':
            args.input_dim = 4
            return args
        elif dist_name == 'nD_Rosenbrock':
            args.input_dim = 6
            return args
        elif dist_name == 'nD_standard_Gaussian':
            args.input_dim = 4
            return args
    return _setup