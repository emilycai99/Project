# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Coded by Som Dhulipala at Idaho National Laboratory
# Parts of this code were borrowed from https://github.com/mfouesneau/NUTS which has an MIT License
# No-U-Turn Sampling with HNNs

import torch, sys, os
import autograd.numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import tensorflow as tf
import tensorflow_probability as tfp
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from nn_models import MLP
from hnn import HNN
from scipy.stats import norm
from scipy.stats import uniform
from .get_args import get_args
from .utils import leapfrog, log_start, log_stop
from .functions import functions
from .data import dynamics_fn


##### Sampling code below #####
# y0 = np.zeros(args.input_dim)
def get_model(args, baseline):
    output_dim = args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity,
                   num_layers=args.num_layers)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              grad_type=args.grad_type, baseline=baseline)
    path = '{}/paper_ckp/{}_n{}_l{}_t{}.tar'.format(args.save_dir, args.dist_name, args.num_samples, args.len_sample, args.total_steps)
    model.load_state_dict(torch.load(path))
    return model

def integrate_model(model, t_span, y0, n, args, **kwargs):
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,args.input_dim)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx
    return leapfrog(fun, t_span, y0, n, args.input_dim)

# hnn_model = get_model(args, baseline=False)

def compute_slice(h_val):
    uni1 = uniform(loc=0,scale=np.exp(-h_val)).rvs()
    return np.log(uni1)

def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)

# def logsumexp1(a, b):
#     c = log(np.exp(a)+np.exp(b))
#     return c

def build_tree(theta, r, logu, v, j, epsilon, joint0, call_lf, hnn_model, args):
    """The main recursion."""
    if (j == 0):
        # joint0 = hamil(hnn_ivp1[:,1])
        t_span1 = [0,v * epsilon]
        kwargs1 = {'t_eval': np.linspace(t_span1[0], t_span1[1], 1), 'rtol': 1e-10}
        y1 = np.concatenate((theta, r), axis=0)
        # one leapfrog step
        hnn_ivp1 = integrate_model(hnn_model, t_span1, y1, 1, args, **kwargs1)
        # new theta
        thetaprime = hnn_ivp1[0:int(args.input_dim/2), 1].reshape(int(args.input_dim/2))
        # new rprime
        rprime = hnn_ivp1[int(args.input_dim/2):int(args.input_dim), 1].reshape(int(args.input_dim/2))
        # get hamiltonian
        joint = functions(hnn_ivp1[:,1])
        # nprime = int(logu <= np.exp(-joint)) # int(logu <= (-joint)) #  int(logu < joint) # 
        
        # call_lf as a flag whether to call the leapfrog method with numerical gradient
        call_lf = call_lf or int((np.log(logu) + joint) > 10.) # int(logu <= np.exp(10. - joint)) # int((logu - 10.) < joint) # int((logu - 10.) < joint) #  int(tmp11 <= np.minimum(1,np.exp(joint0 - joint))) and int((logu - 1000.) < joint) 
        # what does monitor do?
        monitor = np.log(logu) + joint # sprime
        # sprime is to see whether the integration error is too large; note that different threshold is used for hnn and numerical gradient
        sprime = int((np.log(logu) + joint) <= args.hnn_threshold) # 
        
        if call_lf:
            t_span1 = [0,v * epsilon]
            y1 = np.concatenate((theta, r), axis=0)
            # if call_lf, then directly using the numerical gradient
            hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y1, 1, int(args.input_dim))
            thetaprime = hnn_ivp1[0:int(args.input_dim/2), 1].reshape(int(args.input_dim/2))
            rprime = hnn_ivp1[int(args.input_dim/2):int(args.input_dim), 1].reshape(int(args.input_dim/2))
            joint = functions(hnn_ivp1[:,1])
            sprime = int((np.log(logu) + joint) <= args.lf_threshold)
        
        # nprime represents the size of the subtree
        nprime = int(logu <= np.exp(-joint))
        # since there is only one node, thetaminus = thetaplus and same for r
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        alphaprime = min(1., np.exp(joint0 - joint))
        # what is nalphaprime for?
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = build_tree(theta, r, logu, v, j - 1, epsilon, joint0, call_lf, hnn_model, args)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, _, _, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(thetaminus, rminus, logu, v, j - 1, epsilon, joint0, call_lf, hnn_model, args)
            else:
                _, _, thetaplus, rplus, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(thetaplus, rplus, logu, v, j - 1, epsilon, joint0, call_lf, hnn_model, args)
            # Choose which subtree to propagate a sample up from. (see Algorithm 3 in Hoffman)
            if (np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                rprime = rprime2[:]
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics. (what is this for?)
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf

def sample(args):
    ##### result_path #####
    result_path = '{}/results/{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, 
                                                            args.num_samples, args.len_sample, args.step_size)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    log_start(result_path+'/log.txt')
    ##### User-defined sampling parameters #####

    N = args.num_hmc_samples # number of samples
    burn = args.num_burnin_samples # number of burn-in samples
    epsilon = args.epsilon # step size
    N_lf = args.num_cool_down # number of cool-down samples when HNN integration errors are high (see https://arxiv.org/abs/2208.06120)
    hnn_threshold = args.hnn_threshold # HNN integration error threshold (see https://arxiv.org/abs/2208.06120)
    lf_threshold = args.lf_threshold # Numerical gradient integration error threshold

    hnn_model = get_model(args, baseline=False)
    D = int(args.input_dim/2)
    M = N
    Madapt = 0
    theta0 = np.ones(D)
    D = len(theta0)
    samples = np.empty((M + Madapt, D), dtype=float)
    samples[0, :] = theta0
    y0 = np.zeros(args.input_dim)
    # why the position variable also starts from the normal distribution?
    for ii in np.arange(0,int(args.input_dim/2),1):
        y0[ii] = norm(loc=0,scale=1).rvs()
    for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
        y0[ii] = norm(loc=0,scale=1).rvs()
    HNN_accept = np.ones(M)
    traj_len = np.zeros(M)
    alpha_req = np.zeros(M)
    H_store = np.zeros(M)
    monitor_err = np.zeros(M)
    call_lf = 0
    counter_lf = 0
    is_lf = np.zeros(M)

    r_sto = np.zeros(int(args.input_dim/2))

    for m in np.arange(1, M + Madapt, 1):
        if args.verbose and m % args.print_every == 0:
            print(m)
        # resample the momentum variable
        for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
            y0[ii] = norm(loc=0,scale=1).rvs() #  3.0 # -0.87658921 #
        joint = functions(y0) # logp - 0.5 * np.dot(r0, r0.T)

        # sample u, this u should be named after u instead of logu
        logu = np.random.uniform(0, np.exp(-joint))
        # why this step?
        samples[m, :] = samples[m - 1, :]

        # initialize the tree
        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = y0[int(args.input_dim/2):int(args.input_dim)]
        rplus = y0[int(args.input_dim/2):int(args.input_dim)]
        
        j = 0  # initial heigth j = 0
        n = 1  # Initially the only valid point is the initial point.
        s = 1  # Main loop: will keep going until s == 0.
        # call_lf = 0

        # to count how many steps are using leapfrog with numerical gradient
        if call_lf:
            counter_lf +=1
        if counter_lf == N_lf:
            call_lf = 0
            counter_lf = 0

        while (s == 1):
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)

            # Double the size of the tree.
            if (v == -1):
                thetaminus, rminus, _, _, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(thetaminus, rminus, logu, v, j, epsilon, joint, call_lf, hnn_model, args)
            else:
                _, _, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(thetaplus, rplus, logu, v, j, epsilon, joint, call_lf, hnn_model, args)

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (np.random.uniform() < _tmp):
                samples[m, :] = thetaprime[:]
                r_sto = rprime
            # Update number of valid points we've seen.
            n += nprime
            # Decide if it's time to stop.
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # Increment depth.
            j += 1
            monitor_err[m] = monitor
        
        # is_lf: record whether each sample is produced with numerical gradient or hnn
        is_lf[m] = call_lf
        # traj_len: record the trajectory length for each sample
        traj_len[m] = j
        # alpha_req: record the sum of probability (what is this for?)
        alpha_req[m] = alpha
        y0[0:int(args.input_dim/2)] = samples[m, :]
        # H_store: record the Hamitonian value
        H_store[m] = functions(np.concatenate((samples[m, :], r_sto), axis=0))
        # alpha1 = 1.
        # if alpha1 > uniform().rvs():
            
        # else:
        #     samples[m, :] = samples[m-1, :]
        #     HNN_accept[m] = 0
        #     H_store[m] = joint

    np.save(result_path+'/samples.npz', samples)
    np.save(result_path+'/HNN_accept.npz', HNN_accept)
    np.save(result_path+'/traj_len.npz', traj_len)
    np.save(result_path+'/alpha_req.npz', alpha_req)
    np.save(result_path+'/H_store.npz', H_store)
    np.save(result_path+'/monitor_err.npz', monitor_err)
    np.save(result_path+'/is_lf.npz', is_lf)
        
    hnn_tf = tf.convert_to_tensor(samples[burn:M,:])
    ess_hnn = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
    print(ess_hnn)

    log_stop()

    return samples, traj_len, alpha_req, H_store, monitor_err, is_lf

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    sample(args)