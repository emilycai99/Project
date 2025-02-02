# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Training Hamiltonian Neural Networks (HNNs) for Bayesian inference problems
# Original authors of HNNs code: Sam Greydanus, Misko Dzamba, Jason Yosinski (2019)
# Available at https://github.com/greydanus/hamiltonian-nn under the Apache License 2.0
# Modified by Som Dhulipala at Idaho National Laboratory for Bayesian inference problems
# Modifications include:
# - Generalizing the code to any number of dimensions
# - Introduce latent parameters to HNNs to improve expressivity
# - Reliance on the leap frog integrator for improved dynamics stability
# - Obtain the training from probability distribution space
# - Use a deep HNN arichtecture to improve predictive performance

''' Verified'''
import tensorflow as tf
import sys, os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.utils_tf import from_pickle
from tf_version.get_args import get_args
args = get_args()

if args.dist_name == 'Elliptic_PDE':
    path = '{}/data/{}_d{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, args.input_dim,
                                                args.num_samples, args.len_sample, args.step_size)
    obs = from_pickle(path + '_obs.pkl')
    pos = from_pickle(path + '_pos.pkl')

# define a series of Hamiltonian
def functions_tf(coords):
   #******** 1D Gaussian Mixture #********
    if (args.dist_name == '1D_Gauss_mix'):
        q, p = tf.split(coords,2)
        mu1 = 1.0
        mu2 = -1.0
        sigma = 0.35
        term1 = -tf.math.log(0.5*(tf.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(tf.exp(-(q-mu2)**2/(2*sigma**2))))
        H = term1 + p**2/2 # Normal PDF

    # #******** 2D Gaussian Four Mixtures #********
    elif(args.dist_name == '2D_Gauss_mix'):
        q1, q2, p1, p2 = tf.split(coords,4)
        sigma_inv = tf.eye(2,2)
        term1 = 0.
        
        mu = tf.constant([3.,0.])
        y = tf.concat([q1-mu[0],q2-mu[1]], axis=0)
        tmp1 = tf.reshape(tf.convert_to_tensor([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]), 2)
        term1 = term1 + 0.25*tf.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = tf.constant([-3.,0.])
        y = tf.concat([q1-mu[0],q2-mu[1]], axis=0)
        tmp1 = tf.reshape(tf.convert_to_tensor([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]), 2)
        term1 = term1 + 0.25*tf.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = tf.constant([0.,3.])
        y = tf.concat([q1-mu[0],q2-mu[1]], axis=0)
        tmp1 = tf.reshape(tf.convert_to_tensor([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]), 2)
        term1 = term1 + 0.25*tf.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = tf.constant([0.,-3.])
        y = tf.concat([q1-mu[0],q2-mu[1]], axis=0)
        tmp1 = tf.reshape(tf.convert_to_tensor([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]), 2)
        term1 = term1 + 0.25*tf.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        term1 = -tf.math.log(term1)
        term2 = p1**2/2+p2**2/2
        H = term1 + term2

    # ******** 5D Ill-Conditioned Gaussian #********
    elif(args.dist_name == '5D_illconditioned_Gaussian'):
        dic1 = tf.split(coords,args.input_dim)
        var1 = tf.constant([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
        term1 = dic1[0]**2/(2*var1[0])
        for ii in range(1,5,1):
            term1 = term1 + dic1[ii]**2/(2*var1[ii])
        term2 = dic1[5]**2/2
        for ii in range(6,10,1):
            term2 = term2 + dic1[ii]**2/2
        H = term1 + term2

    # ******** nD Funnel #********
    elif(args.dist_name == 'nD_Funnel'):
        dic1 = tf.split(coords,args.input_dim)
        term1 = dic1[0]**2/(2*3**2)
        for ii in range(1,int(args.input_dim/2),1):
            term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
        term2 = 0.0
        for ii in range(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + dic1[ii]**2/2 
        H = term1 + term2
    
    # ******** nD Rosenbrock #********
    elif(args.dist_name == 'nD_Rosenbrock'):
        dic1 = tf.split(coords,args.input_dim)
        term1 = 0.0
        for ii in range(0,int(args.input_dim/2)-1,1):
            term1 = term1 + (100.0 * (dic1[ii+1] - dic1[ii]**2)**2 + (1 - dic1[ii])**2) / 20.0
        term2 = 0.0
        for ii in range(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + 1*dic1[ii]**2/2
        H = term1 + term2

    # ******** nD standard Gaussian #********
    elif(args.dist_name == 'nD_standard_Gaussian'):
        dic1 = tf.split(coords,args.input_dim)
        var1 = tf.ones(int(args.input_dim))
        term1 = dic1[0]**2/(2*var1[0])
        for ii in range(1,int(args.input_dim/2),1):
            term1 = term1 + dic1[ii]**2/(2*var1[ii])
        term2 = 0.0
        for ii in range(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + dic1[ii]**2/2
        H = term1 + term2

    # ******** Allen Cahn PDE #********
    elif(args.dist_name == 'Allen_Cahn_PDE'):
        # ref: Eq.22 in ENSEMBLE SAMPLERS WITH AFFINE INVARIANCE
        dic1 = tf.split(coords, args.input_dim)
        dim = int(args.input_dim / 2) # should be 25
        delta_x = 1.0 / float(dim - 1) # 25 discrete points lead to (25 - 1) intervals
        term1 = 0.0
        for ii in range(0, dim-1, 1):
            term1 = term1 + 0.5 / delta_x * (dic1[ii+1] - dic1[ii])**2 \
                + 0.5 * delta_x * ((1 - dic1[ii+1]**2)**2 + (1 - dic1[ii]**2)**2)
        term2 = 0.0
        for ii in range(dim, args.input_dim, 1):
            term2 = term2 + dic1[ii]**2/2
        H = term1 + term2
    
    # ******** Elliptic PDE #********
    elif(args.dist_name == 'Elliptic_PDE'):
        dic1 = tf.split(coords, args.input_dim)
        # pos should be a list of sensor positions
        # obs should be a list of observed values
        term1 = 0.0
        # likelihood part
        for ii in range(pos.shape[0]):
            x = pos[ii][0]
            y = pos[ii][1]
            term1 = term1 + 0.5 * (obs[ii] - (2 * dic1[0] * tf.math.cos(2 * x) - 4 * (dic1[0] * x + dic1[1] * y) * tf.math.sin(2 * x)
                                              + 2 * dic1[1] * tf.math.cos(2 * y) - 4 * (dic1[0] * x + dic1[1] * y) * tf.math.sin(2 * y))) ** 2
        # prior distribution: two indepedent gaussian with mean 1 variance 1
        term1 = term1 + 0.5 * ((dic1[0] - 1)**2 + (dic1[1] - 1)**2)
        term2 = 0.0
        for ii in range(int(args.input_dim / 2), args.input_dim, 1):
            term2 = term2 + dic1[ii]**2/2
        H = term1 + term2        

    else:
        raise ValueError("probability distribution name not recognized")

    return H