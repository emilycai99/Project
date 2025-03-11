import pytest
from pytest_mock import mocker
import tensorflow as tf
import numpy as np
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.functions_tf import dist_func

# define a series of Hamiltonian
def functions(coords, args):
    #******** 1D Gaussian Mixture #********
    if (args.dist_name == '1D_Gauss_mix'):
        q, p = np.split(coords,2)
        mu1 = 1.0
        mu2 = -1.0
        sigma = 0.35
        term1 = -np.log(0.5*(np.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(np.exp(-(q-mu2)**2/(2*sigma**2))))
        H = term1 + p**2/2 # Normal PDF

    # #******** 2D Gaussian Four Mixtures #********
    elif(args.dist_name == '2D_Gauss_mix'):
        q1, q2, p1, p2 = np.split(coords,4)
        sigma_inv = np.array([[1.,0.],[0.,1.]])
        term1 = 0.
        
        mu = np.array([3.,0.])
        y = np.array([q1-mu[0],q2-mu[1]])
        tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
        term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = np.array([-3.,0.])
        y = np.array([q1-mu[0],q2-mu[1]])
        tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
        term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = np.array([0.,3.])
        y = np.array([q1-mu[0],q2-mu[1]])
        tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
        term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        mu = np.array([0.,-3.])
        y = np.array([q1-mu[0],q2-mu[1]])
        tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
        term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
        
        term1 = -np.log(term1)
        term2 = p1**2/2+p2**2/2
        H = term1 + term2

    # ******** 5D Ill-Conditioned Gaussian #********
    elif(args.dist_name == '5D_illconditioned_Gaussian'):
        dic1 = np.split(coords,args.input_dim)
        var1 = np.array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
        term1 = dic1[0]**2/(2*var1[0])
        for ii in np.arange(1,5,1):
            term1 = term1 + dic1[ii]**2/(2*var1[ii])
        term2 = dic1[5]**2/2
        for ii in np.arange(6,10,1):
            term2 = term2 + dic1[ii]**2/2
        H = term1 + term2

    # ******** nD Funnel #********
    elif(args.dist_name == 'nD_Funnel'):
        dic1 = np.split(coords,args.input_dim)
        term1 = dic1[0]**2/(2*3**2)
        for ii in np.arange(1,int(args.input_dim/2),1):
            term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
        term2 = 0.0
        for ii in np.arange(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + dic1[ii]**2/2 
        H = term1 + term2
    
    # ******** nD Rosenbrock #********
    elif(args.dist_name == 'nD_Rosenbrock'):
        dic1 = np.split(coords,args.input_dim)
        term1 = 0.0
        for ii in np.arange(0,int(args.input_dim/2)-1,1):
            term1 = term1 + (100.0 * (dic1[ii+1] - dic1[ii]**2)**2 + (1 - dic1[ii])**2) / 20.0
        term2 = 0.0
        for ii in np.arange(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + 1*dic1[ii]**2/2
        H = term1 + term2

    # ******** nD standard Gaussian #********
    elif(args.dist_name == 'nD_standard_Gaussian'):
        dic1 = np.split(coords,args.input_dim)
        var1 = np.ones(int(args.input_dim))
        term1 = dic1[0]**2/(2*var1[0])
        for ii in np.arange(1,int(args.input_dim/2),1):
            term1 = term1 + dic1[ii]**2/(2*var1[ii])
        term2 = 0.0
        for ii in np.arange(int(args.input_dim/2),args.input_dim,1):
            term2 = term2 + dic1[ii]**2/2
        H = term1 + term2
    
    else:
        raise ValueError("probability distribution name not recognized")

    return H

@pytest.mark.parametrize('dist_name', ['1D_Gauss_mix', '2D_Gauss_mix', '5D_illconditioned_Gaussian', 'nD_Funnel', 'nD_Rosenbrock', 'nD_standard_Gaussian'])
def test_functions_tf(setup_args, dist_name):
    x = tf.random.normal(shape=[setup_args(dist_name).input_dim])
    func = dist_func(setup_args(dist_name))
    result = func.get_Hamiltonian(x)
    expected = functions(x.numpy(), setup_args(dist_name))
    assert np.allclose(result.numpy(), expected)

    target_log_prob = func.get_target_log_prob_func(x[:int(setup_args(dist_name).input_dim//2)])
    expected = - functions(x.numpy(), setup_args(dist_name)) + 0.5 * tf.tensordot(x[int(setup_args(dist_name).input_dim//2):], \
                                                                                x[int(setup_args(dist_name).input_dim//2):], axes=1).numpy()
    assert np.allclose(target_log_prob.numpy(), expected)


