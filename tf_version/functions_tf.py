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

class dist_func(tf.Module):
    def __init__(self, args, name=None):
        super().__init__(name)
        self.args = args
        self.dist_dic = {
            '1D_Gauss_mix': self._1D_Gauss_mix,
            '2D_Gauss_mix': self._2D_Gauss_mix,
            '5D_illconditioned_Gaussian': self._5D_illconditioned_Gaussian,
            'nD_Funnel': self._nD_Funnel,
            'nD_Rosenbrock': self._nD_Rosenbrock,
            'nD_standard_Gaussian': self._nD_standard_Gaussian,
            'Allen_Cahn_PDE': self._Allen_Cahn_PDE,
            'Elliptic_PDE': self._Elliptic_PDE
        }
        if not args.dist_name in self.dist_dic:
            raise ValueError("probability distribution name not recognized")
        elif args.dist_name == 'Elliptic_PDE':
            path = '{}/data/{}_d{}_ns{}_ls{}_ss{}'.format(args.save_dir, args.dist_name, args.input_dim,
                                                        args.num_samples, args.len_sample, args.step_size)
            if not os.path.exists(path):
                raise ValueError("the data for Elliptic PDE problem do not exist at {}".format(path))
            self.obs = from_pickle(path + '_obs.pkl')
            self.pos = from_pickle(path + '_pos.pkl')
    
    #******** 1D Gaussian Mixture #********
    def _1D_Gauss_mix(self, states):
        q = states
        mu1 = 1.0
        mu2 = -1.0
        sigma = 0.35
        log_prob = tf.math.log(0.5*(tf.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(tf.exp(-(q-mu2)**2/(2*sigma**2))))
        return log_prob

    #******** 2D Gaussian Four Mixtures #********
    def _2D_Gauss_mix(self, states):
        q1, q2 = tf.split(states, 2)
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
        
        log_prob = tf.math.log(term1)
        return log_prob
    
    #******** 5D Ill-Conditioned Gaussian #********
    def _5D_illconditioned_Gaussian(self, states):
        dic1 = tf.split(states, int(self.args.input_dim//2))
        var1 = tf.constant([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
        term1 = dic1[0]**2/(2*var1[0])
        for ii in range(1,5,1):
            term1 = term1 + dic1[ii]**2/(2*var1[ii])
        log_prob = -term1
        return log_prob

    #******** nD Funnel #********
    def _nD_Funnel(self, states):
        dic1 = tf.split(states, int(self.args.input_dim//2))
        term1 = dic1[0]**2/(2*3**2)
        for ii in range(1,int(self.args.input_dim/2),1):
            term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
        log_prob = -term1
        return log_prob
    
    #******** nD Rosenbrock #********
    def _nD_Rosenbrock(self, states):
        dic1 = tf.split(states, int(self.args.input_dim//2))
        term1 = 0.0
        for ii in range(0,int(self.args.input_dim/2)-1,1):
            term1 = term1 + (100.0 * (dic1[ii+1] - dic1[ii]**2)**2 + (1 - dic1[ii])**2) / 20.0
        log_prob = -term1
        return log_prob
    
    #******** nD standard Gaussian #********
    def _nD_standard_Gaussian(self, states):
        dic1 = tf.split(states, int(self.args.input_dim//2))
        var1 = tf.ones(int(self.args.input_dim))
        term1 = dic1[0]**2/(2*var1[0])
        for ii in range(1, int(self.args.input_dim/2),1):
            term1 = term1 + dic1[ii]**2/(2*var1[ii])
        log_prob = -term1
        return log_prob
    
    #******** Allen Cahn PDE #********
    def _Allen_Cahn_PDE(self, states):
        # ref: Eq.22 in ENSEMBLE SAMPLERS WITH AFFINE INVARIANCE
        dic1 = tf.split(states, int(self.args.input_dim//2))
        dim = int(self.args.input_dim / 2) # should be 25
        delta_x = 1.0 / float(dim - 1) # 25 discrete points lead to (25 - 1) intervals
        term1 = 0.0
        for ii in range(0, dim-1, 1):
            term1 = term1 + 0.5 / delta_x * (dic1[ii+1] - dic1[ii])**2 \
                + 0.5 * delta_x * ((1 - dic1[ii+1]**2)**2 + (1 - dic1[ii]**2)**2)
        log_prob = -term1
        return log_prob
    
    #******** Elliptic PDE #********
    def _Elliptic_PDE(self, states):
        dic1 = tf.split(states, int(self.args.input_dim//2))
        # pos should be a list of sensor positions
        # obs should be a list of observed values
        term1 = 0.0
        # likelihood part
        for ii in range(self.pos.shape[0]):
            x = self.pos[ii][0]
            y = self.pos[ii][1]
            term1 = term1 + 0.5 * (self.obs[ii] - (2 * dic1[0] * tf.math.cos(2 * x) - 4 * (dic1[0] * x + dic1[1] * y) * tf.math.sin(2 * x)
                                            + 2 * dic1[1] * tf.math.cos(2 * y) - 4 * (dic1[0] * x + dic1[1] * y) * tf.math.sin(2 * y))) ** 2
        # prior distribution: two indepedent gaussian with mean 1 variance 1
        term1 = term1 + 0.5 * ((dic1[0] - 1)**2 + (dic1[1] - 1)**2)
        log_prob = -term1
        return log_prob

    def get_target_log_prob_func(self, states):
        return tf.squeeze(self.dist_dic[self.args.dist_name](states))
    
    def get_Hamiltonian(self, coords):
        dim = int(self.args.input_dim/2)
        states = coords[:dim]
        dic2 = tf.split(coords[dim:], dim)
        log_prob = self.get_target_log_prob_func(states)
        term = 0.0
        for i in range(dim):
            term += dic2[i]**2/2
        H = -log_prob + tf.squeeze(term)
        return H
