import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import math
import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from utils import from_pickle
from dist_generation import generate_GLMM

class Hamiltonian_func(tf.Module):
    '''
    Description:
        This version is a standard version; no techniques involved in calculation.
        (can be used for testing the later implementations)
    '''
    def __init__(self, args, name=None):
        super().__init__(name)
        self.args = args
        if args.dist_name == 'gauss_mix':
            if not os.path.exists(os.path.join(args.data_pth, 'Z_T{}_n{}_p{}.pkl'.format(args.T, args.n, args.p))):
                print('Generate data for gaussian mixture')
                generate_GLMM(T=args.T, n=args.n, p=args.p, data_pth=args.data_pth)
            else:
                self.Z = from_pickle(os.path.join(args.data_pth, 'Z_T{}_n{}_p{}.pkl'.format(args.T, args.n, args.p)))
                self.Y = from_pickle(os.path.join(args.data_pth, 'Y_T{}_n{}_p{}.pkl'.format(args.T, args.n, args.p)))
        else:
            raise NotImplementedError
    
    def get_func(self):
        if self.args.dist_name == 'gauss_mix':
            return self.gauss_mix_func
        else:
            raise NotImplementedError

    def gauss_mix_func(self, coords):
        # d: length of theta
        d = self.args.p + 5
        # coords = [beta, mu1, mu2, log(lambda1), log(lambda2), logit(w), rho, u, p]
        assert coords.shape[0] == 2*d + 2*(self.args.T * self.args.N), 'incorrect shape of coords'

        # get all the parameters
        theta = coords[:d]
        beta = coords[:self.args.p]
        mu1 = coords[self.args.p]
        mu2 = coords[self.args.p+1]
        lambda1 = tf.math.exp(coords[self.args.p+2])
        lambda2 = tf.math.exp(coords[self.args.p+3])
        w = tf.math.sigmoid(coords[self.args.p+4])
        rho = coords[d:2*d]
        u, p = tf.split(coords[-2*(self.args.T * self.args.N):], 2)

        assert rho.shape[0] == d, 'incorrect shape of rho'
        assert u.shape[0] == self.args.T * self.args.N, 'incorrect shape of u'
        assert p.shape[0] == self.args.T * self.args.N, 'incorrect shape of p'

        # X contains Xil for i = 1 to T and l = 1 to N
        X = tf.reshape(3.0 * u, shape=[self.args.T, self.args.N])

        # define distributions
        ## define f(Xil): gaussian mixture
        x_dist = tfd.Mixture(
            cat=tfd.Categorical(probs=[w, 1.0-w]),
            components=[
                tfd.Normal(loc=mu1, scale=lambda1**(-0.5)),
                tfd.Normal(loc=mu2, scale=lambda2**(-0.5))
            ]
        )
        ## define q(Xil)
        q_dist = tfd.Normal(loc=0.0, scale=3.0)

        ## define g(Yij|Xil): bernoulli
        ### X_expand.shape = [T, N, n]; X.shape = [T, N]
        X_expand = tf.repeat(tf.expand_dims(X, -1), repeats=self.args.n, axis=-1)
        ### Z_expand.shape = [T, N, n, p]; Z.shape = [T, n, p]
        Z_expand = tf.repeat(tf.expand_dims(self.Z, 1), repeats=self.args.N, axis=1)
        g_dist = tfd.Bernoulli(logits=X_expand+tf.tensordot(Z_expand, beta, axes=[[3], [0]]))

        ## N(0,100) prior for each component of theta
        theta_prior = tfd.MultivariateNormalDiag(loc=[0.0 for _ in range(d)], scale_diag=[10.0 for _ in range(d)])
        
        # calculate f(Xil): gaussian mixture, shape = [T, N] -> [T, N, n]
        f_val = x_dist.prob(X_expand)
        # calculate q(Xil): shape = [T, N] -> [T, N, n]
        q_val = q_dist.prob(X_expand)
        # calculate g(Yij|Xil): shape = [T, N, n] (Yilj)
        Y_expand = tf.repeat(tf.expand_dims(self.Y, axis=1), repeats=self.args.N, axis=1)
        g_val = g_dist.prob(Y_expand)

        w_ilj = tf.math.divide(tf.math.multiply(g_val, f_val), q_val)
        w_ij = tf.math.log(tf.reduce_sum(w_ilj, axis=1) / self.args.N)
        # calculate the log(phat(y|theta, u))
        log_phat = tf.reduce_sum(w_ij)

        # calculate logp(theta)
        log_prior = theta_prior.log_prob(theta)

        # calculate the hamiltonian value
        H = -log_prior - log_phat + 0.5*tf.tensordot(rho, rho, axes=1) + 0.5*tf.tensordot(u, u, axes=1) +\
            0.5*tf.tensordot(p, p, axes=1)

        return H

class Hamiltonian_func_debug(tf.Module):
    '''
    Description:
        This version is an improved version:
            when calculating w, the log_sum_exp trick is used
    '''
    def __init__(self, args, name=None):
        super().__init__(name)
        self.args = args
        if args.dist_name == 'gauss_mix':
            if not os.path.exists(os.path.join(args.data_pth, 'Z_T{}_n{}_p{}.pkl'.format(args.T, args.n, args.p))):
                print('Generate data for gaussian mixture')
                generate_GLMM(T=args.T, n=args.n, p=args.p, data_pth=args.data_pth)
            else:
                self.Z = from_pickle(os.path.join(args.data_pth, 'Z_T{}_n{}_p{}.pkl'.format(args.T, args.n, args.p)))
                self.Y = from_pickle(os.path.join(args.data_pth, 'Y_T{}_n{}_p{}.pkl'.format(args.T, args.n, args.p)))
        else:
            raise NotImplementedError
    
    def get_func(self):
        if self.args.dist_name == 'gauss_mix':
            return self.gauss_mix_func
        else:
            raise NotImplementedError

    def gauss_mix_func(self, coords):
        # d: length of theta
        d = self.args.p + 5
        # coords = [beta, mu1, mu2, log(lambda1), log(lambda2), logit(w), rho, u, p]
        assert coords.shape[0] == 2*d + 2*(self.args.T * self.args.N), 'incorrect shape of coords'

        # get all the parameters
        theta = coords[:d]
        beta = coords[:self.args.p]
        mu1 = coords[self.args.p]
        mu2 = coords[self.args.p+1]
        lambda1 = tf.math.exp(coords[self.args.p+2])
        lambda2 = tf.math.exp(coords[self.args.p+3])
        w = tf.math.sigmoid(coords[self.args.p+4])
        rho = coords[d:2*d]
        u, p = tf.split(coords[-2*(self.args.T * self.args.N):], 2)

        assert rho.shape[0] == d, 'incorrect shape of rho'
        assert u.shape[0] == self.args.T * self.args.N, 'incorrect shape of u'
        assert p.shape[0] == self.args.T * self.args.N, 'incorrect shape of p'

        # X contains Xil for i = 1 to T and l = 1 to N
        X = tf.reshape(3.0 * u, shape=[self.args.T, self.args.N])

        # define distributions
        ## define f(Xil): gaussian mixture
        x_dist = tfd.Mixture(
            cat=tfd.Categorical(probs=[w, 1.0-w]),
            components=[
                tfd.Normal(loc=mu1, scale=lambda1**(-0.5)),
                tfd.Normal(loc=mu2, scale=lambda2**(-0.5))
            ]
        )
        ## define q(Xil)
        q_dist = tfd.Normal(loc=0.0, scale=3.0)

        ## define g(Yij|Xil): bernoulli
        ### X_expand.shape = [T, N, n]; X.shape = [T, N]
        X_expand = tf.repeat(tf.expand_dims(X, -1), repeats=self.args.n, axis=-1)
        ### Z_expand.shape = [T, N, n, p]; Z.shape = [T, n, p]
        Z_expand = tf.repeat(tf.expand_dims(self.Z, 1), repeats=self.args.N, axis=1)
        g_dist = tfd.Bernoulli(logits=X_expand+tf.tensordot(Z_expand, beta, axes=[[3], [0]]))

        ## N(0,100) prior for each component of theta
        theta_prior = tfd.MultivariateNormalDiag(loc=[0.0 for _ in range(d)], scale_diag=[10.0 for _ in range(d)])

        # calculate f(Xil): gaussian mixture, shape = [T, N] -> [T, N, n]
        log_f_val = x_dist.log_prob(X_expand)
        # calculate q(Xil): shape = [T, N] -> [T, N, n]
        log_q_val = q_dist.log_prob(X_expand)
        # calculate g(Yij|Xil): shape = [T, N, n] (Yilj)
        Y_expand = tf.repeat(tf.expand_dims(self.Y, axis=1), repeats=self.args.N, axis=1)
        log_g_val = g_dist.log_prob(Y_expand)

        # shape = [T, N, n]
        log_w_ilj = log_g_val + log_f_val - log_q_val
        max_log_w_ilj = tf.math.reduce_max(log_w_ilj, axis=1, keepdims=True)
        w_ilj = tf.math.exp(tf.math.subtract(log_w_ilj, max_log_w_ilj))
        w_ij = tf.math.log(tf.reduce_sum(w_ilj, axis=1)) + tf.squeeze(max_log_w_ilj, axis=1) - tf.math.log(tf.constant([self.args.N], dtype=tf.float32))
        # calculate the log(phat(y|theta, u))
        log_phat = tf.reduce_sum(w_ij)

        # calculate logp(theta)
        log_prior = theta_prior.log_prob(theta)

        # calculate the hamiltonian value
        H = -log_prior - log_phat + 0.5*tf.tensordot(rho, rho, axes=1) + 0.5*tf.tensordot(u, u, axes=1) +\
            0.5*tf.tensordot(p, p, axes=1)

        return H
        

