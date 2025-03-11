import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import math
import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from pseudo_marginal.dist_generation import generate_GLMM
from pseudo_marginal.utils import * 

def stable_log1pexp(x):
    '''
    Description:
        stable version to calculation ln(1+exp(x))
    '''
    out = tf.where(
        x <= -37.0,
        tf.exp(x),
        tf.where(
        x <= 18.0,
        tf.math.log(1 + tf.math.exp(x)),
        tf.where(
        x <= 33.3,
        x + tf.math.exp(-x),
        x
        ))
    )
    return out

def grad_gauss_mix(X, mu1, mu2, log_lambda1, log_lambda2, logit_w):
    '''
    Description:
        function to calculate the log pdf of Gaussian mixture distribution
        and its gradients with respect to parameters
    '''
    # compute log normal
    w = sigmoid(logit_w)
    lambda1 = tf.math.exp(log_lambda1)
    lambda2 = tf.math.exp(log_lambda2)

    scaled_err_square1 = lambda1 * tf.math.squared_difference(X, mu1)
    scaled_err_square2 = lambda2 * tf.math.squared_difference(X, mu2)

    # X = T x N (500 x 128)
    log_pdf1 = 0.5 * log_lambda1 - 0.5 * tf.math.log(2 * math.pi) - 0.5 * scaled_err_square1 + tf.math.log(w)
    log_pdf2 = 0.5 * log_lambda2 - 0.5 * tf.math.log(2 * math.pi) - 0.5 * scaled_err_square2 + tf.math.log(1-w)

    log_pdf = tf.stack([log_pdf1, log_pdf2], axis=1) #500 x 2 x 128
    C = tf.reduce_max(log_pdf, axis=1, keepdims=True) # 500 x 1 x128
    pdf = tf.math.exp(tf.subtract(log_pdf, C)) # 500 x 2 x 128
    log_norm = tf.math.log(tf.reduce_sum(pdf, axis=1)) + tf.squeeze(C, axis=1) # 500 x 128

    share = tf.math.exp(tf.subtract(log_pdf, tf.expand_dims(log_norm, axis=1))) # 500 x 2 x 128
    grad_mu1 = tf.math.multiply(share[:, 0, :], lambda1 * tf.subtract(X, mu1)) # 500 x 128
    grad_mu2 = tf.math.multiply(share[:, 1, :], lambda2 * tf.subtract(X, mu2)) # 500 x 128

    grad_log_lambda1 = tf.math.multiply(share[:, 0, :], 0.5 - 0.5 * scaled_err_square1) # 500 x 128
    grad_log_lambda2 = tf.math.multiply(share[:, 1, :], 0.5 - 0.5 * scaled_err_square2) # 500 x 128

    log_fct = tf.math.log(1.0 / (tf.math.exp(logit_w) + 1))

    term1 = log_pdf1 + log_fct
    term2 = log_pdf2 + log_fct + logit_w
    grad_logit_w = tf.math.exp(term1 - log_norm) - tf.math.exp(term2 - log_norm)

    grad_out = tf.stack([grad_mu1, grad_mu2, grad_log_lambda1, grad_log_lambda2, grad_logit_w], axis=-1) # T x N x p
    grad_u = (-grad_mu1 - grad_mu2) * 3 # T x N

    return log_norm, grad_out, grad_u

def sigmoid(x):
    '''
    Description:
        numerical stable version to calculate sigmoid function
    '''
    out = tf.where(x >= 0, 1 / (1 + tf.exp(-x)), tf.exp(x) / (1 + tf.exp(x)))
    return tf.cast(out, dtype=tf.float32)

def grad_bernoulli(Y, Z, X, beta):
    '''
    Description:
        function to calculate the log pdf of Bernoulli distribution
        and its gradients with respect to parameters
    '''
    # X: T x N
    N = X.shape[-1]
    n = Z.shape[1]
    # Z: 500 x 6 x 8 [T, n, p] -> [T, N, n, p]
    Z_expand = tf.repeat(tf.expand_dims(Z, 1), repeats=N, axis=1)
    # X: [T, N] -> [T, N, n]
    X_expand = tf.repeat(tf.expand_dims(X, -1), repeats=n, axis=-1)
    logits = X_expand + tf.tensordot(Z_expand, beta, axes=[[3], [0]]) # T x N x n
    P = sigmoid(logits)
    # Y: [T, n] -> [T, N, n]
    Y_expand = tf.repeat(tf.expand_dims(Y, axis=1), repeats=N, axis=1)
    Y_expand = tf.cast(Y_expand, dtype=tf.float32)
    tmp = tf.multiply(Y_expand, 1.0-P) - tf.multiply(1 - Y_expand, P) # T x N x n
    grad_beta = tf.expand_dims(tmp, axis=-1) * Z_expand # T x N x n x p
    
    grad_u = 3 * tmp # T x N x n

    log_pdf = -tf.multiply(Y_expand, stable_log1pexp(-logits)) \
              -tf.multiply(1 - Y_expand, stable_log1pexp(logits)) # T x N x n

    return log_pdf, grad_beta, grad_u

def grad_normal(X):
    '''
    Description:
        function to calculate the log pdf of Gaussian distribution
        and its gradients with respect to parameters
    '''
    q_dist = tfd.Normal(loc=0.0, scale=3.0)
    log_pdf = q_dist.log_prob(X) # T x N
    grad_u = (- 1.0 / 3.0) * X # T x N
    return log_pdf, grad_u

class calculate_grad(tf.Module):
    '''
    Description:
        manually calculate the Hamiltonian and its gradients with respect to inputs
    Functions:
        - calculate_H: calculate the Hamiltonian
        - grad_total: calculate the gradients with respect to coords
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
        elif args.dist_name == 'normal':
            if not os.path.exists(os.path.join(args.data_pth, 'special/Z_T{}_n{}_p{}_normal.pkl'.format(args.T, args.n, args.p))):
                print('Generate data for normal distribution')
                generate_GLMM(T=args.T, n=args.n, p=args.p, data_pth=args.data_pth, true_mu=[0.0, 0.0], true_lambda=[3.0, 3.0],)
            else:
                self.Z = from_pickle(os.path.join(args.data_pth, 'special/Z_T{}_n{}_p{}_normal.pkl'.format(args.T, args.n, args.p)))
                self.Y = from_pickle(os.path.join(args.data_pth, 'special/Y_T{}_n{}_p{}_normal.pkl'.format(args.T, args.n, args.p)))
        else:
            raise NotImplementedError
    
    def get_target_log_prob(self, coords):
        d = self.args.p + 5
        theta = coords[:d]
        beta = coords[:self.args.p]
        mu1 = coords[self.args.p]
        mu2 = coords[self.args.p+1]
        log_lambda1 = coords[self.args.p+2]
        log_lambda2 = coords[self.args.p+3]
        logit_w = coords[self.args.p+4]

        u = coords[d:]
        X = tf.reshape(3.0 * u, shape=[self.args.T, self.args.N])

        log_bernoulli, _, _ = grad_bernoulli(self.Y, self.Z, X, beta)
        log_gauss_mix, _, _ = grad_gauss_mix(X, mu1, mu2, log_lambda1, log_lambda2, logit_w)
        log_normal, _ = grad_normal(X)

        log_W = log_bernoulli + tf.repeat(tf.expand_dims(log_gauss_mix - log_normal, axis=-1), repeats=self.args.n, axis=-1)
        max_log_W = tf.math.reduce_max(log_W, axis=1, keepdims=True) # T x 1 x n
        W_tmp = tf.math.exp(tf.math.subtract(log_W, max_log_W))
        log_pval = tf.math.log(tf.reduce_sum(W_tmp, axis=1)) + tf.squeeze(max_log_W, axis=1) - tf.math.log(tf.constant([self.args.N], dtype=tf.float32))
        log_pval = tf.reduce_sum(log_pval)

        log_prior = - 1.0 / 200.0 *tf.tensordot(theta, theta, axes=1) - d / 2.0 * tf.math.log(tf.constant([200 * math.pi], dtype=tf.float32))

        target_log_prob = log_prior + log_pval - 0.5*tf.tensordot(u, u, axes=1)
            
        return tf.squeeze(target_log_prob)
    
    def calculate_H(self, coords):
        d = self.args.p + 5
        theta = coords[:d]
        rho = coords[d:2*d]
        u, p = tf.split(coords[-2*(self.args.T * self.args.N):], 2)
        
        target_log_prob = self.get_target_log_prob(tf.concat([theta, u], axis=-1))
        H = -target_log_prob + 0.5*tf.tensordot(rho, rho, axes=1) + 0.5*tf.tensordot(p, p, axes=1)
            
        return tf.squeeze(H)

    def grad_total(self, coords):
        d = self.args.p + 5
        theta = coords[:d]
        beta = coords[:self.args.p]
        mu1 = coords[self.args.p]
        mu2 = coords[self.args.p+1]
        log_lambda1 = coords[self.args.p+2]
        log_lambda2 = coords[self.args.p+3]
        logit_w = coords[self.args.p+4]

        rho = coords[d:2*d]
        u, p = tf.split(coords[-2*(self.args.T * self.args.N):], 2)
        X = tf.reshape(3.0 * u, shape=[self.args.T, self.args.N])

        log_bernoulli, grad_beta_bernoulli, grad_u_bernoulli = grad_bernoulli(self.Y, self.Z, X, beta)
        log_gauss_mix, grad_out_gauss_mix, grad_u_gauss_mix = grad_gauss_mix(X, mu1, mu2, log_lambda1, log_lambda2, logit_w)
        log_normal, grad_u_normal = grad_normal(X)

        # T x N x n
        log_W = log_bernoulli + tf.repeat(tf.expand_dims(log_gauss_mix - log_normal, axis=-1), repeats=self.args.n, axis=-1)
        max_log_W = tf.math.reduce_max(log_W, axis=1, keepdims=True) # T x 1 x n
        W = tf.math.exp(tf.math.subtract(log_W, max_log_W)) # T x N x n
        sum_W = tf.reduce_sum(W, axis=1, keepdims=True) # T x N x n
        W = tf.math.divide(W, sum_W) # T x N x n

        grad_beta = tf.multiply(tf.expand_dims(W, axis=-1), grad_beta_bernoulli) # T x N x n x p
        grad_beta = tf.reduce_sum(grad_beta, axis=[0,1,2]) # p
        grad_out = tf.multiply(tf.expand_dims(W, axis=-1), 
                            tf.repeat(tf.expand_dims(grad_out_gauss_mix, axis=2), repeats=self.args.n, axis=2))
        grad_out = tf.reduce_sum(grad_out, axis=[0,1,2]) # 5
        grad_u = tf.multiply(W, grad_u_bernoulli + tf.repeat(tf.expand_dims(grad_u_gauss_mix - grad_u_normal, axis=-1), repeats=self.args.n, axis=-1))
        grad_u = tf.reduce_sum(grad_u, axis=-1)

        # adjust for the sign and other components
        grad_theta = -tf.concat([grad_beta, grad_out], axis=0) + 0.01 * theta
        grad_u = -tf.reshape(grad_u, shape=[self.args.aux_dim]) + u
        grad_rho = rho
        grad_p = p

        grads = tf.concat([grad_theta, grad_rho, grad_u, grad_p], axis=0)
        return grads

def numerical_grad_debug(coords, grad_func, target_dim, aux_dim):
    '''
    Description:
        calculate numerical gradients and then reshape:
        [grad_theta, grad_rho, grad_u, grad_p] -> [grad_rho, -grad_theta, grad_p, -grad_u]
    Args:
        coords: [theta, rho, u, p]
        func: hamiltonian function
        target_dim: theta.shape[0]
        aux_dim: u.shape[0]
    Return:
        out: [grad_rho, -grad_theta, grad_p, -grad_u]
    '''
    assert coords.shape[0] == 2 * (target_dim + aux_dim), 'incorrect specification of target or aux dim'
    # dcoords: gradient of "func" evaluated at "coords"
    # dcoords = tfp.math.value_and_gradient(func, coords)[-1]
    dcoords = grad_func(coords)
    c1, c2, c3, c4 = tf.split(dcoords, [target_dim, target_dim, aux_dim, aux_dim])
    out = tf.concat([c2, -c1, c4, -c3], axis=0)
    return out

class calculate_grad_mass(tf.Module):
    '''
    Description:
        manually calculate the Hamiltonian and its gradients with respect to inputs
        (mass matrix is non-identity)
    Functions:
        - calculate_H: calculate the Hamiltonian
        - grad_total: calculate the gradients with respect to coords
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
        
        if isinstance(args.rho_var, float):
            self.rho_precision_mat = tf.eye(args.target_dim, dtype=tf.float32) * 1.0 / args.rho_var
        else:
            self.rho_precision_mat = tf.linalg.diag(1.0 / tf.constant(args.rho_var, dtype=tf.float32))

        self.cal_grad_base = calculate_grad(args)
    
    def get_target_log_prob(self, coords):
        return self.cal_grad_base.get_target_log_prob(coords)
    
    def calculate_H(self, coords):
        d = self.args.p + 5
        theta = coords[:d]
        rho = coords[d:2*d]
        u, p = tf.split(coords[-2*(self.args.T * self.args.N):], 2)
        
        target_log_prob = self.get_target_log_prob(tf.concat([theta, u], axis=-1))
        # difference from identity mass matrix
        rho_expand = tf.expand_dims(rho, axis=-1) # target_dim x 1
        H = -target_log_prob + 0.5*tf.tensordot(p, p, axes=1) + \
            0.5 * tf.squeeze(tf.matmul(tf.matmul(rho_expand, self.rho_precision_mat, transpose_a=True), rho_expand))
            
        return tf.squeeze(H)

    def grad_total(self, coords):
        target_dim = self.args.p + 5
        aux_dim = self.args.T * self.args.N
        rho = coords[target_dim:2*target_dim]
        grads = self.cal_grad_base.grad_total(coords)
        grad_theta, _, grad_u, grad_p = tf.split(grads, [target_dim, target_dim, aux_dim, aux_dim], axis=0)
        grad_rho = grad_rho = tf.squeeze(tf.matmul(self.rho_precision_mat, tf.expand_dims(rho, axis=-1)))
        grads = tf.concat([grad_theta, grad_rho, grad_u, grad_p], axis=0)
        return grads


def integrator_one_step_mass(coords, func, derivs_func, h, target_dim, aux_dim, args):
    '''
    Description:
        Implement the one-step integrator in Appendix A of PM-HMC with rho's covariance mat not identity
    Return:
        coords_new: after one-step integration, same size as coords
    '''

    theta, rho, u, p = tf.split(coords, [target_dim, target_dim, aux_dim, aux_dim])
    if isinstance(args.rho_var, float):
        rho_precision_mat = tf.eye(target_dim, dtype=tf.float32) * 1.0 / args.rho_var
    else:
        rho_precision_mat = tf.linalg.diag(1.0 / tf.constant(args.rho_var, dtype=tf.float32))

    # rho_precision_mat = tf.eye(target_dim, dtype=tf.float32) * 1.0 / args.rho_var
    theta_tmp = theta + 0.5 * h * tf.squeeze(tf.matmul(rho_precision_mat, tf.expand_dims(rho, axis=-1)))
    u_tmp = u * math.cos(0.5 * h) + p * math.sin(0.5 * h)
    p_tmp = p * math.cos(0.5 * h) - u * math.sin(0.5 * h)
    
    coords_tmp = tf.concat([theta_tmp, rho, u_tmp, p_tmp], axis=0)
    dcoords = derivs_func(coords_tmp, func, target_dim, aux_dim)
    
    p_tmp = p_tmp + h * (dcoords[-aux_dim:] + u_tmp)
    rho_new = rho + h * dcoords[target_dim:2*target_dim]
    theta_new = theta_tmp + 0.5 * h * tf.squeeze(tf.matmul(rho_precision_mat, tf.expand_dims(rho_new, axis=-1)))
    u_new = u_tmp * math.cos(0.5 * h) + p_tmp * math.sin(0.5 * h)
    p_new = p_tmp * math.cos(0.5 * h) - u_tmp * math.sin(0.5 * h)

    coords_new = tf.concat([theta_new, rho_new, u_new, p_new], axis=0)
    return coords_new

def integrator_mass(coords, func, derivs_func, h, steps, target_dim, aux_dim, args):
    '''
    Description:
        Implement the multi-step integrator
    Args:
        coords: current value of [theta, rho, u, p]
        func: gradient function
        derivs_func: the function to calculate the derivative
        h: step size
        steps: number of integration steps
        target_dim: theta.shape[0]
        aux_dim: u.shape[0]
    Return:
        out: coords.shape[0] x (steps + 1)
    '''
    out = [coords]
    for i in range(steps):
        new = integrator_one_step_mass(out[-1], func, derivs_func, h, target_dim, aux_dim, args)
        out.append(new)
    out = tf.stack(out, axis=-1)
    return out







