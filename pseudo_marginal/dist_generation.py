import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys, os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

from utils import to_pickle

def generate_GLMM(true_beta=[-1.1671, 2.4665, -0.1918, -1.0080, 0.6212, 0.6524, 1.5410, 0.2653], 
                  true_mu=[0.0, 3.0], true_lambda=[10.0, 3.0], true_w=0.8, 
                  T=500, n=6, p=8, data_pth=None, save_flag=True):
    '''
    Description:
        generate data from generalized linear mixed model
    Args:
        true_beta: of length args.p
        true_mu: of length 2, mean for each Gaussian component
        true_lambda: of length 2, precision for each Gaussian component
        true_w: a scalar, weight of the first component
        T: number of subjects
        n: observations for each subject
        p: dimension of beta
        data_pth: where to store the generated data
        save_flag: whether to save data
    Return:
        X: latent variables
        Y: responses
        Z: the variable being multiplied with beta
    '''
    # check the shapes
    assert len(true_beta) == p, 'incorrect shape of lambda'
    assert len(true_mu) == len(true_lambda), 'the shapes of mu and lambda do not match'
    
    # define gaussian mixture
    gauss_mix = tfd.Mixture(
        cat=tfd.Categorical(probs=[true_w, 1.-true_w]),
        components=[
            tfd.Normal(loc=true_mu[0], scale=true_lambda[0]**(-0.5)),
            tfd.Normal(loc=true_mu[1], scale=true_lambda[1]**(-0.5))
        ]
    )
    # X.shape = [T]
    X = gauss_mix.sample([T])
    # X_expand.shape = [T, n]
    X_expand = tf.repeat(tf.expand_dims(X, axis=-1), repeats=n, axis=-1)

    # define standard normal
    normal = tfd.Normal(loc=[0.0 for _ in range(p)], scale=[1.0 for _ in range(p)])
    # Z.shape = [T, n, p]
    Z = normal.sample([T, n])

    # define bernoulli distribution
    logits = X_expand + tf.tensordot(Z, tf.constant(true_beta), axes=[[2], [0]])
    bernoulli = tfd.Bernoulli(logits=logits)
    # Y.shape = [T, n]
    Y = bernoulli.sample()

    if save_flag:
        if not os.path.exists(data_pth):
            os.makedirs(data_pth)
        # save these data
        to_pickle(X, os.path.join(data_pth, 'X_T{}_n{}_p{}.pkl'.format(T, n, p)))
        to_pickle(Z, os.path.join(data_pth, 'Z_T{}_n{}_p{}.pkl'.format(T, n, p)))
        to_pickle(Y, os.path.join(data_pth, 'Y_T{}_n{}_p{}.pkl'.format(T, n, p)))
    
    return X, Y, Z

if __name__ == '__main__':
    # true_beta = tf.constant([-1.1671, 2.4665, -0.1918, -1.0080, 0.6212, 0.6524, 1.5410, 0.2653], dtype=tf.float32)
    # true_mu = tf.constant([0.0, 3.0], dtype=tf.float32)
    # true_lambda = tf.constant([10.0, 3.0], dtype=tf.float32)
    # true_w = tf.constant([0.8])

    true_beta = [-1.1671, 2.4665, -0.1918, -1.0080, 0.6212, 0.6524, 1.5410, 0.2653]
    true_mu = [0.0, 3.0]
    true_lambda = [10.0, 3.0]
    true_w = 0.8

    T = 500
    n = 6
    p = 8

    data_pth = '/home/r11user3/Yuxi/HMC/pseudo_marginal/data'

    X, Z, Y = generate_GLMM(true_beta, true_mu, true_lambda, true_w, 
                  T, n, p, data_pth, save_flag=True)
