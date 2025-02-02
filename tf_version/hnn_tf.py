import tensorflow as tf
import keras
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PARENT_DIR)
from tf_version.utils_tf import lfrog
from tf_version.get_args import get_args

args = get_args()

@keras.saving.register_keras_serializable()
class HNN(keras.Model):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, grad_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        # the neural network that we use: MLP
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        self.grad_type = grad_type
        self.input_dim = input_dim
    
    def call(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert len(y.shape) == 2 and y.shape[1] == args.input_dim, "Output tensor should have shape [batch_size, 2]"
        # dic1: split each input dim as a separate tensor
        dic1 = (tf.split(y, num_or_size_splits=y.shape[1], axis=1))
        # answer1: a tuple, the first one is the tensor formed by the first half of the column, while the second one is the rest
        # each with dimension: batch_size x input_dim/2
        answer1 = tf.concat(dic1[0:int(args.input_dim/2)], axis=1), tf.concat(dic1[int(args.input_dim/2):args.input_dim], axis=1)
        return answer1
    
    def get_config(self):
        return {"input_dim": self.input_dim, "differentiable_model": self.differentiable_model, 
                "grad_type": self.grad_type, "baseline": self.baseline,
                "assume_canonical_coords": self.assume_canonical_coords}
    
    def lfrog_time_derivative(self, x, dt):
        return lfrog(fun=self.time_derivative, y0=x, t=0, dt=dt)
    
    def time_derivative(self, x, t=None, separate_fields=False):
        # what does solenoidal and conservative mean here? 
        # Helmholtz’ Theorem: certain differentiable vector fields can be resolved into the sum of 
        # an irrotational (curl-free / conservative) vector field and a solenoidal(divergence-free)
        # vector field
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            F1, F2 = self.call(x) # traditional forward pass
            if self.grad_type != 'solenoidal':
                F_sum = tf.reduce_sum(F1)
            elif self.grad_type != 'conservative':
                F_sum = tf.reduce_sum(F2)

        conservative_field = tf.zeros_like(x) # start out with both components set to 0
        solenoidal_field = tf.zeros_like(x)

        # conservative case
        if self.grad_type != 'solenoidal':
            dF1 = tape.gradient(F_sum, x) # gradients for conservative field
            conservative_field = tf.matmul(dF1, tf.eye(*self.M.shape))

        # solenoidal case
        if self.grad_type != 'conservative':
            dF2 = tape.gradient(F_sum, x) # gradients for solenoidal field
            solenoidal_field = tf.matmul(dF2, tf.transpose(self.M))

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field
    
    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = tf.eye(n)
            M = tf.concat([M[n//2:], -M[:n//2]], axis=0)
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = tf.ones(shape=(n,n)) # matrix of ones
            M *= 1 - tf.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1

            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M