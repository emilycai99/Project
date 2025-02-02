import tensorflow as tf
import keras
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

@keras.saving.register_keras_serializable()
class HNN(keras.Model):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, args, differentiable_model, grad_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.input_dim = 2 * (args.target_dim + args.aux_dim)
        # the neural network that we use: MLP
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.grad_type = grad_type
        self.args = args
    
    def call(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert len(y.shape) == 2 and y.shape[1] == self.args.nn_out_dim, \
            "Output tensor should have shape [batch_size, {}]".format(self.args.nn_out_dim)
        
        # answer1: a tuple, the first one is the tensor formed by the first half of the column, while the second one is the rest
        # each with dimension: batch_size x args.nn_out_dim/2
        answer1 = tf.split(y, num_or_size_splits=2, axis=-1)
        return answer1
    
    def get_config(self):
        return {'args': self.args, "differentiable_model": self.differentiable_model, 
                "grad_type": self.grad_type, "baseline": self.baseline,
                "assume_canonical_coords": self.assume_canonical_coords}
     
    def time_derivative(self, x, separate_fields=False):
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
            if self.assume_canonical_coords:
                conservative_field = dF1
            else:
                raise NotImplementedError

        # solenoidal case
        if self.grad_type != 'conservative':
            dF2 = tape.gradient(F_sum, x) # gradients for solenoidal field
            dtheta, drho, du, dp = tf.split(dF2, num_or_size_splits=[self.args.target_dim, self.args.target_dim, 
                                                                     self.args.aux_dim, self.args.aux_dim], axis=-1)
            solenoidal_field = tf.concat([drho, -dtheta, dp, -du], axis=-1)

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field