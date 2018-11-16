# -*- coding: utf-8 -*-
"""Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine import Layer, InputSpec
from keras_contrib import initializers
from keras_contrib import regularizers
from keras_contrib import constraints
from keras_contrib import backend as K
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects


class BatchNormalization(Layer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        momentum: Momentum for the moving mean and the moving variance.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    @interfaces.legacy_batchnorm_support
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=True)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape



class BatchRenormalization(Layer):
    """Batch renormalization layer (Sergey Ioffe, 2017).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchRenormalization`.
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        epsilon: small float > 0. Fuzz parameter.
            Theano expects epsilon >= 1e-5.
        r_max_value: Upper limit of the value of r_max.
        d_max_value: Upper limit of the value of d_max.
        t_delta: At each iteration, increment the value of t by t_delta.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
            Note that the order of this list is [gamma, beta, mean, std]
        beta_initializer: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_initializer: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        gamma_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the gamma vector.
        beta_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the beta vector.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self, axis=-1, momentum=0.99, center=True, scale=True, epsilon=1e-3,
                 r_max_value=3., d_max_value=5., t_delta=1e-3, weights=None, beta_initializer='zero',
                 gamma_initializer='one', moving_mean_initializer='zeros',
                 moving_variance_initializer='ones', gamma_regularizer=None, beta_regularizer=None,
                 beta_constraint=None, gamma_constraint=None, **kwargs):
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        self.r_max_value = r_max_value
        self.d_max_value = d_max_value
        self.t_delta = t_delta
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        super(BatchRenormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape,
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         name='{}_gamma'.format(self.name))
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape,
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        name='{}_beta'.format(self.name))
        else:
            self.beta = None

        self.running_mean = self.add_weight(shape, initializer=self.moving_mean_initializer,
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)

        self.running_variance = self.add_weight(shape, initializer=self.moving_variance_initializer,
                                                name='{}_running_std'.format(self.name),
                                                trainable=False)

        self.r_max = K.variable(1, name='{}_r_max'.format(self.name))

        self.d_max = K.variable(0, name='{}_d_max'.format(self.name))

        self.t = K.variable(0, name='{}_t'.format(self.name))

        self.t_delta_tensor = K.constant(self.t_delta)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs, training=None):
        assert self.built, 'Layer must be built before being called'
        input_shape = K.int_shape(inputs)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        mean_batch, var_batch = K.moments(inputs, reduction_axes, shift=None, keep_dims=False)
        std_batch = (K.sqrt(var_batch + self.epsilon))

        r = std_batch / (K.sqrt(self.running_variance + self.epsilon))
        r = K.stop_gradient(K.clip(r, 1 / self.r_max, self.r_max))

        d = (mean_batch - self.running_mean) / K.sqrt(self.running_variance + self.epsilon)
        d = K.stop_gradient(K.clip(d, -self.d_max, self.d_max))

        if sorted(reduction_axes) == range(K.ndim(inputs))[:-1]:
            x_normed_batch = (inputs - mean_batch) / std_batch
            x_normed = (x_normed_batch * r + d) * self.gamma + self.beta
        else:
            # need broadcasting
            broadcast_mean = K.reshape(mean_batch, broadcast_shape)
            broadcast_std = K.reshape(std_batch, broadcast_shape)
            broadcast_r = K.reshape(r, broadcast_shape)
            broadcast_d = K.reshape(d, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)

            x_normed_batch = (inputs - broadcast_mean) / broadcast_std
            x_normed = (x_normed_batch * broadcast_r + broadcast_d) * broadcast_gamma + broadcast_beta

        # explicit update to moving mean and standard deviation
        self.add_update([K.moving_average_update(self.running_mean, mean_batch, self.momentum),
                         K.moving_average_update(self.running_variance, std_batch ** 2, self.momentum)], inputs)

        # update r_max and d_max
        r_val = self.r_max_value / (1 + (self.r_max_value - 1) * K.exp(-self.t))
        d_val = self.d_max_value / (1 + ((self.d_max_value / 1e-3) - 1) * K.exp(-(2 * self.t)))

        self.add_update([K.update(self.r_max, r_val),
                         K.update(self.d_max, d_val),
                         K.update_add(self.t, self.t_delta_tensor)], inputs)

        if training in {0, False}:
            return x_normed
        else:
            def normalize_inference():
                if sorted(reduction_axes) == range(K.ndim(inputs))[:-1]:
                    x_normed_running = K.batch_normalization(
                        inputs, self.running_mean, self.running_variance,
                        self.beta, self.gamma,
                        epsilon=self.epsilon)

                    return x_normed_running
                else:
                    # need broadcasting
                    broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
                    broadcast_running_std = K.reshape(self.running_variance, broadcast_shape)
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                    broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                    x_normed_running = K.batch_normalization(
                        inputs, broadcast_running_mean, broadcast_running_std,
                        broadcast_beta, broadcast_gamma,
                        epsilon=self.epsilon)

                    return x_normed_running

            # pick the normalized form of inputs corresponding to the training phase
            # for batch renormalization, inference time remains same as batchnorm
            x_normed = K.in_train_phase(x_normed, normalize_inference, training=True)

            return x_normed

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'center': self.center,
                  'scale': self.scale,
                  'momentum': self.momentum,
                  'gamma_regularizer': initializers.serialize(self.gamma_regularizer),
                  'beta_regularizer': initializers.serialize(self.beta_regularizer),
                  'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
                  'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
                  'beta_constraint': constraints.serialize(self.beta_constraint),
                  'gamma_constraint': constraints.serialize(self.gamma_constraint),
                  'r_max_value': self.r_max_value,
                  'd_max_value': self.d_max_value,
                  't_delta': self.t_delta}
        base_config = super(BatchRenormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'BatchRenormalization': BatchRenormalization})
