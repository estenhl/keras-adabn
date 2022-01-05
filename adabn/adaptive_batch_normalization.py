import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer
from typing import Any, List, Tuple

from .utils import get_initializer


class AdaptiveBatchNormalization(Layer):
    """A Keras Layer mimicking the behaviour of the built-in
    BatchNormalization Layer, but with individual normalization
    values for a set of given domains. The layer is called with a tuple
    [batch, domain] where the domain is a tensor of (singular) domain
    keys identifying which values are used. During training, only the
    moving values (mean and variance) for the given domain are updated.
    See also
    tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization"""
    def __init__(self, domains: int = 1, axis: int = -1,
                 momentum: float = 0.99, epsilon: float = 1e-3,
                 center: bool = True, scale: bool = True,
                 beta_initializer: Any ='zeros',
                 gamma_initializer: Any = 'ones',
                 moving_mean_initializer: Any = 'zeros',
                 moving_variance_initializer: Any = 'ones',
                 **kwargs):
        super().__init__(**kwargs)

        self.domains = domains
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer

    def build(self, input_shape: Tuple):
        """Builds the layer by instantiating the trainable and moving
        variables"""
        # Drop domain-part of the input
        input_shape = input_shape[0]

        # Resolve negative indexes, e.g. -1
        axis = self.axis if self.axis > 0 else len(input_shape) + self.axis

        param_shape = [input_shape[i] if axis == i else 1 \
                       for i in range(len(input_shape))]
        param_shape = [self.domains] + param_shape

        self.beta = self.add_weight(
            name=f'beta',
            shape=param_shape,
            dtype=tf.float32,
            initializer=get_initializer(self.beta_initializer,
                                        shape=param_shape)
        )

        self.gamma = self.add_weight(
            name=f'gamma',
            shape=param_shape,
            dtype=tf.float32,
            initializer=get_initializer(self.gamma_initializer,
                                        shape=param_shape)
        )

        self.moving_mean = self.add_weight(
            name=f'moving_mean',
            shape=param_shape,
            dtype=tf.float32,
            trainable=False,
            initializer=get_initializer(self.moving_mean_initializer,
                                        shape=param_shape)
        )

        self.moving_variance = self.add_weight(
            name=f'moving_variance',
            shape=param_shape,
            dtype=tf.float32,
            trainable=False,
            initializer=get_initializer(self.moving_variance_initializer,
                                        shape=param_shape)
        )

    def _train(self, inputs: tf.Tensor, domain: tf.Tensor) -> tf.Tensor:
        """Returns the normalized batch based on batch statistics:

            gamma * (batch - mean(batch)) /
            sqrt(variance(batch) + epsilon) + beta"""
        mean = tf.reduce_mean(inputs, axis=0)
        variance = tf.math.reduce_variance(inputs, axis=0)


        self.moving_mean[domain].assign(self.moving_mean[domain] * self.momentum + \
                                        mean * (1 - self.momentum))
        self.moving_variance[domain].assign(self.moving_variance[domain] * self.momentum + \
                                            variance * (1 - self.momentum))

        return self._calculate(inputs, domain, mean=mean, variance=variance)

    def _predict(self, inputs: tf.Tensor, domain: tf.Tensor) -> tf.Tensor:
        """Returns the normalized batch based on the fitted moving
        variables:

            gamma * (batch - moving_mean) /
            sqrt(moving_variance + epsilon) + beta"""
        return self._calculate(inputs, domain,
                               mean=self.moving_mean[domain],
                               variance=self.moving_variance[domain])

    def _calculate(self, inputs: tf.Tensor, domain: tf.Tensor, mean: tf.Tensor,
                   variance: tf.Tensor) -> tf.Tensor:
        """Calculates the normalized batch values based on the given
        variance and mean"""
        normalized = (inputs - mean) / tf.math.sqrt(variance + self.epsilon)

        if self.scale:
            normalized = normalized * self.gamma[domain]

        if self.center:
            normalized = normalized + self.beta[domain]

        return normalized

    def call(self, inputs: List[tf.Tensor], *args, training: bool = False,
             **kwargs) -> tf.Tensor:
        """Builds the graph operations depending on if the model is in
        training (training=True) or inference (training=False) phase.
        For the former, the batch is normalized according to the batch
        statistics, and the moving mean and average is updated. For the
        latter, the batch is normalized according to the fitted
        moving variables. For both cases, the given domain decides which
        variables are used.

        Args:
            inputs: A tuple of tensors. The first element is contains
                the batch data, the second contains the domain. For
                simplicity, all datapoints of a batch should be from
                the same domain (e.g. the second element
                should be singular)
            training: Boolean determining if the model is in training or
                inference mode

        Returns:
            A tf.Tensor with the normalized batch values

        Raises:
            AssertionError (at construction time): If the domain-tensor
                does not have datatype int
            IllegalArgumentError (at runtime): If the domain-tensor is
                not singular
            IllegalArgumentError: If the domain-tensor contains a value
                outside the scope of allowed domains (i.e. if not
                0<=domain<self.domains)
        """
        inputs, domain = inputs

        if isinstance(domain, tf.Tensor):
            assert domain.dtype.is_integer, 'Domain-tensor must have dtype int'
        else:
            raise ValueError(('Domain (second element of the input-tuple) '
                              'must be either a numpy-array or a tensor'))

        # Validates that each batch only contains a single domain
        tf.Assert(tf.reduce_all(domain[0] == domain),
                  ['Batch had multiple domains', domain])

        domain = domain[0]

        # Validates that the given domain is within the allowed bounds
        tf.Assert(domain >= 0,
                  [f'Batch had illegal domain (< 0)', domain])
        tf.Assert(domain < self.domains,
                  [f'Batch had illegal domain (>= {self.domains})', domain])

        if training:
            return self._train(inputs, domain)

        return self._predict(inputs, domain)

