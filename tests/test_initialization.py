"""Contains tests testing the initialization of the
AdaptiveBatchNormalization layer.
"""
import tensorflow as tf

from tensorflow.keras.layers import Conv3D, Input

from adabn import AdaptiveBatchNormalization

def test_shape_singular():
    inputs = Input((5, 5, 5, 1))
    domains = Input((), dtype=tf.int32)

    tensor = AdaptiveBatchNormalization(domains=2)([inputs, domains])

    assert [5, 5, 5, 1] == tensor.shape[1:], \
        ('AdaptiveBatchNormalization layer has a different shape than '
         'the input tensor')

def test_shape():
    inputs = Input((5, 5, 5, 32))
    domains = Input((), dtype=tf.int32)
    inputs = Conv3D(32, (3, 3, 3), padding='same')(inputs)

    tensor = AdaptiveBatchNormalization(domains=2)([inputs, domains])

    assert [5, 5, 5, 32] == tensor.shape[1:], \
        ('AdaptiveBatchNormalization layer has a different shape than '
         'the input tensor')

