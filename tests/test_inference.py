"""Contains tests comparing the behaviour of the
AdaptiveBatchNormalization layer with the regular BatchNormalization
layer during inference.
"""

import numpy as np
import tensorflow as tf

from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input

from adabn import AdaptiveBatchNormalization


def test_inference_single_domain_no_training():
    """Tests the inference of an untrained AdaptiveBatchNormalization
    layer on a single domain.
    """
    np.random.seed(42)

    adabn = AdaptiveBatchNormalization()
    bn = BatchNormalization()

    inputs = np.random.uniform(3, 4, (10, 10))
    expected = bn(inputs, training=False).numpy()
    inputs = [inputs, np.repeat(0, 10)]
    outputs = adabn(inputs, training=False).numpy()

    assert np.allclose(outputs, expected, 1e-4), \
        ('AdaptiveBatchNormalization does not return the expected result '
         'for a single domain when doing inference without training')


def test_inference_two_domains_no_training():
    """Tests the inference of an untrained AdaptiveBatchNormalization
    layer on a two domains.
    """
    np.random.seed(42)

    gamma_initializers = [1, 2]
    adabn = AdaptiveBatchNormalization(domains=2,
                                       gamma_initializer=gamma_initializers)
    bn1 = BatchNormalization(gamma_initializer='ones')
    bn2 = BatchNormalization(gamma_initializer=tf.constant_initializer(2))

    inputs = np.random.uniform(3, 4, (10, 10))
    expected = [bn1(inputs, training=False).numpy()]
    expected.append(bn2(inputs, training=False).numpy())
    outputs = [adabn([inputs, np.repeat(0, 10)], training=False).numpy()]
    outputs.append(adabn([inputs, np.repeat(1, 10)], training=False).numpy())

    assert not np.allclose(outputs[0], outputs[1], 1e-4), \
        ('AdaptiveBatchNormalization does not differentiate between domains '
         'during inference')

    for i in range(2):
        assert np.allclose(expected[i], outputs[i], 1e-4), \
            ('AdaptiveBatchNormalization does not differentiate between '
             'domains correctly during inference')


def test_inference_from_tensors():
    """Tests the inference of an untrained AdaptiveBatchNormalization
    layer with tensors as input.
    """
    tf.random.set_seed(42)

    gamma_initializers = [1, 2]
    adabn = AdaptiveBatchNormalization(domains=2,
                                       gamma_initializer=gamma_initializers)
    bn1 = BatchNormalization(gamma_initializer='ones')
    bn2 = BatchNormalization(gamma_initializer=tf.constant_initializer(2))

    inputs = tf.random.uniform((10, 10), dtype=float)
    expected = [bn1(inputs, training=False).numpy()]
    expected.append(bn2(inputs, training=False).numpy())
    outputs = [adabn([inputs, np.repeat(0, 10)], training=False).numpy()]
    outputs.append(adabn([inputs, np.repeat(1, 10)], training=False).numpy())

    for i in range(2):
        assert np.allclose(expected[i], outputs[i], 1e-4), \
            ('AdaptiveBatchNormalization does not properly do inference on '
             'tensors')


def test_inference_as_model():
    """Tests the inference of an untrained AdaptiveBatchNormalization
    layer when included as a layer in a functional model.
    """
    tf.random.set_seed(42)

    gamma_initializers = [1, 2]
    inputs = Input((10,))
    domain = Input((), dtype=tf.int32)
    adabn = AdaptiveBatchNormalization(domains=2,
                                       gamma_initializer=gamma_initializers)
    adabn = adabn([inputs, domain])
    adamodel = Model([inputs, domain], adabn)

    inputs = Input((10,))
    bn = BatchNormalization(gamma_initializer='ones')
    bn = bn(inputs)
    bnmodel1 = Model(inputs, bn)

    inputs = Input((10,))
    bn = BatchNormalization(gamma_initializer=tf.constant_initializer(2))
    bn = bn(inputs)
    bnmodel2 = Model(inputs, bn)

    inputs = np.random.uniform(3, 4, (10, 10))
    expected = [bnmodel1.predict(inputs)]
    expected.append(bnmodel2.predict(inputs))
    outputs = [adamodel.predict([inputs, np.repeat(0, 10)])]
    outputs.append(adamodel.predict([inputs, np.repeat(1, 10)]))

    for i in range(2):
        assert np.allclose(expected[i], outputs[i], 1e-4), \
            ('AdaptiveBatchNormalization does not properly do inference as '
             'part of a functional model')

def test_multidomain_batch_raises_exception():
    """Tests that the AdaptiveBatchNormalization layer raises an error
    if it receives multiple domains in a single batch.
    """
    np.random.seed(42)

    adabn = AdaptiveBatchNormalization()

    inputs = np.random.uniform(3, 4, (10, 10))
    inputs = [inputs, np.concatenate([np.repeat(0, 5), np.repeat(1, 5)])]

    exception = False

    try:
        adabn(inputs, training=False)
    except InvalidArgumentError:
        exception = True

    assert exception, \
        ('Having batches with multiple domains does not raise an exception')

def test_invalid_domain():
    """Tests that the AdaptiveBatchNormalization layer raises an error
    if it gets an unknown domain as input.
    """
    np.random.seed(42)

    adabn = AdaptiveBatchNormalization(domains=2)

    inputs = np.random.uniform(3, 4, (10, 10))
    inputs = [inputs, np.repeat(2, 10)]

    exception = False

    try:
        adabn(inputs, training=False).numpy()
    except InvalidArgumentError:
        exception = True

    assert exception, \
        ('Calling AdaptiveBatchNormalization with an invalid domain does not '
         'raise an exception')
