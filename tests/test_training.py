import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization

from adabn import AdaptiveBatchNormalization


def test_training_output():
    np.random.seed(42)

    adabn = AdaptiveBatchNormalization()
    bn = BatchNormalization()

    inputs = np.random.uniform(3, 4, (10, 10))
    expected = bn(inputs, training=True).numpy()
    inputs = [inputs, np.repeat(0, 10)]
    outputs = adabn(inputs, training=True).numpy()

    assert np.allclose(outputs, expected, 1e-4), \
        ('AdaptiveBatchNormalization does not return the expected result '
         'for a single domain during training')

def test_training_output_two_domains():
    np.random.seed(42)

    gamma_initializers = [1, 2]
    adabn = AdaptiveBatchNormalization(domains=2,
                                       gamma_initializer=gamma_initializers)
    bn1 = BatchNormalization(gamma_initializer='ones')
    bn2 = BatchNormalization(gamma_initializer=tf.constant_initializer(2))

    inputs = np.random.uniform(3, 4, (10, 10))
    expected = [bn1(inputs, training=True).numpy()]
    expected.append(bn2(inputs, training=True).numpy())
    outputs = [adabn([inputs, np.repeat(0, 10)], training=True).numpy()]
    outputs.append(adabn([inputs, np.repeat(1, 10)], training=True).numpy())

    assert not np.allclose(outputs[0], outputs[1], 1e-4), \
        ('AdaptiveBatchNormalization does not differentiate between domains '
         'during inference')

    for i in range(2):
        assert np.allclose(expected[i], outputs[i], 1e-4), \
            ('AdaptiveBatchNormalization does not differentiate between '
             'domains correctly during inference')

def test_training_updates_moving_variables():
    bn = BatchNormalization()
    inputs = np.random.uniform(3, 4, (10, 10))

    adabn = AdaptiveBatchNormalization()
    bn = BatchNormalization()

    inputs = np.random.uniform(3, 4, (10, 10))
    bn(inputs, training=True).numpy()
    inputs = [inputs, np.repeat(0, 10)]
    adabn(inputs, training=True).numpy()

    moving_mean = adabn.moving_mean.numpy()
    moving_variance = adabn.moving_variance.numpy()

    expected_moving_mean = bn.moving_mean.numpy()
    expected_moving_variance = bn.moving_variance.numpy()

    assert np.allclose(expected_moving_mean, moving_mean, 1e-4), \
        ('Training AdaptiveBatchNormalization on a batch does not update '
         'the moving mean correctly')
    assert np.allclose(expected_moving_variance, moving_variance, 1e-4), \
        ('Training AdaptiveBatchNormalization on a batch does not update '
         'the moving variance correctly')

def test_training_updates_moving_variables_multiple_domains():
    adabn = AdaptiveBatchNormalization(domains=2)
    bn1 = BatchNormalization()
    bn2 = BatchNormalization()

    inputs = np.random.uniform(3, 4, (10, 10))
    bn1(inputs, training=True).numpy()
    inputs = [inputs, np.repeat(0, 10)]
    adabn(inputs, training=True).numpy()
    expected_moving_mean = [bn1.moving_mean.numpy()]
    expected_moving_variance = [bn1.moving_variance.numpy()]

    inputs = np.random.uniform(3, 4, (10, 10))
    bn2(inputs, training=True).numpy()
    inputs = [inputs, np.repeat(1, 10)]
    adabn(inputs, training=True).numpy()
    expected_moving_mean.append(bn2.moving_mean.numpy())
    expected_moving_variance.append(bn2.moving_variance.numpy())

    moving_mean = adabn.moving_mean.numpy()
    moving_variance = adabn.moving_variance.numpy()

    for i in range(2):
        assert np.allclose(expected_moving_mean[i], moving_mean[i], 1e-4), \
            ('Training AdaptiveBatchNormalization on a batch does not update '
             'the moving mean correctly with multiple domains')
        assert np.allclose(expected_moving_variance[i], moving_variance[i], 1e-4), \
            ('Training AdaptiveBatchNormalization on a batch does not update '
             'the moving variance correctly with multiple domains')

def test_training_converges():
    adabn = AdaptiveBatchNormalization(domains=2)
    bns = [BatchNormalization(), BatchNormalization()]

    inputs = [np.random.normal(0, 1, (10, 10)),
              np.random.normal(2, 3, (10, 10))]

    for _ in range(10):
        batches = [np.random.normal(0, 1, (10, 10)),
                   np.random.normal(2, 3, (10, 10))]

        for i in range(2):
            bns[i](batches[i], training=True)
            adabn([batches[i], np.repeat(i, 10)], training=True)

    expected_predictions = [bns[i](inputs[i], training=False) \
                            for i in range(2)]
    predictions = [adabn([inputs[i], np.repeat(i, 10)], training=False) \
                   for i in range(2)]

    for i in range(2):
        assert np.allclose(expected_predictions[i], predictions[i], 1e-4), \
            ('AdaptiveBatchNormalization does not converge to the same '
             'activations as regular BatchNorm')

