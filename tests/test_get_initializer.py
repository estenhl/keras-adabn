"""Contains tests for testing the custom get_initializer module."""

import numpy as np

from tensorflow.keras.initializers import Zeros

from adabn.utils import get_initializer


def test_tensorflow_default():
    """Tests that get_initializer handles default tensorflow strings."""
    initializer = get_initializer('zeros')
    assert isinstance(initializer, Zeros), \
        ('Getting an initializer with a predefined tensorflow name does '
         'not yield the correct object')

def test_tensorflow_constant():
    """Tests that get_initializer handles constant values."""
    initializer = get_initializer(2.5)

    assert hasattr(initializer, 'value'), \
        ('Getting an initializer with a constant does not yield an object '
         'which has a constant value')
    assert 2.5 == initializer.value, \
        ('Getting an initializer with a constant does not yield an object '
         'with the correct constant value')

def test_mixed_type_list_raises_error():
    """Tests that calling get_initializer with a mix of default
    tensorflow strings and constants raises an error.
    """
    exception = False

    try:
        get_initializer(['ones', 5])
    except AssertionError:
        exception = True

    assert exception, \
        ('Calling get_initializer with list of mixed types does not raise an '
         'exception')

def test_initializer_broadcasting():
    """Tests that get_initializer is able to broadcast domain-specific
    constant values across a tensor with a feature-dimension.
    """
    shape = [2, 1, 3]
    initial = [1, 2]
    initializer = get_initializer(initial, shape=shape)
    variable = initializer(shape).numpy()

    expected = np.asarray([
        [[1, 1, 1]],
        [[2, 2, 2]]
    ])

    assert np.array_equal(expected, variable), \
        ('get_initializer is not able to broadcast initial values into '
         'requested shape')
