import numpy as np

from tensorflow import constant_initializer
from tensorflow.keras.initializers import get
from typing import Any, List


def get_initializer(identifier: Any, *, shape: List = None):
    if isinstance(identifier, list):
        numbers = [isinstance(x, (int, float, complex)) for x in identifier]

        assert all(numbers), \
            ('Unable to get variable initializer for a list of constant '
             'values with mixed types')

        if shape is not None:
            assert len(identifier) == shape[0], \
                ('Unable to broadcast initial variable values into shape '
                 'with different first dimension')
            assert all([x == 1 for x in shape[1:-1]]), \
                ('Only able to broadcast initial variable values into shapes '
                 'on the form [X, 1, ..., 1, Y]')
            identifier = np.concatenate([np.repeat(value, shape[-1]) \
                                         for value in identifier])

        return constant_initializer(identifier)
    elif isinstance(identifier, int) or isinstance(identifier, float):
        return constant_initializer(float(identifier))
    elif isinstance(identifier, str):
        return get(identifier)
    else:
        raise ValueError(f'Invalid variable initializer {identifier}')
