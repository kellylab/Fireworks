from functools import lru_cache
import math
import numpy as np
from Fireworks import source as ds

@lru_cache(maxsize=32)
def one_hot(index, max):

    hot = np.zeros(max)
    hot[index] = 1
    return hot

def train_test_split(source, test=.2):
    """
    Splits input source into a training source and a test source.
    """
    if not hasattr(source, '__getitem__'):
        raise ValueError("Input source must be indexable via __getitem__")

    l = len(source)
    num_test = math.floor(l*test)
    indices = [i for i in range(l)]
    test_indices = sorted(np.random.choice(indices, num_test, replace=False))
    train_indices = [i for i in indices if i not in test_indices]

    test_source = ds.IndexMapperSource(inputs={'data': source}, input_indices=range(0,len(test_indices)), output_indices=test_indices)
    train_source = ds.IndexMapperSource(inputs={'data': source}, input_indices=range(0,len(train_indices)), output_indices=train_indices)

    return train_source, test_source

def oversample(): pass

def apply_noise(): pass
