from functools import lru_cache
import math
import numpy as np
from Fireworks import pipeline as pl

@lru_cache(maxsize=32)
def one_hot(index, max):

    hot = np.zeros(max)
    hot[index] = 1
    return hot

def train_test_split(pipe, test=.2):
    """
    Splits input pipe into a training pipe and a test pipe.
    """
    if not hasattr(pipe, '__getitem__'):
        raise ValueError("Input pipe must be indexable via __getitem__")

    l = len(pipe)
    num_test = math.floor(l*test)
    indices = [i for i in range(l)]
    test_indices = sorted(np.random.choice(indices, num_test, replace=False))
    train_indices = [i for i in indices if i not in test_indices]

    test_pipe = pl.IndexMapperPipe(inputs={'data': pipe}, input_indices=range(0,len(test_indices)), output_indices=test_indices)
    train_pipe = pl.IndexMapperPipe(inputs={'data': pipe}, input_indices=range(0,len(train_indices)), output_indices=train_indices)

    return train_pipe, test_pipe

def oversample(): pass

def apply_noise(): pass
