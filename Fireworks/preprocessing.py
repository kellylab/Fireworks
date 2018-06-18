from functools import lru_cache
import numpy as np

@lru_cache(maxsize=32)
def one_hot(index, max):

    hot = np.zeros(max)
    hot[index] = 1
    return hot
