from functools import lru_cache
import numpy as np

@lru_cache(maxsize=32)
def one_hot(index, max):
    """
    This converts an integer to a one-hot vector, which is a format that is often used by statistical classifiers to represent predictions.
    Args:
        - index: The index in the one-hot array to set as 1
        - max: The size of the one-hot array.

    Returns:
        - hot: A one-hot array, which consists of all 0s except a 1 at a certain index which corresponds to the label classification.
    """
    hot = np.zeros(max)
    hot[index] = 1
    return hot

def index_to_list(index):
    """
    Converts an index to a list. This is used by some of the methods in message.py.
    """
    if type(index) is slice:
        index = slice_to_list(index)
    if type(index) is int:
        index = [index]
    return index

def slice_to_list(s):
    """
    Converts a slice object to a list of indices. This is used by some of the methods in message.py.
    """
    step = s.step or 1
    start = s.start
    stop = s.stop
    return [x for x in range(start,stop,step)]

def get_indices(values, listlike):
    """
    Returns the indices in litlike that match elements in values. This is used by some of the methods in message.py.
    """

    return [i for i, l  in zip(count(), listlike) if l in values]

def slice_length(orange):
    """
    Returns the length of the index corresponding to a slice.
    For example, slice(0,4,2) has a length of two.
    This is used by some of the methods in message.py.
    """
    t = type(orange)
    if t is slice:
        if orange.step:
            return int((orange.stop-orange.start)/orange.step) # This will also work for negative steps
        else: # Step size is 1 by default
            return orange.stop - orange.start
    else:
        return len(orange)

def subset_dict(dictionary, keys):
    """
    Returns a dict that contains all key,value pairs in dictionary where the key is one of the provided keys.
    This is used by some of the methods in message.py.    
    """
    keys = set(keys)
    return {key: value for key, value in dictionary.items() if key in keys}
