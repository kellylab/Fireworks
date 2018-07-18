

def index_to_list(index):
    """
    Converts an index to a list.
    """
    if type(index) is slice:
        index = slice_to_list(index)
    if type(index) is int:
        index = [index]
    return index

def slice_to_list(s):
    """
    Converts a slice object to a list of indices
    """
    step = s.step or 1
    start = s.start
    stop = s.stop
    return [x for x in range(start,stop,step)]

def get_indices(values, listlike):
    """
    Returns the indices in litlike that match elements in values
    """

    return [i for i, l  in zip(count(), listlike) if l in values]

def slice_length(orange):
    """
    Returns the length of the index corresponding to a slice.
    For example, slice(0,4,2) has a length of two.
    """
    t = type(orange)
    if t is slice:
        if orange.step:
            return int((orange.stop-orange.start)/orange.step) # This will also work for negative steps
        else: # Step size is 1 by default
            return orange.stop - orange.start
    else:
        return len(orange)
