

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
