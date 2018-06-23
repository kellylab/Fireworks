from Fireworks.message import Message
from abc import abstractmethod
import pandas as pd
from itertools import count

# caches = {
#     'LRU': cachetools.LRUCache,
# }

class MessageCache:
    """
    A message cache stores parts of a larger method and supports retrievals and
    insertions based on index.
    The use case for a MessageCache is for storing parts of a large dataset in memory.
    The MessageCache can keep track of which elmeents and which indices are present
    in memory at a given time and allow for updates and retrievals.
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = Message()
        # Initialize a list of pointers to keep track of internal indices
        self.pointers = pd.Series({-1:-1}, dtype=int)
        del self.pointers[-1]

    def __getitem__(self, index):

        if type(index) is slice:
            index = slice_to_list(index)

        cache_indices = self.indices[index] # This will raise an error if index not present
        # try:
        #     cache_indices = self.indices[index]
        # except KeyError as e:
        #     raise KeyError("Index {0} not in cache.".format(e))

        return self.cache[cache_indices]

    @abstractmethod
    def __setitem__(self, index, message): pass

    def __getattr__(self, *args, **kwargs):
        return self.cache.__getattr__(args, kwargs)

    def search(self, **kwargs):
        pass

    # def free(self, n):
    #     """ Ensure there is enough free space for n elements. """
    #     free_space = self.max_size - self.size
    #     if n > free_space:
    #         self._free(n-free_space)
    #
    # def _free(self, n): pass

    @property
    def size(self):
        return len(self.cache)

class DummyCache:
    """
    This is a basic implementation of a MessageCache that simply appends new
    elements and never clears memory internally
    """

    def __setitem__(self, index, message):
        if type(index) is int:
            index = [index]
        if type(index) is slice:
            index = slice_to_list(index)
        present = index.intersection(self.pointers)
        not_present = index.difference(self.pointers)
        # Get indices in message
        present_indices = get_indices(index, present)
        not_present_indices = get_indices(index, not_present)
        self.add_new(present, message[present_indices])
        self.update_existing(not_present, message[not_present_indices])

    def add_new(self, index, message):
        """
        Adds new elements to cache and updates pointers
        """
        start = len(self.cache)
        self.cache.append(message)
        stop = len(self.cache)
        self.pointers[index] = [i for i in range(start,stop)]

    def update_exsiting(self, index, message):
        """
        Updates elements already in the message
        """
        indices = self.pointers[index]
        self.cache[indices] = data 

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
    return [i if l in values for i,l  in zip(count, listlike)]
