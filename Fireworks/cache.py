from Fireworks.message import Message
from abc import ABC, abstractmethod
import pandas as pd
from itertools import count

# caches = {
#     'LRU': cachetools.LRUCache,
# }

class MessageCache(ABC):
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

        pointers = self.pointers[index] # This will raise an error if index not present
        # try:
        #     cache_indices = self.indices[index]
        # except KeyError as e:
        #     raise KeyError("Index {0} not in cache.".format(e))

        return self.cache[pointers.tolist()]

    @abstractmethod
    def __setitem__(self, index, message): pass

    def __getattr__(self, *args, **kwargs):
        return self.cache.__getattr__(args, kwargs)

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        return "MessageCache with indices {0}.".format(self.pointers.index.tolist())

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

class DummyCache(MessageCache):
    """
    This is a basic implementation of a MessageCache that simply appends new
    elements and never clears memory internally
    """

    def __setitem__(self, index, message):
        if type(index) is int:
            index = [index]
        if type(index) is slice:
            index = slice_to_list(index)
        present = set(index).intersection(set(self.pointers))
        not_present = set(index).difference(set(self.pointers))
        # Get indices in message
        present_indices = get_indices(index, present)
        not_present_indices = get_indices(index, not_present)
        self.add_new(not_present, message[not_present_indices])
        self.update_existing(present, message[present_indices])

    def add_new(self, index, message):
        """
        Adds new elements to cache and updates pointers
        """
        start = len(self.cache)
        assert False
        self.cache = self.cache.append(message)
        stop = len(self.cache)
        for i,j in zip(range(start, stop), index):
            self.pointers[j] = i

    def update_existing(self, index, message):
        """
        Updates elements already in the message
        """
        indices = self.pointers[index]
        for i,x in zip(indices, message):
            assert False
            self.cache[i] = x

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
    return [i for i,l  in zip(count(), listlike) if l in values]
