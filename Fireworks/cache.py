from Fireworks.message import Message
from abc import ABC, abstractmethod
import pandas as pd
from itertools import count
from bidict import bidict

# caches = {
#     'LRU': cachetools.LRUCache,
# }

# class RangedBidict(Bidict):
#     """
#     Bidict that supports access using a list of keys. For example, d[[2,3,5]] would return [d[2], d[3], d[5]]
#     """
#     pass

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
        # self.pointers = pd.Series({-1:-1}, dtype=int)
        self.pointers = bidict()
        # del self.pointers[-1]

    def __getitem__(self, index):

        if type(index) is slice:
            index = slice_to_list(index)

        if type(index) is int:
            pointers = self.pointers[index]
        else:
            pointers = [self.pointers[i] for i in index] # This will raise an error if index not present

        # try:
        #     cache_indices = self.indices[index]
        # except KeyError as e:
        #     raise KeyError("Index {0} not in cache.".format(e))

        return self.cache[pointers]#list(pointers.keys())]#.tolist()]

    @abstractmethod
    def __setitem__(self, index, message): pass

    def __getattr__(self, *args, **kwargs):
        return self.cache.__getattr__(args, kwargs)

    def __contains__(self, item):

        return item in self.pointers.keys()

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        return "MessageCache with indices {0}.".format(list(self.pointers.keys()))#.tolist())

    def _insert(self, index, message):
        """
        Inserts message into cache along with the desired indices.
        This method should be called by __setitem__ as needed to perform the insertion.
        """
        if type(index) is int:
            index = [index]
        if type(index) is slice:
            index = slice_to_list(index)
        present = set(index).intersection(set(self.pointers.keys()))
        not_present = set(index).difference(set(self.pointers.keys()))
        # Get indices in message
        present_indices = get_indices(index, present)
        not_present_indices = get_indices(index, not_present)
        self._add_new(not_present, message[not_present_indices])
        self._update_existing(present, message[present_indices])

    def _add_new(self, index, message):
        """
        Adds new elements to cache and updates pointers.
        This method should be called by __setitem__ or _insert as needed to perform insertions.
        """
        start = len(self.cache)
        self.cache = self.cache.append(message)
        stop = len(self.cache)
        for i,j in zip(range(start, stop), index):
            self.pointers[j] = i

    def _update_existing(self, index, message):
        """
        Updates elements already in the message.
        This method should be called by __setitem__ or _insert as needed to perform updates.
        """
        if index:
            if type(index) is int:
                indices = self.pointers[index]
            else:
                indices = [self.pointers[i] for i in index]
            self.cache[indices] = message #indices.tolist()

    def _delete(self, index):
        """
        Deletes elements in the message corresponding to index.
        This method should be called by __setitem__ or __delitem__ as needed.
        """
        # Get the index in the internal cache corresponding to the virtual (argument) index
        if type(index) is slice:
            index = slice_to_list(index)
        if type(index) is int:
            index = [index]

        index = [i for i in index if i in self.pointers]
        internal_indices = [self.pointers[i] for i in index]
        # Delete those elements
        del self.cache[internal_indices]
        # Delete the pointers for those elements
        for i in index:
            del self.pointers[i]
        f = pointer_adjustment_function(internal_indices)
        for key, internal_index in self.pointers.items():
            self.pointers[key] -= f(internal_index)

    def _permute(self, permutation, target = None):
        """
        Rearranges elements of cache based on permutation.
        Permutation should be a list of indices. Can optionally specify target, which is the indices to apply the permutation to.
        By default, the entire index will be treated as the target.
        """
        if target is None:
            target = [self.pointers.inv[i] for i in range(len(self.cache))]

        pointers = bidict({t:self.pointers[self.pointers.inv[p]] for t,p in zip(target, permutation)}) # Permute which internal index a pointer points to
        index = [self.pointers[t] for t in target]
        self.cache[index] = self.cache[permutation]  # Permute internal indices
        self.pointers = pointers

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
        """
        Simply inserts the message with the corresponding index without ever freeing memory.
        """
        self._insert(index, message)

    def __delitem__(self, index):
        self._delete(index)

def pointer_adjustment_function(index):
    """
    Given an index, returns a function that that takes an integer as input and returns how many elements of the index the number is greater than.
    This is used for readjusting pointers after a deletion. For example, if you delete index 2, then every index greater than 2 must slide down 1
    but index 0 and 1 do not more.
    """
    if not type(index) is list:
        if type(index) is slice:
            index = slice_to_list(index)
        if type(index) is int:
            index = [index]

    index = sorted(index)
    def adjustment_function(x):
        for i,c in zip(index, count()):
            if x < i:
                return c
        return c+1 # If x > every index

    return adjustment_function

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
