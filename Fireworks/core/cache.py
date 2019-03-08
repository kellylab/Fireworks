from .message import Message
from abc import ABC, abstractmethod
import pandas as pd
from itertools import count
from bidict import bidict
import numpy as np

class MessageCache(ABC):
    """
    A message cache stores parts of a larger method and supports retrievals and
    insertions based on index.
    The use case for a MessageCache is for storing parts of a large dataset in memory.
    The MessageCache can keep track of which elements and which indices are present
    in memory at a given time and allow for updates and retrievals.
    """

    def __init__(self, max_size):
        """
        Args:
            max_size (int): The maximum size of this cache.
        """
        self.max_size = max_size
        self.cache = Message()
        self.pointers = bidict()

    def __getitem__(self, index): # TODO: Optimize for range queries.
        """
        Translates the given index into the appropriate index for the internal message and returns those values.
        """
        if type(index) is slice:
            index = slice_to_list(index)

        if type(index) is int:
            pointers = self.pointers[index]
        else:
            pointers = [self.pointers[i] for i in index] # This will raise an error if index not present

        return self.cache[pointers] #list(pointers.keys())]#.tolist()]

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

    def insert(self, index, message):
        """
        Inserts message into cache along with the desired indices.
        This method should be called by __setitem__ as needed to perform the insertion.

        Args:
            index: The index to insert into. Can be an int, slice, or list of integer indices.
            message: The Message to insert. Should have the same length as the provided idnex.
        """
        message = Message(message)

        if type(index) is int:
            index = [index]
        if type(index) is slice:
            index = slice_to_list(index)
        present = [i for i in index if i in self.pointers.keys()] # Intersection
        not_present = [i for i in index if i not in self.pointers.keys()] # Difference
        # Get indices in message
        present_indices = get_indices(index, present)
        not_present_indices = get_indices(index, not_present)

        self._add_new(not_present, message[not_present_indices])
        self._update_existing(present, message[present_indices])

    def _add_new(self, index, message):
        """
        Adds new elements to cache and updates pointers.
        This method should be called by __setitem__ or insert as needed to perform insertions.

        Args:
            index: The index to insert into. Can be an int, slice, or list of integer indices.
            message: The Message to insert. Should have the same length as the provided idnex.

        """
        start = len(self.cache)
        self.cache = self.cache.append(message)
        stop = len(self.cache)
        for i,j in zip(range(start, stop), index):
            self.pointers[j] = i

    def _update_existing(self, index, message):
        """
        Updates elements already in the message.
        This method should be called by __setitem__ or insert as needed to perform updates.

        Args:
            index: The index to insert into. Can be an int, slice, or list of integer indices.
            message: The Message to insert. Should have the same length as the provided idnex.

        """
        if index:
            if type(index) is int:
                indices = self.pointers[index]
            else:
                indices = [self.pointers[i] for i in index]
            self.cache[indices] = message #indices.tolist()

    def delete(self, index):
        """
        Deletes elements in the message corresponding to index.
        This method should be called by __setitem__ or __delitem__ as needed.

        Args:
            index: The index to insert into. Can be an int, slice, or list of integer indices.

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
        self.pointers = dict(self.pointers) # Convert to normal dict to prevent duplication errors
        for key, value in self.pointers.items():
            self.pointers[key] -= f(value)
        self.pointers = bidict(self.pointers)

    def _permute(self, permutation, target = None):
        """
        Rearranges elements of cache based on permutation.
        Permutation should be a list of indices. Can optionally specify target, which is the indices to apply the permutation to.
        By default, the entire index will be treated as the target.

        Args:
            permutation: A list of indices
        """
        if target is None:
            target = [self.pointers.inv[i] for i in range(len(self.cache))]

        pointers = bidict({t:self.pointers[self.pointers.inv[p]] for t,p in zip(target, permutation)}) # Permute which internal index a pointer points to
        index = [self.pointers[t] for t in target]
        self.cache[index] = self.cache[permutation]  # Permute internal indices
        self.pointers = pointers

    def sort(self):
        """
        Rearranges internal cache indices to be in sorted order.
        """
        # Identify sort permutation
        # Apply sort permutation
        pass

    def search(self, **kwargs):
        pass

    @property
    def size(self):
        return len(self.cache)

class UnlimitedCache(MessageCache):
    """
    This is a basic implementation of a MessageCache that simply appends new
    elements and never clears memory internally
    """

    def __init__(self):
        super().__init__(None) # No max size

    def __setitem__(self, index, message):
        """
        Simply inserts the message with the corresponding index without ever freeing memory.

        Args:
            index: The index to insert into. Can be an int, slice, or list of integer indices.
            message: The Message to insert. Should have the same length as the provided idnex.

        """
        self.insert(index, message)

    def __delitem__(self, index):
        """
        Args:
            index: The index to insert into. Can be an int, slice, or list of integer indices.

        """
        self.delete(index)

class BufferedCache(MessageCache):
    """
    This implements a setitem method that assumes that when the cache is full, elements must be deleted until it is max_size - buffer_size
    in length. The deletion method, _free, must be implemented by a subclass.
    """


    def init_buffer(self, buffer_size = 0):

        self.buffer_size = buffer_size # Specifies how far above max_size the cache can go before having to delete elements

    def __setitem__(self, index, message):
        """

        Args:
            index: The index to insert into. Can be an int, slice, or list of integer indices.
            message: The Message to insert. Should have the same length as the provided idnex.

        """
        index = index_to_list(index)
        message = Message(message)
        if len(index) != len(message):
            raise ValueError("Message length does not match length of index for insertion.")
        # Determine how much space to free in order to insert message
        free_space = self.max_size - len(self)

        if len(message) > free_space:
            how_much = max(self.buffer_size - free_space, len(message) - free_space)

            self.free(how_much)

        self.insert(index, message)

    def __delitem__(self, index):
        """
        Args:
            index: The index to insert into. Can be an int, slice, or list of integer indices.
            message: The Message to insert. Should have the same length as the provided idnex.

        """
        self.delete(index)

    @abstractmethod
    def free(self, n): pass

class RRCache(BufferedCache):

    def free(self, n):
        indices_present = self.cache.keys()
        delete_indices = list(np.random.choice(indices_present, n, replace=False))
        self.__delete__(delete_indices)

class RankingCache(MessageCache):
    """
    Implements a free method that deletes elements based on a ranking function.
    """

    def init_rank_dict(self):
        self.rank_dict = {}

    def free(self, n, x=0):

        to_sort = list(self.rank_dict.keys())
        sort_by = list(self.rank_dict.values())
        sorter = lambda k: self.rank_dict[k]
        ranks = sorted(to_sort, key=sorter)
        del_indices = ranks[0:n]

        self.__delitem__(del_indices)

    def __delitem__(self, index):

        self.delete(index)
        # Remove index components from rank_dict
        index = index_to_list(index)
        for i in index:
            del self.rank_dict[i]
        self.on_delete(index)

    def on_update_existing(self, index, message): pass

    def on_add_new(self, index, message): pass

    def on_delete(self, index): pass

    def on_getitem(self, index): pass

    def __getitem__(self, index):

        message = super().__getitem__(index)
        self.on_getitem(index)
        return message

    def _update_existing(self, index, message):
        super()._update_existing(index, message)
        self.on_update_existing(index, message)

    def _add_new(self, index, message):
        super()._add_new(index, message)
        self.on_add_new(index, message)

class LRUCache(RankingCache, BufferedCache):
    """
    Implements a Least Recently Used cache. Items are deleted in descending order of how recently they were accessed.
    A call to __getitem__ or __setitem__ counts as accessing an element.
    """

    def __init__(self, *args, buffer_size = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_rank_dict()
        self.init_buffer(buffer_size)
        self.counter = 0

    def update_rank(self, index):
        index = index_to_list(index)
        for i in index:
            self.rank_dict[i] = self.counter
        self.counter += 1

    def on_update_existing(self, index, message):
        self.update_rank(index)

    def on_add_new(self, index, message):
        self.update_rank(index)

    def on_delete(self, index): pass

    def on_getitem(self, index):
        self.update_rank(index)

class LFUCache(RankingCache, BufferedCache):
    """
    Implements a Least Frequently Used cache. Items are deleted in increasing order of how frequently they are accessed.
    A call to __getitem__ or __setitem__ counts as accessing an element.
    """
    def __init__(self, *args, buffer_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_rank_dict()
        self.init_buffer(buffer_size)

    def update_rank(self, index):
        index = index_to_list(index)
        for i in index:
            if i in self.pointers:
                self.rank_dict[i] += 1

    def on_update_existing(self, index, message):
        self.update_rank(index)

    def on_add_new(self, index, message):
        index = index_to_list(index)
        for i in index:
            self.rank_dict[i] = 0

    def on_delete(self, index):
        self.update_rank(index)

    def on_getitem(self, index):
        self.update_rank(index)

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
