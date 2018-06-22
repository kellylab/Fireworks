from Fireworks.message import Message
from abc import abstractmethod
# caches = {
#     'LRU': cachetools.LRUCache,
# }

class MessageCache:

    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = Message()

    def __getitem__(self, index):
        if type(index) is int:
            index = slice(index, index+1)

        cache_indices = self.get_indices(index)

        return self.read_cache(cache_indices)

    @abstractmethod
    def __setitem__(self, index, message):
        if type(index) is int:
            pass
        if type(index) is slice:
            pass

    def __getattr__(self, *args, **kwargs):
        return self.cache.__getattr__(args, kwargs)

    def search(self, **kwargs):
        pass

    def free(self, n): 
        """ Ensure there is enough free space for n elements. """
        free_space = self.max_size - self.size
        if n > free_space:
            self._free(n-free_space)

    def _free(self, n): pass

    @property
    def size(self):
        return len(self.cache)
