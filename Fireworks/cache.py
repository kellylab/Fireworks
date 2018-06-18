from Fireworks.message import Message

# caches = {
#     'LRU': cachetools.LRUCache,
# }

class Cache:

    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = Message()

    def __getitem__(self, index):
        if type(index) is int:
            pass

        if type(index) is slice:
            pass

    def search(self, **kwargs):
        pass 

    def insert(self, data, index):
        if type(index) is int:
            pass
        if type(index) is slice:
            pass

    def free(self, n): pass
        """ Ensure there is enough free space for n elements. """
        free_space = self.max_size - self.size
        if n > free_space:
            self._free(n-free_space)

    def _free(self, n): pass

    @property
    def size(self):
        return len(self.cache)
