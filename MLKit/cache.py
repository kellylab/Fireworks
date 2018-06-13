import cachetools

caches = {
    'LRU': cachetools.LRUCache,
}
class Cache:

    def __init__(self, max_size, cache_type = 'LRU'):
        self.max_size = max_size
        self.cache = caches[cache_type](max_size)

    def __getitem__(self, index):
        if type(index) is slice:
            step = slice.step or 1
            return Message([self.cache[i] for i in range(index.start, index.stop, step)])
        else:
            return Message(self.cache[index])

    def update(self, index, data):
        if type(index) is int:
            self.cache.

    def clear(self, n): pass
        """ Ensure there is enough free space for n elements. """
        free_space = self.max_size - self.size
        if n > free_space:
            self._clear(n-free_space)

    def _clear(self, n): pass

    @property
    def size(self):
        return len(self.cache)
