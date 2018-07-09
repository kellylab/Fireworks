from abc import ABC, abstractmethod
from Bio import SeqIO
import pandas as pd
import Fireworks
from Fireworks.message import Message
from Fireworks.utils import index_to_list
from Fireworks.cache import LRUCache, LFUCache
from abc import ABC, abstractmethod
from itertools import count

class Source(ABC):
    """
    The core data structure in fireworks.
    A source can take sources as inputs, and its outputs can be piped to other sources.
    All communication is done via Message objects.
    Method calls are deferred to input sources recursively until a source that implements the method is reached.
    """

    name = 'base_source'

    def __init__(self, *args, inputs = None, **kwargs):
        self.input_sources = inputs

    def __getattr__(self, *args, **kwargs):

        if self.input_sources is None:
            raise AttributeError("Source {0} does not have attribute {1}.".format(self.name, str(args)))

        responses = [source.__getattribute__(args[0])(**kwargs) for source in self.input_sources.values()]
        return Fireworks.merge(responses)

class DataSource(Source):
    """ Class for representing a data source. It formats and reads data, and is able to convert batches into tensors. """

    name = 'DataSource'

    # @abstractmethod
    def to_tensor(self, batch: Message, embedding_function: dict = None):
        """
        Converts a batch (stored as dictionary) to a dictionary of tensors. embedding_function is a dict that specifies optional
        functions that construct embeddings and are called on the element of the given key.
        """
        # TODO: If no embedding_function is provided, or if a key maps to None, attempt to automatically convert the batch to tensors.
        pass

    # def __next__(self):
    #     return {key: next(souce) for key, source in self.inputs.values()}
    #
    # def __getitem__(self, index):
    #     return {key: _input.__getitem__(index) for key, _input in self.inputs.values()}

    def __iter__(self):
        return self

class BioSeqSource(DataSource):
    """ Class for representing biosequence data. """

    name = 'BioSeqSource'

    def __init__(self, path, filetype = 'fasta', **kwargs):
        self.path = path
        self.filetype = filetype
        self.kwargs = kwargs
        self.seq = SeqIO.parse(self.path, self.filetype, **self.kwargs)

    def reset(self):
        self.seq = SeqIO.parse(self.path, self.filetype, **self.kwargs)
        return self

    def to_tensor(self, batch: Message, embedding_function: dict):

        metadata = {
        'rawsequences': batch['sequences'],
        'names': batch['names'],
        'ids': batch['ids'],
        'descriptions': batch['descriptions'],
        'dbxrefs': batch['dbxrefs'],
            }

        tensor_dict = {
        'sequences': embedding_function['sequences'](batch['sequences']),
        }

        return Message(tensor_dict, metadata)

    def __next__(self):

        gene = self.seq.__next__()

        return pd.DataFrame({
            'sequences': [str(gene.seq)],
            'ids': [gene.id],
            'names': [gene.name],
            'descriptions': [gene.description],
            'dbxrefs': [gene.dbxrefs],
        })

    def __iter__(self):
        return self

class LoopingSource(Source):
    """
    Given input sources that implement __next__ and reset (to be repeatable),
    will simulate __getitem__ by repeatedly looping through the iterator as needed.
    """

    name = 'LoopingSource'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_inputs()
        self.reset()
        self.length = None


    def __getitem__(self, index):
        """
        Retreives items in index by looping through inputs as many times as needed.
        """

        # TODO: Check if index exceeds length, either explicitly or implicitly.

        # Sort index
        index = sorted(index_to_list(index))
        above = [i for i in index if i >= self.position] # Go forward to reach these
        below = [i for i in index if i < self.position] # Will have to reset the loop to reach these
        if len(above) > 0:
            above_values = Fireworks.cat([self.step_forward(i+1) for i in above])
        else:
            above_values = Message()
        if len(below) > 0:
            self.reset() # Position will now be reset to 0
            below_values = Fireworks.cat([self.step_forward(i+1) for i in below])
        else:
            below_values = Message()
        return below_values.append(above_values) # TODO: Resort this message so values are in the order requested by index

    def __len__(self):
        if self.length is not None:
            return self.length
        else:
            self.compute_length()
            return self.length

    def __next__(self):
        """
        Go forward one step. This can be used to loop over the inputs while automatically recording
        length once one cycle has been performed.
        """
        if self.length is None:
            return self.step_forward(1)
        else:
            p = (self.position + 1) % self.length
            self.position = p
            return self[p]

    def check_inputs(self):
        """
        Checks inputs to determine if they implement __next__ and reset methods.
        """
        for name, source in self.input_sources.items():
            if not (hasattr(source, '__next__') and hasattr(source, 'reset')):
                raise TypeError('Source {0} does not have __next__ and reset methods.'.format(name))

    def reset(self):

        for source in self.input_sources.values():
            source.reset()
        self.position = 0

    def compute_length(self):
        """
        Step forward as far as the inputs will allow and compute lenght.
        Note: If the inputs are infinite, then this will go on forever.
        """
        while True:
            try:
                self.step_forward(self.position+1)
            except StopIteration:
                self.reset()
                break

    def step_forward(self, n):
        """
        Steps forward through inputs until position = n and then returns that value.
        """
        if self.length is not None and n > self.length:
            raise ValueError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
        if n < self.position:
            raise ValueError("Can only step forward to a value higher than current position.")
        x = Message()
        for _ in range(n - self.position):
            try:
                # x = x.append(Fireworks.merge([source.__next__() for source in self.input_sources.values()]))
                x = Fireworks.merge([source.__next__() for source in self.input_sources.values()])
                self.position += 1
            except StopIteration:
                self.length = self.position
                raise StopIteration # raise ValueError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
        return x

class CachingSource(Source):
    """
    Given input sources that implement __getitem__, will store all calls to __getitem__ into an internal cache and therafter __getitem__
    calls will either access from the cache or trigger __getitem__ calls on the input and an update to the cache.
    """
    def __init__(self, *args, cache_size = 100, cache_type = 'LRU', infinite = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_inputs()
        self.length = None
        self.lower_bound = 0
        self.upper_bound = None
        self.infinite = infinite
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.init_cache(*args, **kwargs)

    # @abstractmethod # TODO: Make different types of caches implementable via subclasses
    def init_cache(self, *args, **kwargs):
        """
        This should initialize a cache object called self.cache
        """
        choices = {'LRU': LRUCache, 'LFU': LFUCache}
        self.cache = choices[self.cache_type](max_size = self.cache_size,)

    def check_inputs(self):
        """
        Checks inputs to determine if they implement __getitem__.
        """
        for name, source in self.input_sources.items():
            if not (hasattr(source, '__getitem__')):
                raise TypeError('Source {0} does not have __getitem__ method.'.format(name))

    def __getitem__(self, index):

        if self.length and index: # Implicit length check if length is known
            if max(index) >= self.length:
                raise ValueError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
        index = index_to_list(index)
        # Identify what is in the cache and what isn't.
        in_cache = [i for i in index if i in self.cache.pointers] # Elements of index corresponding to in_cache elements
        in_cache_indices = [j for i,j in zip(index, count()) if i in self.cache.pointers] # Indices in index corresponding to in_cache elements
        not_in_cache = [i for i in index if i not in self.cache.pointers] # Elements of index corresponding to not_in_cache elements
        not_in_cache_indices = [j for i,j in zip(index, count()) if i not in self.cache.pointers] # Indices in index corresponding to not_in_cache elements
        # Retrieve from cache existing elements
        in_cache_elements = self.cache[in_cache] # elements in cache corresponding to indices in cache
        # Update cache to have other elements
        not_in_cache_elements = Fireworks.merge([source[not_in_cache] for source in self.input_sources.values()])
        self.cache[not_in_cache] = not_in_cache_elements
        # Reorder and merge requested elements
        message = in_cache_elements.append(not_in_cache_elements)
        indices = in_cache_indices
        indices.extend(not_in_cache_indices)
        # Elements must be reordered based on their order in index
        permutation = indices.copy()
        for i,j in zip(indices, count()):
            permutation[i] = j
        # permutation = in_cache_indices.extend(not_in_cache_indices)
        message = message.permute(permutation)

        # Implicit update of internal knowledge of length
        if index and self.length is None and not self.infinite:
            l = max(index)
            if l > self.lower_bound:
                self.lower_bound = l

        return message

    def __len__(self):
        """
        Length is computed implicitly and lazily. If any operation causes the source
        to reach the end of it's inputs, that position is stored as the length.
        Alternatively, if this method is called before that happens, the source will attempt to
        loop to the end and calculate the length.
        """
        if self.infinite:
            raise ValueError("Source is labelled as having infinite length (ie. yields items indefinitely).")
        elif self.length is None:
            self.compute_length()
            return self.length
        else:
            return self.length

    def compute_length(self):
        """
        Step forward as far as the inputs will allow and compute length.
        Note: If the inputs are infinite, then this will go on forever.
        """
        if not self.infinite:
            while True:
                try:
                    self[1 + self.lower_bound] # Use knowledge of lower bound from previous getitem calls to accelerate this process
                except ValueError:
                    self.length = 1 + self.lower_bound
                    break
        else:
            raise ValueError("Source is labelled as having infinite length (ie. yields items indefinitely).")

class ShufflerSource(Source):
    """
    Given input sources that implement __getitem__ and __len__, will shuffle the indices so that iterating through
    the source or calling __getitem__ will return different values.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_inputs()

    def check_inputs(self):
        """
        Check inputs to see if they implement __getitem__ and __len__
        """
        for name, source in self.input_sources.items():
            if not (hasattr(source, '__getitem__')):
                raise TypeError('Source {0} does not have __getitem__ method.'.format(name))
            if not(hasattr(source, '__len__')):
                raise TypeError('Source {0} does not have __len__ method.'.format(name))

    def __getitem__(self, index): pass

    def shuffle(self, order = None): pass

    def reset(self):
        """
        Triggers a shuffle on reset.
        """
        pass

class
