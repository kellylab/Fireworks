from abc import ABC, abstractmethod
from Bio import SeqIO
import pandas as pd
import Fireworks
from Fireworks.message import Message
from Fireworks.utils import index_to_list
from abc import ABC, abstractmethod

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

    # def reset_inputs(self):
    #
    #     new_inputs = []
    #     for source in self.input_sources:
    #         new_inputs.append(source.reset())
    #     self.input_sources = new_inputs
    #     self.position = 0

class CachingSource(Source):
    """
    Given input sources that implement __next__, will store all calls to __next__ into an internal cache and therafter allow __getitem__
    calls that either access from the cache or trigger __next__ calls to add to the cache.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position = 0
        self.length = None
        self.init_cache(*args, **kwargs)

    @abstractmethod
    def init_cache(self, *args, **kwargs):
        """
        This should initialize a cache object called self.cache
        """
        pass

    def __getitem__(self, index):

        index = index_to_list(index)
        self.step_forward(max(index))
        return self.cache[index]

    def __len__(self):
        """
        Length is computed implicitly and lazily. If any operation causes the source
        to reach the end of it's inputs, that position is stored as the length.
        Alternatively, if this method is called before that happens, the source will attempt to
        loop to the end and calculate the length.
        """
        if self.length is not None:
            self.compute_length()
        else:
            return self.length

    def compute_length(self):
        """
        Step forward as far as the inputs will allow and compute length.
        Note: If the inputs are infinite, then this will go on forever.
        """
        while True:
            try:
                self.step_forward(1)
            except StopIteration:
                self.length = self.position
                self.reset()
                break

    def step_forward(self, n):
        """
        Calls __next__ until self.position == n, adding elements to the cache at each step.
        """
        if self.length is not None and n < self.length:
            if self.position < n:
                for i in range(n-self.position+1):
                    try:
                        self.cache[i] = self.__next__()
                        self.position += 1
                    except StopIteration:
                        self.length = self.position
                        raise StopIteration
        else:
            raise ValueError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
