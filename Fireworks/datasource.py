from abc import ABC, abstractmethod
from Bio import SeqIO
import pandas as pd
import Fireworks
from Fireworks.message import Message
from Fireworks.utils import index_to_list
from Fireworks.cache import LRUCache, LFUCache, UnlimitedCache
from Fireworks.preprocessing import one_hot
from abc import ABC, abstractmethod
from itertools import count
import types
import random
from bidict import bidict
import torch
import math

class Source(ABC):
    """
    The core data structure in fireworks.
    A source can take sources as inputs, and its outputs can be piped to other sources.
    All communication is done via Message objects.
    Method calls are deferred to input sources recursively until a source that implements the method is reached.
    """

    name = 'base_source'

    def __init__(self, *args, inputs = None, **kwargs):

        if type(inputs) is dict:
            self.input_sources = inputs
        elif isinstance(inputs, Source): # Can give just one source as input without having to type out an entire dict
            self.input_sources = {'data': inputs}
        elif inputs is None: # Subclasses can have their own method for creating an inputs_dict and just leave this argument blank
            self.input_sources = {}
        else:
            raise TypeError("inputs must be a dict of sources or a single source")

    def recursive_call(self, attribute, *args, ignore_first = True, **kwargs):
        """
        Recursively calls method/attribute on input_sources until reaching an upstream source that implements the method and
        returns the response as a message (empty if response is None).
        """

        if not ignore_first:
            if hasattr(self, attribute):
                if args or kwargs: # Is a method call
                    return self.__getattribute__(attribute)(*args,**kwargs)

                else: # Is an attribute
                    try:
                        return self.__getattribute__(attribute)
                    except AttributeError:
                        return self.__getattr__(attribute)

        if not self.input_sources:
            raise AttributeError("Source {0} does not have method/attribute {1}.".format(self.name, str(attribute)))

        responses = [source.recursive_call(attribute, *args, ignore_first=False, **kwargs) for source in self.input_sources.values()]

        if responses:
            if isinstance(responses[0], Source):
                return Fireworks.merge(responses)
            elif len(responses) == 1:
                return responses[0]
            else:
                return {key: response for key, respone in zip(self.input_sources.keys(), responses)}

    def check_inputs(self): pass

    # def __getattr__(self, *args, **kwargs):
    #
    #     if len(args) > 1:
    #         positional_args = args[1:]
    #     else:
    #         positional_args = []
    #
    #     return self.recursive_call(args[0], *positional_args, **kwargs)


# class DataSource(Source):
#     """ Class for representing a data source. It formats and reads data, and is able to convert batches into tensors. """
#
#     name = 'DataSource'
#
#     # @abstractmethod
#     def to_tensor(self, batch: Message, embedding_function: dict = None):
#         """
#         Converts a batch (stored as dictionary) to a dictionary of tensors. embedding_function is a dict that specifies optional
#         functions that construct embeddings and are called on the element of the given key.
#         """
#         # TODO: If no embedding_function is provided, or if a key maps to None, attempt to automatically convert the batch to tensors.
#         pass
#
#     # def __next__(self):
#     #     return {key: next(souce) for key, source in self.inputs.values()}
#     #
#     # def __getitem__(self, index):
#     #     return {key: _input.__getitem__(index) for key, _input in self.inputs.values()}
#
#     def __iter__(self):
#         return self

class BioSeqSource(Source):
    """ Class for representing biosequence data. """

    name = 'BioSeqSource'

    def __init__(self, path, inputs = None, filetype = 'fasta', **kwargs):
        self.path = path
        self.filetype = filetype
        self.kwargs = kwargs
        self.seq = SeqIO.parse(self.path, self.filetype, **self.kwargs)
        self.input_sources = {}

    def reset(self):
        self.seq = SeqIO.parse(self.path, self.filetype, **self.kwargs)
        return self

    # def to_tensor(self, batch: Message, embedding_function: dict):
    #
    #     metadata = {
    #     'rawsequences': batch['sequences'],
    #     'names': batch['names'],
    #     'ids': batch['ids'],
    #     'descriptions': batch['descriptions'],
    #     'dbxrefs': batch['dbxrefs'],
    #         }
    #
    #     tensor_dict = {
    #     'sequences': embedding_function['sequences'](batch['sequences']),
    #     }
    #
    #     return Message(tensor_dict, metadata)

    def __next__(self):

        gene = self.seq.__next__()

        try:
            return Message({
                'sequences': [str(gene.seq)],
                'ids': [gene.id],
                'names': [gene.name],
                'descriptions': [gene.description],
                'dbxrefs': [gene.dbxrefs],
            })
        except StopIteration:
            raise StopIteration

    def __iter__(self):
        return self.reset()

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

    def __next__(self): # QUESTION: Should __next__ calls be supported for this source?
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
            except (StopIteration, IndexError):
                self.reset()
                break

    def step_forward(self, n):
        """
        Steps forward through inputs until position = n and then returns that value.
        """
        if self.length is not None and n > self.length:
            raise IndexError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
        if n < self.position:
            raise IndexError("Can only step forward to a value higher than current position.")
        x = Message()
        for _ in range(n - self.position):
            try:
                # x = x.append(Fireworks.merge([source.__next__() for source in self.input_sources.values()]))
                x = Fireworks.merge([source.__next__() for source in self.input_sources.values()])
                self.position += 1
            except StopIteration:
                self.length = self.position
                raise IndexError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
        return x

class CachingSource(Source):
    """
    Given input sources that implement __getitem__, will store all calls to __getitem__ into an internal cache and therafter __getitem__
    calls will either access from the cache or trigger __getitem__ calls on the input and an update to the cache.
    """
    def __init__(self, *args, cache_size = 100, buffer_size = 0, cache_type = 'LRU', infinite = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_inputs()
        self.length = None
        self.lower_bound = 0
        self.upper_bound = None
        self.infinite = infinite
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.buffer_size = buffer_size
        self.init_cache(*args, **kwargs)

    # @abstractmethod # TODO: Make different types of caches implementable via subclasses
    def init_cache(self, *args, **kwargs):
        """
        This should initialize a cache object called self.cache
        """
        choices = {'LRU': LRUCache, 'LFU': LFUCache}
        self.cache = choices[self.cache_type](max_size = self.cache_size, buffer_size=self.buffer_size)

    def check_inputs(self):
        """
        Checks inputs to determine if they implement __getitem__.
        """
        for name, source in self.input_sources.items():
            if not (hasattr(source, '__getitem__')):
                raise TypeError('Source {0} does not have __getitem__ method.'.format(name))

    def __getitem__(self, index):

        index = index_to_list(index)
        if self.length and len(index): # Implicit length check if length is known
            if max(index) >= self.length:
                raise IndexError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
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
        if len(index) and self.length is None and not self.infinite:
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
                except IndexError:
                    self.length = 1 + self.lower_bound
                    break
        else:
            raise ValueError("Source is labelled as having infinite length (ie. yields items indefinitely).")

class PassThroughSource(Source):
    """
    This source passes through data access calls and methods to its (single) input source except for whatever is overridden by subclasses.
    NOTE: Only the special methods explicitly defined here (getitem, len, delitem, setitem, next, iter) are passed through.
    Non-special methods are passed through normally.
    """

    name = 'Passthrough source'
    def __init__(self, *args, labels_column = 'labels', **kwargs):

        super().__init__(*args, **kwargs)
        self.check_inputs()
        for title, source in self.input_sources.items(): # There is only one
            self.input_title = title
            self.input_source = source

    def check_inputs(self):
        if len(self.input_sources) > 1:
            raise ValueError("A pass-through source can only have one input source.")

    def __getitem__(self, *args, **kwargs):
        return self.input_source.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.input_source.__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        return self.input_source.__delitem__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.input_source.__len__(*args, **kwargs)

    def __next__(self, *args, **kwargs):
        return self.input_source.__next__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return self.input_source.__iter__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        """
        Pass through all methods of the input source while adding labels. This does not intercept special methods (__x__ methods)
        """
        return self.recursive_call(*args, **kwargs) #self.input_source.__getattribute__(*args, **kwargs)

class HookedPassThroughSource(PassThroughSource): # BUG NOTE: Methods that return self will break the passthrough at the moment
    """
    This source is the same as PassThroughSource, but it has hooks which can be implemented by subclasses to modify the behavior of
    passed through calls.
    """

    name = 'Hooked-passthrough source'

    def _getitem_hook(self, message): return message

    # def _setitem_hook(self, *args, **kwargs): pass
    #
    # def _delitem_hook(self, *args, **kwargs): pass
    #
    # def _len_hook(self, *args, **kwargs): return args[0]

    def _next_hook(self, message): return message

    # def _iter_hook(self, *args, **kwargs): return args[0]

    def __getitem__(self, *args, **kwargs):

        return self._getitem_hook(self.input_source.__getitem__(*args, **kwargs)) #self.input_source.__getitem__(*args, **kwargs))

    # def __setitem__(self, *args, **kwargs):
    #     self._setitem_hook(self.input_source.__setitem__(*args, **kwargs))
    #
    # def __delitem__(self, *args, **kwargs):
    #     self._delitem_hook(self.input_source.__delitem__(*args, **kwargs))
    #
    # def __len__(self, *args, **kwargs):
    #     return self._len_hook(self.input_source.__len__(*args, **kwargs))

    def __next__(self, *args, **kwargs):
        return self._next_hook(self.input_source.__next__(*args, **kwargs))

    def __iter__(self, *args, **kwargs):

        self.input_source = self.input_source.__iter__(*args, **kwargs)
        return self

    # def __getattr__(self, *args, **kwargs):
    #     """
    #     Pass through all methods of the input source while adding labels. This does not intercept special methods (__x__ methods)
    #     """
    #     return self.input_source.__getattribute__(*args, **kwargs)

class Title2LabelSource(Source):
    """
    This source takes one source as input and inserts a column called 'label' to all outputs where the label is
    the name of the input source.
    """

    def __init__(self, *args, labels_column = 'labels', **kwargs):

        super().__init__(*args, **kwargs)
        self.labels_column = labels_column
        self.check_inputs()

    def check_inputs(self):
        if len(self.input_sources) > 1:
            raise ValueError("A label source can only have one input source.")
        for label, source in self.input_sources.items(): # There is only one
            self.label = label
            self.input_source = source

    def __getattr__(self, *args, **kwargs):
        """
        Pass through all methods of the input source while adding labels.
        """
        output = self.input_source.__getattribute__(*args, **kwargs)
        if type(output) is types.MethodType: # Wrap the method in a converter
            return self.method_wrapper(output)
        else:
            return self.attribute_wrapper(output)

    def method_wrapper(self, function):
        """
        Wraps method with a label attacher such that whenever the method is called, the output is modified
        by adding the label.
        """

        def new_function(*args, **kwargs):

            output = function(*args, **kwargs)
            try:
                output = Message(output)
            except:
                return output
            return self.insert_labels(output)

        return new_function

    def attribute_wrapper(self, attribute):
        """
        Wraps attribute with new label if attribute returns a message.
        """
        try:
            output = Message(attribute)
        except:
            return attribute
        return self.insert_labels(output)

    def insert_labels(self, message):

        l = len(message)
        message[self.labels_column] = [self.label for _ in range(l)]

        return message

class LabelerSource(PassThroughSource):
    """
    This source implements a to_tensor function that converts labels contained in messages to tensors based on an internal labels dict.
    """

    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.labels_dict = bidict({label:i for label,i in zip(labels,count())})

    def to_tensor(self, batch, labels_column = 'labels'):

        batch['labels_name'] = batch[labels_column]
        labels = batch[labels_column]
        labels = [self.labels_dict[label] for label in labels]
        labels = torch.Tensor([one_hot(label,len(self.labels_dict)) for label in labels])
        batch['labels'] = labels
        try: # Allow an upstream source to attempt to perform to_tensor if it can
            batch = self.recursive_call('to_tensor', batch)
            return batch
        except AttributeError:
            return batch

class AggregatorSource(Source):
    """
    This source takes multiple sources implementing __next__ as input and implements a new __next__ method that samples
    its input sources.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_inputs()

    def check_inputs(self):

        for name, source in self.input_sources.items():
            if not (hasattr(source, '__next__') and hasattr(source, 'reset')):
                raise ValueError("Input sources must implement __next__ and reset.")

    def __next__(self):
        """
        Call __next__ on one of the input sources based on sampling algorithm until all sources run out (can run indefinetely if one
        or more inputs are infinite.)
        """
        # Choose which input to sample
        sample = self.sample_inputs()
        # Return value
        try:
            return self.input_sources[sample].__next__()
        except StopIteration: # Remove sample from available_inputs list
            self.available_inputs.remove(sample)
            if not self.available_inputs: # Set of inputs is empty, because they have all finished iterating
                raise StopIteration
            else: # Recursively try again
                return self.__next__()

    def reset(self):
        for name, source in self.input_sources.items():
            source.reset()
        self.available_inputs = set(self.input_sources.keys()) # Keep track of which sources have not yet run out.

    def __iter__(self):
        self.reset()
        return self

    @abstractmethod
    def sample_inputs(self):
        """
        Returns the key associated with an input source that should be stepped through next.
        """
        pass

class RandomAggregatorSource(AggregatorSource):
    """
    AggregatorSource that randomly chooses inputs to step through.
    """
    # TODO: Add support for weighted random sampling

    def sample_inputs(self):
        return random.sample(self.available_inputs, 1)[0]

class ClockworkAggregatorSource(AggregatorSource):
    """
    AggregatorSource that iterates through input sources one at a time.
    """
    # TODO: Add support for weighted iteration and setting order (ie. spend two cycles on one input and one on another)

    def reset(self):
        super().reset()
        self.cycle_dict = {i: name for i,name in zip(count(),self.available_inputs)}
        self.current_cycle = 0

    def sample_inputs(self):
        """
        Loops through inputs until finding one that is available.
        """
        # if not self.available_inputs:
        #     raise StopIteration
        while True:
            sample = self.cycle_dict[self.current_cycle]
            self.current_cycle = (self.current_cycle + 1) % len(self.cycle_dict)
            if sample in self.available_inputs:
                return sample

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

class IndexMapperSource(Source):
    """
    Given input sources that implement __getitem__, returns a source that maps indices in input_indices to output_indices via __getitem__
    """
    def __init__(self, input_indices, output_indices, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.check_inputs()
        if len(input_indices) != len(output_indices):
            raise ValueError("The number of input indices does not match the number of output indices.")
        self.pointers = UnlimitedCache()
        self.pointers[input_indices] = Message({'output':output_indices})

    def check_inputs(self):
        for name, source in self.input_sources.items():
            if not hasattr(source, '__getitem__'):
                raise ValueError("Input sources must be indexable.")

    def __getitem__(self, index):

        index = index_to_list(index)
        return Fireworks.merge([source[self.pointers[index]['output'].values] for source in self.input_sources.values()])

    def __len__(self):
        return len(self.pointers)
