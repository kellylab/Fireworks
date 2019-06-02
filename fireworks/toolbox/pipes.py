from abc import ABC, abstractmethod
from Bio import SeqIO
import pandas as pd
import fireworks
from fireworks.core import Pipe, HookedPassThroughPipe, recursive
from fireworks.core import Message
from fireworks.utils import index_to_list
from fireworks.core.cache import LRUCache, LFUCache, UnlimitedCache
from fireworks.utils import one_hot
from abc import ABC, abstractmethod
from itertools import count
import types
import random
from bidict import bidict
import torch
import numpy as np

class BioSeqPipe(Pipe):
    """
    Class for representing biosequence data.
    Specifically, this class can read biological data files (such as fasta) and iterate through them as a Pipe.
    This can serve as the first Pipe in a pipeline for analyzing genomic data.
    """

    name = 'BioSeqPipe'

    def __init__(self, path, input = None, filetype = 'fasta', **kwargs):
        """
        Args:
            path: Path on disk where file is located; will be supplied to the SeqIO.parse function.
            inputs: Input Pipes. Default is None.
            filetype: Type of file that will be supplied to the SeqIO.parse function. Default is 'fasta'
            kwargs: Optional key word arguments that will be supplied to the SeqIO.parse function.
        """
        Pipe.__init__(self, input=input)
        self.path = path
        self.filetype = filetype
        self.kwargs = kwargs
        self.seq = SeqIO.parse(self.path, self.filetype, **self.kwargs)

    @recursive
    def reset(self):
        """
        Resets the iterator to the beginning of the file. Specifically, it calls SeqIO.parse again as if reinstantiating the object.
        """
        self.seq = SeqIO.parse(self.path, self.filetype, **self.kwargs)

    def __next__(self):
        """
        Each iteration through a SeqIO object will produce a named tuple. This Pipe converts that tuple to a Message and produces the result.
        """
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

        self.reset()
        return self

class LoopingPipe(Pipe):
    """
    This Pipe can take any iterator and make it appear to be indexable by iterating through the input
    as needed to reach any given index.

    The input Pipe must implement __next__ and reset (to be repeatable),
    and this will simulate __getitem__ by repeatedly looping through the iterator as needed.

    For example, say we have a Pipe that iterates through the lines of a FASTA file:
    ::

        fasta = BioSeqPipe('genes.fasta')

    This Pipe can only iterate through the file in one direciton. If we want to access arbitrary elements,
    we can do this:
    ::

        clock = LoopingPipe(inputs=fasta)
        clock[10]
        clock[2:6]
        len(clock)

    All of these actions are now possible. Note that this is in general an expensive process, because the Pipe
    has to iterate one at a time to get to the index it needs. In practice, this Pipe should pipe its output
    to a CachingPipe that can store values in memory. This approach enables you to process datasets that don't
    entirely fit in memory; you can stream in what you need and cache portions. From the perspective of the downstream
    Pipes, every element of the dataset is accessible as if it were in memory.

    """

    name = 'LoopingPipe'

    def __init__(self, input, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self.check_input()
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
            above_values = fireworks.cat([self.step_forward(i+1) for i in above])
        else:
            above_values = Message()
        if len(below) > 0:
            self.reset() # Position will now be reset to 0
            below_values = fireworks.cat([self.step_forward(i+1) for i in below])
        else:
            below_values = Message()
        return below_values.append(above_values) # TODO: Resort this message so values are in the order requested by index

    def __len__(self):
        """
        Length is also calculated implicitly as items are accessed.
        """
        if self.length is not None:
            return self.length
        else:
            self.compute_length()
            return self.length

    def __next__(self): # QUESTION: Should __next__ calls be supported for this Pipe?
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

    def check_input(self):
        """
        Checks input to determine if it implements __next__ and reset methods.
        """
        # for name, pipe in self.inputs.items():
        pipe = self.input

        if pipe is None:
            raise TypeError("This pipe has no inputs.")
        if not (hasattr(pipe, '__next__') and hasattr(pipe, 'reset')):
            raise TypeError('Pipe {0} does not have __next__ and reset methods.'.format(name))

    @recursive()
    def reset(self):

        self.position = 0

    def compute_length(self):
        """
        Step forward as far as the inputs will allow and compute length.

        Length is also calculated implicitly as items are accessed.

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

        This also updates the internal length variable if the iterator ends due to this method call.
        """
        if self.length is not None and n > self.length:
            raise IndexError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
        if n < self.position:
            raise IndexError("Can only step forward to a value higher than current position.")
        x = Message()
        for _ in range(n - self.position):
            try:
                # x = x.append(fireworks.merge([Pipe.__next__() for Pipe in self.inputs.values()]))
                # x = fireworks.merge([pipe.__next__() for pipe in self.inputs.values()])
                x = self.input.__next__()
                self.position += 1
            except StopIteration:
                self.length = self.position
                raise IndexError("Requested index is out of bounds for inputs with length {0}.".format(self.length))
        return x

class CachingPipe(Pipe):
    """
    This Pipe can be used to dynamically cache elements from upstream Pipes.
    Whenever data is requested by index, this Pipe will intercept the request and add that message alongside
    the index to its internal cache.
    This can be useful for dealing with datasets that don't fit in memory or are streamed in. You can cache portions
    of the dataset as you use them. By combining this with a LoopingPipe, you can create the illusion of making
    the entire dataset available to downstream Pipes regardless of the type and size of the original data.

    More specifically, fiven input Pipes that implement __getitem__, will store all calls to __getitem__ into
    an internal cache and therafter __getitem__ calls will either access from the cache or trigger __getitem__ calls
    on the input and an update to the cache.

    For example,

    ::

        fasta = BioSeqPipe(path='genes.fasta')
        clock = LoopingPipe(inputs=fasta)
        cache = CachingPipe(inputs=clock, cache_size=100) # cache_size is optional; default=100

    Will set up a pipeline that reads lines from a FASTA file and acessess and caches elements as requests are made

    ::

        cache[20:40] # This will be cached
        cache[25:30] # This will read from the cache
        cache[44] # This will update the cache
        cache[40:140] # This will fill the cache, flushing out old elements
        cache[25:30] # This will read from the dataset and update the cache again

    """

    def __init__(self, input, *args, cache_size = 100, buffer_size = 0, cache_type = 'LRU', infinite = False, **kwargs):
        """
        Args:
            args: Positional arguments for superclass initialization.
            cache_size (int): Number of elements to hold in cache. Default = 100
            buffer_size (int): Number of elements to clear once cache is full. This is useful if you expect frequent cache misses and
                don't expect to reuse previous cached elements. Default = 0
            cache_type (str): Type of cache. Choices are 'LRU' (least-recently-used) and 'LFU' (least-frequently-used). Default = 'LRU'
            infinite (bool): If set to true, the cache will have no upper limit on size and will never clear itself.
        """
        super().__init__(input, *args, **kwargs)
        self.check_input()
        self.length = None
        self.lower_bound = 0
        self.upper_bound = None
        self.infinite = infinite
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.buffer_size = buffer_size
        self.init_cache(*args, **kwargs)


    def init_cache(self, *args, **kwargs):
        """
        This initializes a cache object at self.cache. There are currently two types of Cache available; LRUCache and LFUCache, and you can
        choose which one by specifying the cache_type argument in the initializer. See fireworks/core/cache.py for more information on
        Message caches.
        """
        choices = {'LRU': LRUCache, 'LFU': LFUCache}
        self.cache = choices[self.cache_type](max_size = self.cache_size, buffer_size=self.buffer_size)

    def check_input(self):
        """
        Checks inputs to determine if they implement __getitem__.
        """
        pipe = self.input
        if pipe is None:
            raise AttributeError("This pipe has no inputs.")

        if not (hasattr(pipe, '__getitem__')):
            raise TypeError('Pipe {0} does not have __getitem__ method.'.format(name))

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
        not_in_cache_elements = self.input[not_in_cache]
        self.cache[not_in_cache] = not_in_cache_elements
        # Reorder and merge requested elements
        message = in_cache_elements.append(not_in_cache_elements)
        indices = in_cache_indices
        indices.extend(not_in_cache_indices)
        # Elements must be reordered based on their order in index
        permutation = indices.copy()
        for i,j in zip(indices, count()):
            permutation[i] = j
        message = message.permute(permutation)
        # Implicit update of internal knowledge of length
        if len(index) and self.length is None and not self.infinite:
            l = max(index)
            if l > self.lower_bound:
                self.lower_bound = l

        return message

    def __len__(self):
        """
        Length is computed implicitly and lazily. If any operation causes the Pipe
        to reach the end of it's inputs, that position is stored as the length.
        Alternatively, if this method is called before that happens, the Pipe will attempt to
        loop to the end and calculate the length.
        """
        if self.infinite:
            raise ValueError("Pipe is labelled as having infinite length (ie. yields items indefinitely).")
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
            raise ValueError("Pipe is labelled as having infinite length (ie. yields items indefinitely).")



class Title2LabelPipe(HookedPassThroughPipe):
    """
    This Pipe takes one Pipe as input and inserts a column called 'label'
    containing the provided title of the input Pipe to to all outputs.
    """

    def __init__(self, title, input, *args, labels_column = 'labels', **kwargs):

        super().__init__(input, *args, **kwargs)
        self.labels_column = labels_column
        self.label = title

    def _get_item_hook(self, message):

        return self.insert_labels(message)

    def _next_hook(self, message):

        return self.insert_labels(message)

    def insert_labels(self, message):

        l = len(message)
        message[self.labels_column] = [self.label for _ in range(l)]

        return message

class LabelerPipe(Pipe):
    """
    This Pipe implements a to_tensor function that converts labels contained in messages to tensors based on an internal labels dict.
    """

    def __init__(self, input, labels, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self.labels = labels
        self.labels_dict = bidict({label:i for label,i in zip(labels,count())})

    def to_tensor(self, batch, labels_column = 'labels'):

        batch['labels_name'] = batch[labels_column]
        labels = batch[labels_column]
        labels = [self.labels_dict[label] for label in labels]
        labels = torch.Tensor([one_hot(label,len(self.labels_dict)) for label in labels])
        batch['labels'] = labels
        try: # Allow an upstream Pipe to attempt to perform to_tensor if it can
            batch = self.recursive_call('to_tensor', batch)
            return batch
        except AttributeError:
            return batch

class RepeaterPipe(Pipe):
    """
    Given an input Pipe that is iterable, enables repeat iteration. In other words, one loop through a RepeaterPipe is equivalent to n loops
    through the original dataset, where n is the number of repetitions that have been configured. Thsi can be useful for oversampling a data
    set without having to duplicate it.
    """

    def __init__(self, input, *args, repetitions=10,**kwargs):
        super().__init__(input, *args,**kwargs)
        if not type(repetitions) is int:
            raise ValueError("Number of repetitions must be provided as an integer.")

        self.repetitions = repetitions
        if not isinstance(self.input, Pipe): # TODO: Test this scenario
            self.iterator = iter(input)

    @recursive()
    def reset(self):
        self.iteration = 0

        if not isinstance(self.input, Pipe):
            self.iterator = iter(self.input)

    def __iter__(self):

        self.reset()
        return self

    def __next__(self):

        if not isinstance(self.input, Pipe):
            return self.iterator.__next__()
        else:
            try:
                return self.recursive_call('__next__')
            except StopIteration:
                self.iteration += 1
                if self.iteration >= self.repetitions:
                    raise StopIteration
                else:
                    self.recursive_call('reset')
                    return self.__next__()

class ShufflerPipe(Pipe):
    """
    Given input Pipes that implement __getitem__ and __len__, will shuffle the indices so that iterating through
    the Pipe or calling __getitem__ will return different values.
    """

    def __init__(self, input, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self.check_input()
        self.current_index = 0
        self.shuffle_indices = np.array([i for i in range(len(self))])

    def check_input(self):
        """
        Check inputs to see if they implement __getitem__ and __len__
        """
        # for name, pipe in self.inputs.items():
        pipe = self.input
        if pipe is None:
            raise TypeError("This Pipe has no inputs.")
        if not (hasattr(pipe, '__getitem__')):
            raise TypeError('Pipe {0} does not have __getitem__ method.'.format(name))
        if not(hasattr(pipe, '__len__')):
            raise TypeError('Pipe {0} does not have __len__ method.'.format(name))

    def __getitem__(self, index):
        new_index = self.shuffle_indices[index]
        return self.recursive_call('__getitem__', new_index.tolist())

    def __next__(self):

        if self.current_index >= len(self):
            raise StopIteration

        message = self[self.current_index]
        self.current_index += 1
        return message

    def shuffle(self, order = None): # TODO: Add support for ordering

        np.random.shuffle(self.shuffle_indices)

    @recursive()
    def reset(self):

        self.shuffle()
        self.current_index = 0

    def __iter__(self):
        self.reset()
        return self

class IndexMapperPipe(Pipe):
    """
    Given input Pipes that implement __getitem__, returns a Pipe that maps indices in input_indices to output_indices via __getitem__
    """
    def __init__(self, input_indices, output_indices, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.check_input()
        if len(input_indices) != len(output_indices):
            raise ValueError("The number of input indices does not match the number of output indices.")
        self.pointers = UnlimitedCache()
        self.pointers[input_indices] = Message({'output':output_indices})
        self._current_index = 0

    def check_input(self):
        # for name, pipe in self.inputs.items():
        pipe = self.input
        if pipe is None:
            raise TypeError("This Pipe has no inputs.")
        if not hasattr(pipe, '__getitem__'):
            raise ValueError("Input Pipes must be indexable.")

    def __iter__(self):

        self._current_index = 0
        return self

    def __next__(self):

        try:
            iteration = self[self._current_index]
            self._current_index += 1
            return iteration
        except (KeyError, IndexError) as e:
            raise StopIteration

    def __getitem__(self, index):

        index = index_to_list(index)

        return self.input[self.pointers[index]['output'].values.tolist()]

    def __len__(self):
        return len(self.pointers)

class BatchingPipe(Pipe):
    """
    Generates minibatches.
    """

    def __init__(self, *args, batch_size=5, **kwargs):

        super().__init__(*args,**kwargs)
        self.current_index = 0
        self.batch_size = batch_size

    def __iter__(self):

        self.reset()
        return self

    @recursive()
    def reset(self):

        self.current_index = 0

    def __next__(self):

        if self.current_index + self.batch_size > len(self):
            raise StopIteration

        try:
            batch = self[self.current_index:self.current_index+self.batch_size]
            self.current_index += self.batch_size
            return batch
        except IndexError:
            raise StopIteration

class FunctionPipe(HookedPassThroughPipe):
    """
    This Pipe is initialized with a function that is applied to all instances of __getitem__, __next__, and __call__.
    This can be useful for quickly inserting a function into the middle of a pipeline.
    """
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        self._function = function

    def _call_hook(self, message): return self._function(message)

    def _next_hook(self, message): return self._function(message)

    def _getitem_hook(self, message): return self._function(message)

class TensorPipe(HookedPassThroughPipe):
    """
    This Pipe converts Messages to tensors. You can specify which columns should be converted.
    """
    def __init__(self, *args, columns=None, cuda=True, device=0, **kwargs):

        super().__init__(*args, **kwargs)
        self.columns = columns
        self.cuda = cuda and torch.cuda.is_available()
        self.device = device

    def _hook(self, message):

        if self.cuda:
            return message.to_tensors(self.columns).cuda(device=self.device, keys=self.columns)
        else:
           return message.to_tensors(self.columns)


    def _call_hook(self, message): return self._hook(message)

    def _next_hook(self, message): return self._hook(message)

    def _getitem_hook(self, message): return self._hook(message)

class GradientPipe(HookedPassThroughPipe):
    """
    This Pipe sets requires_grad=True for Tensor Columns. This is necessary for computing gradients when training a model.
    You can specify which columns should be converted.
    """
    def __init__(self, *args, columns=None, cuda=True, device=0, **kwargs):

        super().__init__(*args, **kwargs)
        self.columns = columns
        self.cuda = cuda and torch.cuda.is_available()
        self.device = device

    def _hook(self, message):

        message.enable_gradients(self.columns)
        return message

    def _call_hook(self, message): return self._hook(message)

    def _next_hook(self, message): return self._hook(message)

    def _getitem_hook(self, message): return self._hook(message)

"""
NOTE: Explore using the below code as an approach to make a general purpose wrapper Pipes,
ie. Pipes that modify all function calls from their input.
"""
# class GenericWrapperPipe(HookedPassThroughPipe):
#     """
#     Wraps all method and attribute calls with a hook.
#     """
#     def __getattr__(self, *args, **kwargs):
#         """
#         Pass through all methods of the input pipe while adding labels.
#         """
#         output = self.input.__getattribute__(*args, **kwargs)
#         if type(output) is types.MethodType: # Wrap the method in a converter
#             return self.method_wrapper(output)
#         else:
#             return self.attribute_wrapper(output)
#
#     def method_wrapper(self, function):
#         """
#         Wraps method with a label attacher such that whenever the method is called, the output is modified
#         by adding the label.
#         """
#
#         def new_function(*args, **kwargs):
#
#             output = function(*args, **kwargs)
#             try:
#                 output = Message(output)
#             except:
#                 return output
#
#             return self.insert_labels(output)
#
#         return new_function
#
#     def attribute_wrapper(self, attribute):
#         """
#         Wraps attribute with new label if attribute returns a message.
#         """
#         try:
#             output = Message(attribute)
#         except:
#             return attribute
#
#         return self.insert_labels(output)
