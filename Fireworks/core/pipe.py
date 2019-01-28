from abc import ABC, abstractmethod
from Bio import SeqIO
import pandas as pd
from .message import Message
from Fireworks.utils import index_to_list
from .cache import LRUCache, LFUCache, UnlimitedCache
from abc import ABC, abstractmethod
from itertools import count
import types
import random
from bidict import bidict
import torch
import math
import numpy as np

class Pipe(ABC):
    """
    The core data structure in fireworks.
    A Pipe can take Pipes as inputs, and its outputs can be streamed to other Pipes.
    All communication is done via Message objects.
    Method calls are deferred to input Pipes recursively until a Pipe that implements the method is reached.

    This is made possible with a recursive function call method. Any Pipe can use this method to call a method on its inputs; this will recursively loop until reaching a Pipe that implements the method and return those outputs (as a Message) or raise an error if there are none. For example, we can do something like this:

    ::

        reader = pipe_for_reading_from_some_dataset(...)
        cache = CachingPipe(reader, type='LRU')
        embedder = CreateEmbeddingsPipe(cache})
        loader = CreateMinibatchesPipe(embedder})

        loader.reset()
        for batch in loader:
        	# Code for training

    Under the hood, the code for loader.__next__() can choose to recursively call a to_tensor() method which is implemented by embedder. Index queries and other magic methods can also be implemented recursively, and this enables a degree of commutativity when stacking Pipes together (changing the order of Pipes is often allowed because of the pass-through nature of recursive calls).

    Note that in order for this to work well, there must be some consistency among method names. If a Pipe expects ‘to_tensor’ to convert batches to tensor format, then an upstream Pipe must have a method with that name, and this should remain consistent across projects to maintain reusability. Lastly, the format for specifying inputs to a Pipe is a dictionary of Pipes. The keys in this dictionary can provide information for the Pipe to use or be ignored completely.

    """

    name = 'base_pipe'

    def __init__(self, input = None, *args, **kwargs):

        self.input = input

    def __getitem__(self, *args, **kwargs):
        return self.input.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.input.__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        return self.input.__delitem__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.input.__len__(*args, **kwargs)

    def __next__(self, *args, **kwargs):
        return self.input.__next__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.input.__call__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return self.input.__iter__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        """
        Pass through all methods of the input Pipe while adding labels. This does not intercept special methods (__x__ methods)
        """

        return self.recursive_call(*args, **kwargs) #self.input.__getattribute__(*args, **kwargs)

    def recursive_call(self, attribute, *args, ignore_first = True, **kwargs):
        """
        Recursively calls method/attribute on input until reaching an upstream Pipe that implements the method and
        returns the response as a message (empty if response is None).
        Recursive calls enable a stack of Pipes to behave as one entity; any method implemented by any component can be accessed
        recursively.

        Args:
            attribute: The name of the attribute/method to call.
            args: The arguments if this is a recursive method call.
            ignore_first: If True, then ignore whether or not the target attribute is implemented by self. This can be useful if a Pipe
                implements a method and wants to use an upstream call of the same method as well.
            kwargs: The kwargs is this is a recursive method call.

        Returns:
            Responses (dict): A dictionary mapping the name of each input Pipe to the response that was returned.
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

        if not hasattr(self, 'input') or self.input is None:
            raise AttributeError("Pipe {0} does not have method/attribute {1}.".format(self.name, str(attribute)))

        if not isinstance(self.input, Pipe): # If input is not a pipe, just attempt a non-recursive method/attribute call on input.
            if args or kwargs: # Is a method call
                try:
                    return self.input.__getattribute__(attribute)(*args, **kwargs)
                except AttributeError:
                    raise AttributeError("Pipe {0} does not have method {1}.".format(self.name, str(attribute)))
            else: # Is an attribute
                try:
                    return self.input.__getattribute__(attribute)
                except AttributeError:
                    try:
                        return self.input.__getattr__(attribute)
                    except AttributeError:
                        raise AttributeError("Pipe {0} does not have attribute {1}".format(self.name, str(attribute)))

        response = self.input.recursive_call(attribute, *args, ignore_first=False, **kwargs)
        return response

    # class recursive_decorator:
    #     """
    #     Decorator that labels a Pipe method as recursive. This means, that method func will first be called on
    #     the Pipe's inputs and then on the Pipe itself.
    #     If accumulate is set to True, then the result from calling the method on a given Pipe will be
    #     used as input to the next one. If False, then the original arguments will be used when calling
    #     the method each time.
    #     """
    #     def __init__(self, outer):
    #         self.outer = outer
    #
    #     def __call__(self, accumulate=True):
    #         def wrapper(func, *args, **kwargs):
    #             response = self.outer.recursive_call(func.__name__, *args, **kwargs)
    #             if accumulate:
    #                 return func(response)
    #             else:
    #                 return func(*args, **kwargs)
    #         return wrapper

        # if response:
        #     if isinstance(responses[0], Pipe):
        #         return Fireworks.merge(responses)
        #     elif len(responses) == 1:
        #         return responses[0]
        #     else:
        #         return {key: response for key, respone in zip(self.inputs.keys(), responses)}

# @deprecated(version='0.2.8', reason="Use Pipe instead.")
# class PassThroughPipe(Pipe):
#     """
#     This Pipe passes through data access calls and methods to its (single) input Pipe except for whatever is overridden by subclasses.
#     NOTE: Only the special methods explicitly defined here (getitem, len, delitem, setitem, next, iter) are passed through.
#     Non-special methods are passed through normally.
#     """
#
#     # TODO: Make every Pipe passthrough.
#     name = 'Passthrough Pipe'

class HookedPassThroughPipe(Pipe): # BUG NOTE: Methods that return self will break the passthrough at the moment
    """
    This Pipe has hooks which can be implemented by subclasses to modify the behavior of
    passed through calls.
    """

    name = 'Hooked-passthrough Pipe'

    def _getitem_hook(self, message): return message

    # def _setitem_hook(self, *args, **kwargs): pass
    #
    # def _delitem_hook(self, *args, **kwargs): pass
    #
    # def _len_hook(self, *args, **kwargs): return args[0]

    # TODO: Implement and test a hook functionality for __call__

    def _call_hook(self, messsage): return message

    def _next_hook(self, message): return message

    # def _iter_hook(self, *args, **kwargs): return args[0]

    def __getitem__(self, *args, **kwargs): # TODO: wrap access methods in try/catch statements

        return self._getitem_hook(Message(self.input.__getitem__(*args, **kwargs))) #self.input.__getitem__(*args, **kwargs))

    # def __setitem__(self, *args, **kwargs):
    #     self._setitem_hook(self.input.__setitem__(*args, **kwargs))
    #
    # def __delitem__(self, *args, **kwargs):
    #     self._delitem_hook(self.input.__delitem__(*args, **kwargs))
    #
    # def __len__(self, *args, **kwargs):
    #     return self._len_hook(self.input.__len__(*args, **kwargs))

    def __next__(self, *args, **kwargs):
        return self._next_hook(Message(self.input.__next__(*args, **kwargs)))

    def __call__(self, *args, **kwargs):

        if hasattr(self, 'input') and hasattr(self.input, '__call__'):
            return self._call_hook(self.input.__call__(*args, **kwargs))
        else:
            return self._call_hook(*args, **kwargs)

    def __iter__(self, *args, **kwargs):

        self.input = self.input.__iter__(*args, **kwargs)
        return self

    # def __getattr__(self, *args, **kwargs):
    #     """
    #     Pass through all methods of the input Pipe while adding labels. This does not intercept special methods (__x__ methods)
    #     """
    #     return self.input.__getattribute__(*args, **kwargs)

# def recursive(pipin, target=None, accumulate=False):
#     """
#     Decorator that labels a Pipe method as recursive. This means, that method func will first be called on
#     the Pipe's inputs and then on the Pipe itself.
#     If accumulate is set to True, then the result from calling the method on a given Pipe will be
#     used as input to the next one. If False, then the original arguments will be used when calling
#     the method each time.
#     """
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             # response = pipe.recursive_call(func.__name__, *args, **kwargs)
#             response = 2
#             assert False
#             if accumulate:
#                 return func(response)
#             else:
#                 return func(*args, **kwargs)
#
#         return wrapper
#     return decorator
