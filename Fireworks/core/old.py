from abc import ABC, abstractmethod
from Bio import SeqIO
import pandas as pd
from Fireworks.utils import index_to_list
from .message import Message
from .cache import LRUCache, LFUCache, UnlimitedCache
from .pipe import Pipe
from abc import ABC, abstractmethod
from itertools import count
import types
import random
from bidict import bidict
import torch
import math
import numpy as np

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

    def __iter__(self, *args, **kwargs):

        self.input = self.input.__iter__(*args, **kwargs)
        return self

    # def __getattr__(self, *args, **kwargs):
    #     """
    #     Pass through all methods of the input Pipe while adding labels. This does not intercept special methods (__x__ methods)
    #     """
    #     return self.input.__getattribute__(*args, **kwargs)
