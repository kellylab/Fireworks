import Fireworks
import os
import pandas as pd
from Fireworks import Message, Junction, Pipe
from Fireworks.utils import index_to_list
import numpy as np
import math
import itertools

test_dir = Fireworks.test_dir

class one_way_dummy(Pipe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def __next__(self):
        self.count += 1
        if self.count <= 20:
            return {'count': np.array([self.count-1])}
        else:
            raise StopIteration# This will trigger StopIteration

    def reset(self):
        self.count = 0

class one_way_iter_dummy(one_way_dummy):

    def __iter__(self):
        self.reset()
        return self

class reset_dummy(Pipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def reset(self):
        self.count = 0

class next_dummy(Pipe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def __next__(self):
        self.count += 1
        if self.count < 20:
            return {'count': np.array([self.count])}
        else:
            raise StopIteration # This will trigger StopIteration

class getitem_dummy(Pipe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = 20

    def __getitem__(self, index):

        index = index_to_list(index)

        # if index == []:
        if len(index) == 0:
            return None
        elif max(index) < self.length and min(index) >= 0:
            return {'values': np.array(index)}
        else:
            raise IndexError("Out of bounds for dummy pipe with length {0}.".format(self.length))

    def __len__(self):
        return self.length

class smart_dummy(Pipe):
    """
    Implements all of the methods in the above dummies
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = 20
        self.count = 0

    def __getitem__(self, index):

        index = index_to_list(index)

        if index == []:
            return None
        elif max(index) < self.length and min(index) >= 0:
            return Message({'values': np.array(index)})
        else:
            raise IndexError("Out of bounds for dummy pipe with length {0}.".format(self.length))

    def reset(self):
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        if self.count <= 20:
            return Message({'values': np.array([self.count-1])})
        else:
            raise StopIteration# This will trigger StopIteration

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.reset()
