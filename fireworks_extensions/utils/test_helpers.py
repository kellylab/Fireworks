import fireworks
import os
import pandas as pd
from fireworks.core.pipe import recursive
from fireworks import Message, Junction, Pipe, Model, PyTorch_Model, model_from_module
from fireworks.utils import index_to_list
import numpy as np
import math
import itertools
from fireworks.utils.exceptions import ParameterizationError
from fireworks.toolbox.pipes import BatchingPipe, LoopingPipe, ShufflerPipe, RepeaterPipe
from fireworks import Message, Junction
import random
import torch
from torch.nn import Parameter
from random import randint

test_dir = fireworks.test_dir
"""
This file contains numerous mock objects which are used by the tests.
"""

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

class recursion_dummy(Pipe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.height = 0
        self.depth = 0

    @recursive()
    def jump(self, n=1):
        self.height += n
        return self.height

    @recursive(accumulate=True)
    def somersault(self, x):

        x = x or 0
        y = x+1
        self.depth += y
        return y

    def poop(self):
        return 3

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

    def __call__(self, *args, **kwargs):

        return Message({'values': np.array([1,2,3])})

    def reset(self):
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        if self.count <= self.length:
            return Message({'values': np.array([self.count-1])})
        else:
            raise StopIteration# This will trigger StopIteration

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.reset()

class ComplexModel(PyTorch_Model):

    required_components = ['activation']

    def __init__(self, components = {}, input = None, in_column = 'x', out_column = 'y', in_features=4, out_features=2):
        PyTorch_Model.__init__(self, components, input = input)
        self.in_column = in_column
        self.out_column = out_column
        self.in_features = in_features
        self.out_features = out_features
        self.components['layer1'] = torch.nn.modules.Linear(100, 23)

    def init_default_components(self):
        self.components['activation'] = torch.nn.Sigmoid()
        

    def forward(self, message):

        y = self.activation(self.linear(message[self.in_column]))
        message[self.out_column] = y

        return message 

class DummyModel(PyTorch_Model):
    """ Implements y = m*x + b """
    required_components = ['m', 'b']

    def __init__(self, components = {}, input = None, in_column = 'x', out_column = 'y'):
        PyTorch_Model.__init__(self, components, input = input)
        self.in_column = in_column
        self.out_column = out_column

    def init_default_components(self):
        """ Default y-intercept to 0 """
        self.components['b'] = Parameter(torch.Tensor([0.]))

    def forward(self, message):

        y = self.m*message[self.in_column]+self.b
        message[self.out_column] = y

        return message

class DummyMultilinearModel(PyTorch_Model):
    """ Implements y = m1*x1 +m2*x1 + b """
    required_components = ['m1', 'm2', 'b']

    def init_default_components(self):
        """ Default y-intercept to 0 """
        self.components['b'] = Parameter(torch.Tensor([0.]))

    def forward(self, message):

        y = self.m1*message['x1']+ self.m2*message['x2'] + self.b
        message['y'] = y
        return message

class LinearJunctionModel(PyTorch_Model):
    """ Implements y = f(x) + b, where the function f(x) is provided as a junction input. """

    required_components = ['b', 'f']

    def init_default_components(self):
        """ Default y-intercept to 0 """
        self.components['b'] = Parameter(torch.Tensor([0.]))

    def forward(self, message):

        y = self.f(message)['z'] + self.b
        message['y'] = y
        return message

class LinearModule(torch.nn.Module):
    """ Dummy PyTorch Module. """
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.m = Parameter(torch.randn(1))
        self.b = Parameter(torch.randn(1))
        self.conv1 = torch.nn.Conv2d(1, 20, 5)

    def forward(self, message):

        message['y'] = self.m*message['x']+self.b

        return message

class DummyUpdateModel(PyTorch_Model):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._count = 0

    def update(self, batch, **kwargs):

        self._count += len(batch)

    def forward(self, batch):
        return batch.append(batch)

class RandomJunction(Junction):

    def __call__(self, *args, **kwargs):

        target = random.sample(self.components.keys(),1)[0]
        return self.components[target](*args, **kwargs)

def generate_linear_model_data(n=300):
    """
    Generates n samples from a linear model with a small variability.
    """
    m = randint(-3,3)
    b = randint(-10,10)
    x = np.random.rand(n)*100
    errors = np.random.normal(0, .4, n) # Gaussian samples for errors
    y = m*x+b + errors

    return Message({'x':x, 'y_true':y}), {'m': m, 'b': b, 'errors': errors} # Second dict is for debugging

def generate_multilinear_model_data(n=550):
    """
    Generates n samples from a multilinear model y = m1*x1 + m2*x2 +b with a small variability.
    """

    m1 = randint(-5,5)
    m2 = randint(-5,5)
    b = randint(-10,10)
    x1 = np.random.rand(n)*100
    x2 = np.random.rand(n)*100
    errors = np.random.normal(0,.1,n) # Gaussian samples for errors
    y = m1*x1 + m2*x2 + b + errors

    return Message({'x1':x1, 'x2': x2, 'y_true':y}), {'m1': m1, 'm2':m2, 'b': b, 'errors': errors} # Second dict is for debugging
