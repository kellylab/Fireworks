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

class Junction:
    """
    A junction can take pipes as inputs, and its outputs can be piped to other pipes.
    All communication is done via Message objects.

    Unlike Pipes, junctions do not automatically have recursive method calling. This is because they have multiple input sources,
    which would result in ambiguity. Instead, junctions are meant to act as bridges between multiple pipes in order to enable
    complex workflows which require more than a linear pipeline.
    """

    def __init__(self, *args, components=None, **kwargs):

        if type(components) is dict:
            self.components = components
        # elif isinstance(inputs, Pipe): # Can give just one pipe as input without having to type out an entire dict
        #     self.input_sources = {'data': inputs}
        elif components is None: # Subclasses can have their own method for creating an inputs_dict and just leave this argument blank
            self.components = {}
        else:
            raise TypeError("Inputs must be a dict of sources, which can be pipes, junctions, or some other object.")


class AggregatorJunction(Junction):
    """
    This junction takes multiple sources implementing __next__ as input and implements a new __next__ method that samples
    its input sources.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_inputs()

    def check_inputs(self):

        for name, source in self.components.items():
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
            return self.components[sample].__next__()
        except StopIteration: # Remove sample from available_inputs list
            self.available_inputs.remove(sample)
            if not self.available_inputs: # Set of inputs is empty, because they have all finished iterating
                raise StopIteration
            else: # Recursively try again
                return self.__next__()

    def reset(self):
        for name, source in self.components.items():
            source.reset()
        self.available_inputs = set(self.components.keys()) # Keep track of which sources have not yet run out.

    def __iter__(self):
        self.reset()
        return self

    @abstractmethod
    def sample_inputs(self):
        """
        Returns the key associated with an input source that should be stepped through next.
        """
        pass

class RandomAggregatorJunction(AggregatorJunction):
    """
    AggregatorJunction that randomly chooses inputs to step through.
    """
    # TODO: Add support for weighted random sampling

    def sample_inputs(self):
        return random.sample(self.available_inputs, 1)[0]

class ClockworkAggregatorJunction(AggregatorJunction):
    """
    AggregatorJunction that iterates through input sources one at a time.
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

class SwitchJunction(Junction):
    """
    This junction has an internal switch that determines which of it's components all method calls will be routed to.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.switch = random.sample(self.components.keys())

    @parameter
    def route(self):
        """
        Returns the component to route method calls to based on the internal switch.
        """
        return self.components[self.switch]

    def __call__(self, *args, **kwargs):

        return self.route(*args, **kwargs)

    def __next__(self):

        return self.route.__next__()

    def __getitem__(self, index):

        return self.route[index]

    def __getattr__(self, attr):

        return getattr(self.route, attr)
