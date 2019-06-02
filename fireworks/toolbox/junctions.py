from abc import ABC, abstractmethod
from Bio import SeqIO
import pandas as pd
from fireworks.core import Message, Junction
from fireworks.utils import index_to_list
from fireworks.core.cache import LRUCache, LFUCache, UnlimitedCache
from abc import ABC, abstractmethod
from itertools import count
import types
import random
from bidict import bidict
import torch

class HubJunction(Junction):
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
            self._available_inputs.remove(sample)
            if not self._available_inputs: # Set of inputs is empty, because they have all finished iterating
                raise StopIteration
            else: # Recursively try again
                return self.__next__()

    def reset(self):
        for name, source in self.components.items():
            if name != '_available_inputs':
                source.reset()
        self._available_inputs = set(self.components.keys()) # Keep track of which sources have not yet run out.
        if '_available_inputs' in self._available_inputs:
            self._available_inputs.remove('_available_inputs')

    def __iter__(self):
        self.reset()
        return self

    @abstractmethod
    def sample_inputs(self):
        """
        Returns the key associated with an input source that should be stepped through next.
        """
        pass

class RandomHubJunction(HubJunction):
    """
    HubJunction that randomly chooses inputs to step through.
    """
    # TODO: Add support for weighted random sampling

    def sample_inputs(self):
        return random.sample(self._available_inputs, 1)[0]

class ClockworkHubJunction(HubJunction):
    """
    HubJunction that iterates through input sources one at a time.
    """
    # TODO: Add support for weighted iteration and setting order (ie. spend two cycles on one input and one on another)

    def reset(self):
        super().reset()
        self.cycle_dict = {i: name for i,name in zip(count(),self._available_inputs)}
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
            if sample in self._available_inputs:
                return sample

class SwitchJunction(Junction):
    """
    This junction has an internal switch that determines which of it's components all method calls will be routed to.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.switch = random.sample(self.components.keys())

    @property
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
