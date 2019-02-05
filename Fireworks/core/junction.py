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

    def save(self, *args, **kwargs):

            save_dict = self._save_hook(*args, **kwargs)
            if save_dict == {}:
                pass
            else:
                save_df = Message.from_objects(save_dict).to_dataframe().df
                # Save the df using the given method and arguments
                # TODO: Implement

                # Save input

            for name, component in self.components.items():
                component.save(*args, **kwargs)

    def _save_hook(self):

        return {}
