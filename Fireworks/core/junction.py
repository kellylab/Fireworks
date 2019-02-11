from abc import ABC, abstractmethod
from Bio import SeqIO
import pandas as pd
from .message import Message
from Fireworks.utils import index_to_list
from Fireworks.utils.exceptions import ParameterizationError
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
        elif components is None: # Subclasses can have their own method for creating an inputs_dict and just leave this argument blank
            self.components = {}
        else:
            raise TypeError("Inputs must be a dict which can contain pipes, junctions, or some other object.")
        self.check_components()
        self.update_components()

    def check_components(self, components = None):
        """
        Checks to see if the provided components dict provides all necessary params for this model to run.
        """
        if components is None:
            components = self.components
        missing_components = []
        error = False
        # missing_junctions = []
        for key in self.required_components:
            if key not in components:
                missing_components.append(key)
                error = True
        if error:
            raise ParameterizationError("Missing required components {0}".format(missing_components))

    def type_check(self, key, components):

        if type(self.required_components) is dict:
            if not isinstance(components[key], self.required_components[key]):
                # raise ValueError("Component {0} with value {1} is not the correct type. Must be an instance of {2}".format(key, components[key], self.required_components[key))
                raise ValueError
        else:
            return

    def update_components(self, components=None):

        if components is None:
            components = self.components

        for name in components:
            self.type_check(name, components)
            setattr(self, name, self.components[name])

    @property
    def required_components(self):
        """
        This should be overridden by a subclass in order to specify components that should be provided during initialization. Otherwise,
        this will default to just return the components already present within the Model.
        """
        return self.components.keys()

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
