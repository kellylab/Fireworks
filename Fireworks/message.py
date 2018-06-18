import collections
import torch
import pandas as pd

"""
Messages passed between objects in this framework are represented as dictionaries of tensors.
This is analogous to a pandas dataframe, except the elements of the table are tensors rather than series objects. Because
a message is just a dictionary, it can be converted to a dataframe. The inverse will only be possible if every element
of the dataframe can be tensorized (strings can't be converted to tensors.)

This file contains utility functions for working with messages. In order to avoid coupling this framework to a message class,
messages are represented as standard python dicts and are assumed to conform the necessary interface.
"""

class Message:

    def __init__(self, *args, cache = None):
        """
        Initializes a message
        If *args is empty --> empty message
        If *args is two args --> construct a message assuming arg0 is tensors and arg1 is dataframe
        If *args has __getitem__ and keys() attributes --> construct a message from elements of dict, separating tensors from pd series
        An optional cache class can be provided to specify a caching strategy.
        """
        """ Init must be idempotent, attempt to perform type conversions as necessary, and compute/check length. """
        pass

    def __len__(self):
        """
        Must give sensible results for online and streaming type datasets.
        """
        pass

    def __getitem__(self):
        """
        This method must be able to access any index in the global dataset, caching as needed.
        """
        pass

    def append(self, other):
        """
        Compines messages together.
        Should initialize other if not a message already.
        """
        pass

    def map(self, map_dict):
        """
        Applies functions in map_dict to the correspondign keys. map_dict is a dict of keys:functions specifying the mappings.
        """
        pass

    def tensors(self, keys = None):
        """
        Return tensors associated with message as a tensormessage.
        If keys are specified, returns tensors associated with those keys, performing conversions as needed.
        """
        pass

    def df(self, keys = None):
        """
        Returns message as a dataframe. If keys are specified, only returns those keys as a dataframe.
        """
        pass

class TensorMessage(Message):

    def __init__(self, *args):
        """
        Initizes TensorMessage, performing type conversions as necessary.
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def __getattr__(self, *args, **kwargs):
        pass

    def append(self, other):
        pass

# class TensorMessage:
#
#     def __init__(self, message_dict):
#
#         self.dict = {}
#         # Ensure lengths match up
#         for key, value in message_dict.items():
#
#             # Make value list like if not already. We have to ensure this in order to make conversion 2 tensor work properly
#             if not hasattr(value, '__len__'):
#                 value = [value]
#
#             # Attempt to convert to tensor if not already
#             if not isinstance(value, torch.Tensor):
#                 try:
#                     value = torch.Tensor(value)
#                 except Exception as err:
#                     raise TypeError(
#                     ("Elements of a message must be torch tensors. Tried to convert value associated with key {0} " + \
#                     "but got error: {1}").format(key, err))
#
#             # Make sure is not a 0-dim tensor
#             if value.shape == torch.Size([]):
#                 value = torch.Tensor([value])
#
#             self.dict[key] = value
#
#         # for key, value in message_dict.items():
#         #     if not hasattr(value, '__len__') or len(value) == 1 or isinstance(value, str):
#         #         self.dict[key] = [value]
#         #     elif isinstance(value, collections.Iterable) and hasattr(value, 'extend'):
#         #         self.dict[key] = value
#         #     else:
#         #         raise TypeError("Elements of a message must be listlike, being iterable, \
#         #         having a length, and implementing an extend method which enables combining messages together.")
#
#         lengths = [len(value) for value in self.dict.values()]
#         if len(set(lengths)) != 1:
#             raise ValueError("Every element of the message must have the same length.")
#
#         self._length = lengths[0] # Every length should be the same at this point
#
#     def __len__(self):
#         return self._length
#
#     def __getitem__(self, index):
#
#         if not isinstance(index, slice) and index in self.dict.keys():
#             return self.dict[index]
#         else:
#             return Message({key: value[index] for key, value in self.dict.items()})
#
#     def __getattr__(self, *args, **kwargs):
#         return self.dict.__getattribute__(*args, **kwargs)
#
#     def extend(self, other):
#         """
#         Note if the target message has additional keys, those will be dropped.
#         The target message must also have every key present in this message in
#         order to avoid an value error due to length differences.
#         """
#
#         # TODO: Implement iterable specific versions of this (there is too much variability between
#         #       different data structures to rely on one method that can use this.)
#
#         return Message({key: torch.cat([self[key], other[key]]) for key in self.dict})

def combine(*args):
    """ Combines list like objects together. """
        pass
# class Message:
#
#     def __init__(self, tensors, metadata = None):
#
#         if type(tensors) is TensorMessage:
#             self.tensors = tensors
#             self.metadata = metadata
#
#         if not isinstance(metadata, pd.DataFrame):
#             raise TypeError("Metadata must be a pandas dataframe.")
#         self.metadata = pd.DataFrame([])
#         if self.metadata and self.tensors and len(self.tensors) != len(self.metadata):
#             raise ValueError("Tensor data and metadata must have the same length if they are defined.")
#         self._length = len(self.tensors)
#
#     def __len__(self):
#         return self._length
#
#     def __getitem__(self, index):
#
#         if not isinstance(index, slice):
#             if index in self.tensors.keys():
#                 return self.tensors[index]
#             if index in self.metadata.keys():
#                 return self.metadata[index]
#         else:
#             return Message(({key: value[index] for key, value in self.tensors.items()}, metadata={key: value[index] for key, value in self.metadata.items()})
#
#     def extend(self, other):
#         return Message(self.tensors.extend(other.tensors), pd.concat([self.metadata, other.metadata]))
#
#     def _from_tensors(self, tensors, metadata = None):
#         """ Constructs message from a dict of tensors. """
#         return Message(tensors = TensorMessage(tensors))
#
#     def _from_metadata(self, metadata):
#         """ Constructs message from metadata only. """
#         return Message(tensors = None, metadata = metadata)
