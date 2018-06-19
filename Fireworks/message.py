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
    """
    A Message is a class for representing data in a way that can be consumed by different analysis pipelines in python. It does this by
    representing the data as a dictionary of arrays. This is the same approach that pandas takes, but Messages are specifically designed to
    be used with pytorch, in which the core data structure is a tensor, which cannot be mixed into a pandas dataframe.
    Hence, a Message has two elements: a TensorMessage which specifically stores tensor objects and a dataframe which can be used for handling
    any other type of array data. With this structure, you could put your training data in as a TensorMessage and any associated metadata as
    the df and lump all of that into a message. All of the existing df methods can be run on the metadata, and any pytorch operations can be
    performed on the tensor part.
    Messages support operations such as appending one message to another, joining two messages along keys, applying maps to the message's values,
    and equality checks.
    Additionally, messages behave as much as possible like dicts. In many scenarios, a dict will be able to substitute for a message and vice-versa.
    """

    def __init__(self, *args, cache = None, **kwargs):
        """
        Initializes a message
        If *args is empty --> empty message
        If *args is two args --> construct a message assuming arg0 is tensors and arg1 is dataframe
        Alternatively, tensors and dataframes can be specified with kwargs (tensors  = ... and df = ...)
        If *args has __getitem__ and keys() attributes --> construct a message from elements of dict, separating tensors from pd series
        An optional cache class can be provided to specify a caching strategy.
        """
        """ Init must be idempotent, attempt to perform type conversions as necessary, and compute/check length. """
        if len(args) == 0:
            tensors = {}
            df = {}
        if 'tensors' in kwargs:
            tensors = kwargs['tensors']
        if 'df' in kwargs:
            df = kwargs['df']
        if len(args) == 2:
            tensors = args[0]
            df = args[1]
        if len(args) == 1:
            # Identify tensors and separate them out
            # The argument may be an existing Message/TensorMessage
            pass
        self.tensor_message = TensorMessage(tensors)
        self.df = pd.DataFrame(df)
        tensor_length = len(self.tensor_message)
        df_length = len(self.df)
        if hasattr(self.df, 'shape') and (len(self.df.shape) == 1 or self.df.shape[1] == 1):
            df_length = 1
        # if 'baba' in kwargs:
        #     assert False
        if tensor_length == df_length:
            self.length = tensor_length
        elif tensor_length == 0: # Has only dataframes
            self.length = df_length
        elif df_length == 0: # Has only tensors
            self.length = tensor_length
        else:
            raise ValueError("Every element of the message, including tensors and arrays, must have the same length.")

    def __len__(self):
        """
        Must give sensible results for online and streaming type datasets.
        """
        return self.length

    def __eq__(self, other):
        """
        Two messages are equal if they have the same keys, and for each key the elements are equal
        """
        tensors_equal = (self.tensor_message == other.tensor_message)
        df_equal = (self.df.equals(other.df))

        return tensors_equal and df_equal

    def __getitem__(self, index):
        """
        This method must be able to access any index in the global dataset, caching as needed.
        """
        if not isinstance(index, slice):
            # Attempt to access elements by key
            if index in self.tensor_message.keys():
                return self.tensor_message[index]
            elif index in self.df.keys():
                return self.df[index]

        # Access elements by index
        # assert False
        return Message(self.tensor_message[index], self.df.iloc[index], baba=True) # {k: v[index] for k, v in self.tensor_dict.items()}, {k: v.iloc[index] for k, v in self.df.items()}

    def __repr__(self):
        return "Message with \n Tensors: \n {0} \n Metadata: \n {1}".format(self.tensor_message, self.df)

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
        return self.tensor_message

    def dataframe(self, keys = None):
        """
        Returns message as a dataframe. If keys are specified, only returns those keys as a dataframe.
        """
        return self.df

    def cpu(self, keys = None):
        """ Moves tensors to system memory. Can specify which ones to move by specifying keys. """
        pass

    def cuda(self, device = 0, keys = None):
        """ Moves tensors to gpu with given device number. Can specify which ones to move by specifying keys. """
        pass

class TensorMessage:
    """
    A TensorMessage is a class for representing data meant for consumption by pytorch as a dictionary of tensors.
    """
    def __init__(self, message_dict, map_dict = None):
        """
        Initizes TensorMessage, performing type conversions as necessary.
        """
        self.tensor_dict = {}
        if type(message_dict) is type(self):
            self.tensor_dict = message_dict.tensor_dict
            self.length = message_dict.length
        else:
            for key, value in message_dict.items():
                # Make value list like if not already. For example, for length-1 messages.
                if not hasattr(value, '__len__'):
                    value = [value]
                # Apply (optional) mappings. For example, to specify a torch.BytesTensor instead of a FloatTensor.
                if type(map_dict) is dict and key in map_dict:
                    value = map_dict[key](value)
                # Convert to tensor if not already
                if not isinstance(value, torch.Tensor):
                    try:
                        value = torch.Tensor(value)
                    except Exception as err:
                        raise TypeError(
                        ("Elements of a message must be torch tensors. Tried to convert value associated with key {0} " + \
                        "but got error: {1}").format(key, err))
                # Make sure is not a 0-dim tensor
                if value.shape == torch.Size([]):
                    value = torch.Tensor([value])

                self.tensor_dict[key] = value

        self.length = compute_length(self.tensor_dict)

    def __getitem__(self, index):
        if not isinstance(index, slice) and index in self.tensor_dict.keys():
            return self.tensor_dict[index]
        else:
            return TensorMessage({k: v[index] for k, v in self.tensor_dict.items()})

    def __getattr__(self, *args, **kwargs):
        return self.tensor_dict.__getattribute__(*args, **kwargs)

    def __len__(self):
        return self.length

    def __eq__(self, other):
        """
        Two tensor messages are equivalent if they have the same keys and their elements are equal.
        """
        keys_equal = set(self.keys()) == set(other.keys())
        if not keys_equal:
            return False
        for key in self.keys():
            if not torch.equal(self[key], other[key]):
                return False
        return True

    def __repr__(self):
        return "TensorMessage: {0}".format(self.tensor_dict)

    def append(self, other):
        """
         Note if the target message has additional keys, those will be dropped.
         The target message must also have every key present in this message in
         order to avoid an value error due to length differences.
         """
        return TensorMessage({key: torch.cat([self[key], other[key]]) for key in self.tensor_dict})

    def join(self, other):
        """
        Combines self with other into a message with the keys of both.
        self and other must have distinct keys.
        """
        pass

def compute_length(of_this):

    lengths = [len(value) for value in of_this.values()]
    if lengths == []:
        # An empty dict has 0 length.
        return 0
    if len(set(lengths)) != 1: # Lengths are not all the same
        raise ValueError("Every element of dict must have the same length.")

    return lengths[0]

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
