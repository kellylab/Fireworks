import collections
import torch
import pandas as pd
import numpy as np
from collections import Hashable

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
            if type(args[0]) is Message:
                tensors = args[0].tensor_message
                df = args[0].df
            else:
                tensors, df = extract_tensors(args[0])

        self.tensor_message = TensorMessage(tensors) # The TensorMessage constructor will handle type conversions as needed.

        if type(df) is Message:
            df = df.dataframe() # Pull out the df from the message
        self.df = pd.DataFrame(df)

        self.check_length()

    def check_length(self):

        if len(self.df) == 0:
            self.length = len(self.tensor_message)
        elif len(self.tensor_message) == 0:
            self.length = len(self.df)
        elif len(self.tensor_message) != len(self.df):
            raise ValueError("Every element of the message, including tensors and arrays, must have the same length unless one or both are None.")
        else:
            self.length = len(self.tensor_message)

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
        if type(index) is int: # Making point queries into range queries of width 1 ensures correct formatting of dataframes by pandas.
            index = slice(index, index+1)
        return Message(self.tensor_message[index], self.df.iloc[index].reset_index(drop=True)) # {k: v[index] for k, v in self.tensor_dict.items()}, {k: v.iloc[index] for k, v in self.df.items()}

    def __setitem__(self, index, value):

        if isinstance(index, Hashable) and index in self.tensor_message.keys():
            self.tensor_message[index] = value
            self.check_length()
        elif isinstance(index, Hashable) and index in self.df.keys():
            self.df[index] = value
            self.check_length()
        else:
            if type(index) is int: # To ensure proper formatting of dataframes, convert point queries to length 1 range queries.
                index = slice(index, index+1)
            value = Message(value)
            self.tensor_message[index] = value.tensor_message
            self.df.iloc[index] = value.df

    def __repr__(self):
        return "Message with \n Tensors: \n {0} \n Metadata: \n {1}".format(self.tensor_message, self.df)

    def append(self, other):
        """
        Compines messages together.
        Should initialize other if not a message already.
        """

        other = Message(other)
        appended_tensors = self.tensor_message.append(other.tensor_message)
        appended_df = self.df.append(other.df).reset_index(drop=True)
        return Message(appended_tensors, appended_df)

    def merge(self, other):
        """
        Combines messages horizontally by producing a message with the keys/values of both.
        """
        other = Message(other)
        return Message({**self.tensor_message, **other.tensor_message}, {**self.df, **other.df})

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
        if keys is None:
            return self.tensor_message
        else:
            return TensorMessage({key:self[key] for key in keys})

    def dataframe(self, keys = None):
        """
        Returns message as a dataframe. If keys are specified, only returns those keys as a dataframe.
        """
        if keys is None:
            return self.df
        else:
            return pd.DataFrame({key: np.array(self[key]) for key in keys})

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
    def __init__(self, message_dict = None, map_dict = None):
        """
        Initizes TensorMessage, performing type conversions as necessary.
        """
        self.tensor_dict = {}

        if type(message_dict) is Message:
            self.tensor_dict = message_dict.tensor_message.tensor_dict
            self.length = message_dict.tensor_message.length
        elif type(message_dict) is TensorMessage:
            self.tensor_dict = message_dict.tensor_dict
            self.length = message_dict.length
        elif message_dict is None:
            pass
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

    def __setitem__(self, index, value):

        if isinstance(index, Hashable) and index in self.tensor_dict: # Index is the name of a key
            self.tensor_dict[index] = value
            self.length = compute_length(self)

        else:
            value = TensorMessage(value)
            for key in self.tensor_dict:
                self.tensor_dict[key][index] = value[key]

    # def __delitem__(self, index): # BUG: delitem raises sigfaults when called on a tensor.
    #
    #     for key in self.tensor_dict:
    #         del self[key][index]

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
        other = TensorMessage(other)
        return TensorMessage({key: torch.cat([self[key], other[key]]) if key in other.keys() else self[key] for key in self.tensor_dict})

    def merge(self, other):
        """
        Combines self with other into a message with the keys of both.
        self and other must have distinct keys.
        """
        other = TensorMessage(other)
        return TensorMessage({**self, **other})

def compute_length(of_this):
    """
    Of_this is a dict of listlikes. This function computes the length of that object, which is the length of all of the listlikes, which
    are assumed to be equal. This also implicitly checks for the lengths to be equal, which is necessary for Message/TensorMessage.
    """
    lengths = [len(value) for value in of_this.values()]
    if lengths == []:
        # An empty dict has 0 length.
        return 0
    if len(set(lengths)) != 1: # Lengths are not all the same
        raise ValueError("Every element of dict must have the same length.")

    return lengths[0]

def extract_tensors(from_this):
    """
    Given a dict from_this, returns two dicts, one containing all of the key/value pairs corresponding to tensors in from_this, and the other
    containing the remaining pairs.
    """
    tensor_dict = {k:v for k, v in from_this.items() if type(v) is torch.Tensor}
    other_keys = from_this.keys() - tensor_dict.keys() if tensor_dict else from_this.keys()
    other_dict = {k: from_this[k] for k in other_keys}
    return tensor_dict, other_dict

# def to_array(listlike):
#     """
#     If listlike is a tensor, converts it to a numpy array.
#     """
#     if type(listlike) is torch.Tensor
#         return np.array()

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
