import collections

"""
Messages passed between objects in this framework are represented as dictionaries of iterables.
This is similar to a pandas dataframe, except the iterables are allowed to be anything and not just a series objects.
This means messages can contain lists, numpy arrays, tensors, etc. In particular, pandas dataframes are messages.

This file contains utility functions for working with messages. In order to avoid coupling this framework to a message class,
messages are represented as standard python dicts and are assumed to conform the necessary interface.
"""

class Message:

    def __init__(self, message_dict):

        self.dict = {}
        # Ensure lengths match up
        for key, value in message_dict.items():
            if not hasattr(value, '__len__') or len(value) == 1 or isinstance(value, str):
                self.dict[key] = [value]
            elif isinstance(value, collections.Iterable) and hasattr(value, 'extend'):
                self.dict[key] = value
            else:
                raise TypeError("Elements of a message must be listlike, being iterable, \
                having a length, and implementing an extend method which enables combining messages together.")

        lengths = [len(value) for value in self.dict.values()]
        if len(set(lengths)) != 1:
            raise ValueError("Every element of the message must have the same length.")

        self._length = lengths[0] # Every length should be the same at this point

    def __len__(self):
        return self._length

    def __getitem__(self, index):

        if not isinstance(index, slice) and index in self.dict.keys():
            return self.dict[index]
        else:
            return Message({key: value[index] for key, value in self.dict.items()})

    def __getattr__(self, *args, **kwargs):
        return self.dict.__getattribute__(*args, **kwargs)

    def extend(self, other):
        """
        Note if the target message has additional keys, those will be dropped.
        The target message must also have every key present in this message in
        order to avoid an value error due to length differences.
        """

        # TODO: Implement iterable specific versions of this (there is too much variability between
        #       different data structures to rely on one method that can use this.)

        return Message({key: self[key] + (other[key]) for key in self.dict})
