
"""
Messages passed between objects in this framework are represented as dictionaries of iterables.
This is similar to a pandas dataframe, except the iterables are allowed to be anything and not just a series objects.
This means messages can contain lists, numpy arrays, tensors, etc. In particular, pandas dataframes are messages.

This file contains utility functions for working with messages. In order to avoid coupling this framework to a message class,
messages are represented as standard python dicts and are assumed to conform the necessary interface.
"""

class Message:

    def __init__(self, message_dict):

        self.dict = message_dict
        # Ensure lengths match up
        # Make elements iterable 

    def __len__(self): pass

    def __getitem__(self, index):
        return Message({key: value[index] for key, value in self.dict.items()})
