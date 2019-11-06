import collections
import torch
import pandas as pd
import numpy as np
import os 
import io
import pyarrow
import tarfile
from copy import deepcopy
from collections import Hashable
from fireworks.utils import index_to_list, slice_length

to_methods = ['csv', 'json', 'dict', 'html', 'feather', 'latex', 'stata', 'msgpack', 'gbq', 'records', 'sparse', 'dense', 'string', 'clipboard']
read_methods = {
    'json': pd.read_json, 'csv': pd.read_csv, 'excel': pd.read_excel, 'hdf': pd.read_hdf, 'parquet': pd.read_parquet,
    'pickle': pd.read_pickle, 'sql_table': pd.read_sql_table, 'state': pd.read_stata, 'table': pd.read_table,
    }

class Message:
    """
    A Message is a class for representing data in a way that can be consumed by different analysis pipelines in python. It does this by
    representing the data as a dictionary of arrays. This is the same approach that Pandas takes, but Messages are specifically designed to
    be used with PyTorch, in which the core data structure is a tensor, which cannot be mixed into a Pandas dataframe.
    Hence, a Message has two elements: a TensorMessage which specifically stores tensor objects and a dataframe which can be used for handling
    any other type of array data. With this structure, you could put your training data in as a TensorMessage and any associated metadata as
    the df and lump all of that into a message. All of the existing df methods can be run on the metadata, and any PyTorch operations can be
    performed on the tensor part.

    Messages support operations such as appending one message to another, joining two messages along keys, applying maps to the message's values,
    and equality checks.

    Additionally, messages behave as much as possible like dicts. In many scenarios, a dict will be able to substitute for a message and vice-versa.
    For example, for preparing batches to feed into an RNN to classify DNA sequences, one could create a Message like this:

    ::

        my_message = Message({
        'embedded_sequences': torch.Tensor([...]),
        'labels': torch.LongTensor([...]),
        'raw_sequences': ['TCGA...',...,],
        ...})

    The Message constructor will parse this dictionary and store the labels and embedded sequences inside a TensorMessage and the
    raw_sequences and other metadata in the dataframe.

    Now we can access elements of this Message:

    ::

        my_message[0]
        my_message[2:10]
        my_message['labels']
        len(my_message)

    We can also move tensors to the GPU and back:

    ::

        my_message.cpu()
        my_message.cuda(device_num=1) # Defaults to 0 for device number
        my_message.cuda(keys=['labels']) # Can specify only certain columns to move if desired

    """

    def __init__(self, *args, metadata = None, length = None, **kwargs):
        """
        Initializes a message
        If *args is empty --> empty message
        If *args is two args --> construct a message assuming arg0 is tensors and arg1 is dataframe
        Alternatively, tensors and dataframes can be specified with kwargs (tensors  = ... and df = ...)
        If *args has __getitem__ and keys() attributes --> construct a message from elements of dict, separating tensors from pd series
        """

        """ The initializer is designed to be idempotent, perform type conversions as necessary, and ensure consistency of lengths. """

        if len(args) == 0:
            tensors = {}
            df = {}

        if len(args) == 2:
            tensors = args[0]
            df = args[1]

        if len(args) == 1:
            # Identify tensors and separate them out
            # The argument may be an existing Message/TensorMessage
            if type(args[0]) is Message:
                tensors = args[0].tensor_message
                df = args[0].df

            elif args[0] is None:
                tensors = {}
                df = {}

            else:
                tensors, df = extract_tensors(args[0])

        if 'tensors' in kwargs:
            tensors = kwargs['tensors']
        if 'df' in kwargs:
            df = kwargs['df']

        self.tensor_message = TensorMessage(tensors) # The TensorMessage constructor will handle type conversions as needed.

        if type(df) is Message:
            df = df.dataframe() # Pull out the df from the message
        self.df = pd.DataFrame(df)
        # Ensure columns are in the same order always
        self.df = self.df.reindex(sorted(self.df.columns), axis=1)

        if length is None:
            self.check_length()
        else:
            self.length = length

        self._current_index = 0 # For iteration

    @classmethod
    def from_objects(cls, *args, **kwargs):
        """
        Returns a message by treating all of the provided values as atomic elments and ignoring length differences. This results in a Message
        of length 1. This can be useful if you want to treat an entire blob as data as a single unit. For example, if you want a row of the
        Message to contain the state of a neural network, one column could be 'bias', and the element could be an entire bias-vector, and
        another column could be 'weights', containing an entire weight matrix in a single row.
        """

        if len(args) == 0:
            tensors = {}
            df = {}

        if len(args) == 2:

            tensors = {key: value.reshape(1,*value.shape) for key, value in args[0].items()}
            df = {key: [value] for key, value in args[1].items()}

        if len(args) == 1:
            if type(args[0]) is Message:
                tensors = {key: value.reshape(1,*value.shape) for key, value in args[0].tensor_message.items()}
                df = {key: [value] for key, value in args[1].df.items()}
            else:
                tensors = {}
                df = {}
                for key, value in args[0].items():
                    if type(value) is torch.Tensor or type(value) is torch.nn.Parameter:
                        tensors[key] = value.reshape(1,*value.shape)
                    else:
                        df[key] = [value]

        if 'tensors' in kwargs:
            tensors = {key: value.reshape(1,*value.shape) for key, value in kwargs['tensors'].items()}
        if 'df' in kwargs:
            df = {key: [value] for key, value in kwargs['df'].items()}

        return cls(tensors, df)

    @classmethod
    def read(cls, method, path, *args, **kwargs):
        """
        Reads the file located at a provided path using a given method and loads the data into a Message. This method uses the Pandas .read_
        functions and converts the resulting DataFrame to a Message. Thus, there is no way currently to go straight from data in a file to
        a TensorMessage. Additionally, only the methods in the read_methods dict at the top of the file are supported.

        You can provide additional positional and key-word arguments which will be passed to the relevant Pandas method to parameterize the
        read.

        Args:
            - method: The method to read the file as. Must be one of: json, csv, excel, hdf, parquet, pickle, sql_table, stata, table.
            - path: The location of the file. This file must exist, be readable, and must be the type specified by method.
            - *args, **kwargs: Additional arguments for the underlying Pandas function call.

        Returns:
            - message: A Message representation of the loaded data.
        """
        if method in read_methods:
            df = read_methods[method](path, *args, **kwargs)
            return cls(df)
        else:
            raise TypeError("Method {0} is not a supported file type for reading. Must be one of {1}".format(method, read_methods.keys()))

    def to(self, method, *args, **kwargs):
        """
        Converts all elements of Message to dataframe and then calls df.to_x based on the given method
        This can be used to serialize and save Messages or convert them into a different format

        Args:
            - method: The method to save the file as. Must be one of: 'json', 'dict', 'html', 'feather', 'latex', 'stata', 'msgpack', 'gbq', 'records', 'sparse', 'dense', 'string', 'clipboard'
            - *args, **kwargs: Additional arguments for the underlying Pandas pd.to_ function call as desired. If the first argument (or kwarg
                               path_or_buf) is provided, it will be used as a filepath to save to.

        Returns:
            - Depending on the method chosen, this will save the Message to a file location and then return the converted Message (eg. to_json will return a json)
        """
        # The conversion to dataframe is essential, because that enforces that all of the data being serialized is on the CPU.
        if method in to_methods:
            tensor_message = deepcopy(self.tensor_message) # This is copied so that the original message is not perturbed by serialization.
            to_serialize = Message(tensor_message, self.df)
            to_serialize = to_serialize.to_dataframe().df # Convert everything to dataframe
            if 'path' in kwargs:
                kwargs['path_or_buf'] = kwargs['path']
                del kwargs['path']
            return getattr(to_serialize, 'to_{0}'.format(method))(*args, **kwargs)
        else:
            raise ValueError("Method {0} is not a valid method for Message serialization. Valid methods include {1}.".format(method, to_methods))

    def to_csv(self, *args, **kwargs):
        """ Converts Message to a CSV. """
        return self.to('csv', *args, **kwargs)

    def to_pickle(self, *args, **kwargs):
        """ Converts Message to a Python Pickle. """
        return self.to('pickle', *args, **kwargs)

    def to_sql(self, *args, **kwargs):
        """ Saves Message to a SQL Database using pd.to_sql function. """
        return self.to('sql', *args, **kwargs)

    def to_dict(self, *args, **kwargs):
        """ Converts Message to a dict. """
        if 'orient' not in kwargs:
            kwargs['orient']  = 'list'
        return self.to('dict', *args, **kwargs)

    def to_excel(self, *args, **kwargs):
        """ Saves Message to an Excel file. """
        return self.to('excel', *args, **kwargs)

    def to_json(self, *args, **kwargs):
        """ Converts Message to a JSON. """
        if 'orient' not in kwargs:
            kwargs['orient']  = 'list'
        return self.to('json', *args, **kwargs)

    def to_string(self, *args, **kwargs):
        """ Converts Message to a string. """
        return self.to('string', *args, **kwargs)

    def save(self, path_or_buffer):
        """
        Converts message to a custom serialized format which retains column structure for tensor and DataFrame components.
        The format is essentially a tarfile containing:
        - Parquet representation of the DataFrame
        - Pickle representation of each tensor
        - JSON mapping column names to their values

        Args:
            path_or_buffer: Can by a path to a file on disk, or a io.BytesIO buffer for serializing/deserializing in memory.
        """
        kwargs = {'mode': 'w:gz'}
        if type(path_or_buffer) is str:
            kwargs['name'] = path_or_buffer
        elif isinstance(path_or_buffer, io.BufferedIOBase):
            kwargs['fileobj'] = path_or_buffer

        with tarfile.open(**kwargs) as tar:
            # Save DataFrame
            df_buffer = io.BytesIO()
            self.df.to_parquet(df_buffer)
            df_buffer.seek(0)
            df_info = tarfile.TarInfo("df.parquet")
            df_info.size = len(df_buffer.read())
            df_buffer.seek(0)
            tar.addfile(df_info, df_buffer)
            df_buffer.close()

            # Save tensors one by one
            for key, value in self.tensor_message.items():
                buffer = io.BytesIO()
                torch.save(value, buffer)
                buffer.seek(0) # Make sure we are at beginning of buffer after writing
                info = tarfile.TarInfo("{key}.torch".format(key=key))
                info.size = len(buffer.read())
                buffer.seek(0)
                tar.addfile(info, buffer)
                buffer.close()

    @classmethod
    def load(cls, path_or_buffer):
        """
        Loads in serialized message at the given path.        

        Args:
            path_or_buffer: Can by a path to a file on disk, or a io.BytesIO buffer for serializing/deserializing in memory.
        """
        kwargs = {'mode': 'r:gz'}
        if type(path_or_buffer) is str:
            kwargs['name'] = path_or_buffer
        elif isinstance(path_or_buffer, io.BufferedIOBase):
            kwargs['fileobj'] = path_or_buffer
            path_or_buffer.seek(0)
        
        df = None
        tensors = {}
        with tarfile.open(**kwargs) as tar:
            for filename in tar.getnames():
                if filename.endswith('.parquet'): # Is the DataFrame
                    buffer = tar.extractfile(tar.getmember(filename))
                    df = pd.read_parquet(buffer)
                elif filename.endswith('.torch'): # Is a Tensor
                    buffer = tar.extractfile(tar.getmember(filename))
                    name = filename.rstrip('.torch')
                    tensors[name] = torch.load(buffer)
        return cls(tensors, df)


    def __setstate__(self, state):
        loaded = self.__class__.load(state)
        self.df = loaded.df
        self.tensor_message = loaded.tensor_message

    def __getstate__(self):
        buffer = io.BytesIO()
        self.save(buffer)
        return buffer

    def check_length(self):
        """
        Checks that lengths of the internal tensor_message and dataframe are the same and equalto self.len
        If one of the two is empty (length 0), then that is fine.
        """
        if len(self.df) == 0:
            self.length = len(self.tensor_message)
        elif len(self.tensor_message) == 0:
            self.length = len(self.df)
        elif len(self.tensor_message) != len(self.df):
            raise ValueError("Every element of the message, including tensors and arrays, must have the same length unless one or both are None.")
        else:
            self.length = len(self.tensor_message)

    def __iter__(self):

        self._current_index = 0
        return self

    def reset(self):

        return self.__iter__()
    
    def __next__(self):

        try:
            iteration = self[self._current_index]
            self._current_index += 1
            return iteration
        except (KeyError, IndexError) as e:
            raise StopIteration

    def __len__(self):
        """
        The length of a Message is the length of its internal DataFrame and/or internal TensorFrame. These two must have the same length
        for consistency. In this way, row i of a Message is row i of the DataFrame and row i of the TensorFrame.
        The only exception to this rule is if one of the two are empty. In that case, the empty component is ignored. However, the moment you
        insert data into the empty component (eg. insert a Tensor into a Message that previously only had DataFrame elements), then the lengths
        must again be equal (ie. you can only insert a Tensor that has the same length as the Message.)
        """
        return self.length

    def __eq__(self, other):
        """
        Two messages are equal if they have the same keys, and for each key the elements are equal.
        """
        tensors_equal = (self.tensor_message == other.tensor_message)
        df_equal = (self.df.equals(other.df))

        return tensors_equal and df_equal

    def __getitem__(self, index):
        """

        Args:
            index: An integer, slice, or list of integer indices.

        Returns:
            item: The element of the Message referenced by the index.
        """

        t = type(index)
        if t is int:
            if index >= self.length:
                raise IndexError
            index = slice(index,index+1)
            return self._getindex(index)
        elif t is slice:
            start = index.start or 0
            stop = index.stop or self.length
            if max(start, stop) > self.length:
                raise IndexError
            return self._getindex(index)
        elif t is list: # Must account for list of indices and list of strings
            if len(index) == 0:
                return Message(length=0)
            tt = type(index[0])
            if tt is int: # Is index
                return self._getindex(index)
            if tt is str: # Is list of column names
                return Message({s:self.tensor_message[s] if s in self.tensor_message.keys() else self.df[s] for s in index}, length = slice_length(index))
        elif t is str:
            if index in self.tensor_message.keys():
                return self.tensor_message[index]
            elif index in self.df.keys():
                return self.df[index]
        else:
            raise KeyError("{0} is not a column name or a valid index for this message.".format(str(index)))

    def _getindex(self, index):
        """
        Returns a submessage based on requested index.

        Args:
            index: An integer, slice, or list of integer indices.

        """
        if len(self.tensor_message) == 0:
            return Message(self.df.iloc[index].reset_index(drop=True), length=slice_length(index))
        if len(self.df) == 0:
            return Message(self.tensor_message[index], length=slice_length(index))
        else:
            return Message(self.tensor_message[index], self.df.iloc[index].reset_index(drop=True), length=slice_length(index))

    def __setitem__(self, index, value):
        """

        Args:
            index: An integer, slice, or list of integer indices.
            value: The value to set the index to. Must be an object that can be fed to the Message initializer and produce a message
                with the same length as the index.
        """
        t = type(index)

        if t is str:
            # Check if the update would require moving a column from df to tensor_message or vice-versa, in which case, delete the original
            # and then insert the new value into the correct location.
            if (type(value) is torch.Tensor and index in self.df.columns) or (not type(value) is torch.Tensor and index in self.tensor_message.columns):
                del self[index]

            # Update or insert
            if type(value) is torch.Tensor:
                self.tensor_message[index] = value # This will trigger a length check automatically
            else:
                if len(value) == self.length: # Check value before updating df
                    self.df[index] = value
                elif self.length == 0: # Inserting into empty Message is allowed
                    self.df[index] = value
                    self.check_length()
                else:
                    raise ValueError("Cannot set value {0} of length {1} to column name {2} for message with length {3}. Lengths of all columns \
                must be the same.".format(value, len(value), index, self.length))

        if t is int:
            index = slice(index, index+1)
            value = Message(value)
            self.tensor_message[index] = value.tensor_message
            value.df.index = self.df.iloc[index].index
            self.df.iloc[index] = value.df
        if t is slice:
            value = Message(value)
            self.tensor_message[index] = value.tensor_message
            value.df.index = self.df.iloc[index].index
            self.df.iloc[index] = value.df
        elif t is list:
            if len(index):
                tt = type(index[0])
                if tt is str: # Update multiple columns at once with a dict
                    for column in index:
                        self[column] = value[column]
                elif tt is int:
                    value = Message(value)
                    self.tensor_message[index] = value.tensor_message
                    value.df.index = self.df.iloc[index].index
                    self.df.iloc[index] = value.df
                else:
                    raise IndexError("{0} is not a valid index.".format(index))
            else:
                raise ValueError("No index specified.")

    def __delitem__(self, index):
        """
        Checks if index is a string (column name) or index (row, range of rows) and deletes the appropriate part of the message.
        If index is a list, it can be a list of either column names or indices.

        Args:
            index: An integer, slice, or list of integer indices.

        """
        t = type(index)
        if t is str: # Delete column
            if index in self.tensor_message.tensor_dict.keys():
                del self.tensor_message.tensor_dict[index]
            elif index in self.df.columns:
                del self.df[index]
            else:
                raise KeyError("Column {0} not found in message.")
        elif t is int: # Delete single row
            if index >= self.length:
                raise IndexError("Cannot delete element {0} from message with length {1}".format(index, self.length))
            self._delindex(index)
        elif t is slice:
            self._delindex(index)
        elif t is list:
            if len(index): # Otherwise list is empty and do nothing
                tt = type(index[0])
                if tt is str: # Delete columns
                    for column in index:
                        if column in self.tensor_message.tensor_dict.keys():
                            del self.tensor_message.tensor_dict[column]
                        elif column in self.df.columns:
                            del self.tensor_message[column]
                        else:
                            raise KeyError("Column {0} not found in message.")
                elif tt is int: # Delete multiple indices
                    m = max(index)
                    if m > self.length:
                        raise IndexError("Cannot delete element {0} from message with length {1}.".format(m, self.length))
                    else:
                        self._delindex(index)
        else:
            raise IndexError("{0} is not a column(s) or index in this message.".format(str(index)))
        if len(self.tensor_message.keys()) == 0:
            self.tensor_message.length = 0

    def _delindex(self, index):
        """
        Deletes the given index after it has been validated by __delitem__

        Args:
            index: An integer, slice, or list of integer indices.

        """
        if len(self.df):
            self.df.drop(self.df.index[index], inplace=True)
            self.length = len(self.df)
        if len(self.tensor_message):
            del self.tensor_message[index]
            self.length = self.tensor_message.length

    def __contains__(self, key):
        """
        This checks if the given key is present as a column in this Message.
        """
        if type(key) is str:
            return key in self.df or key in self.tensor_message
        else:
            return False

    def __getattr__(self, name):
        """
        If name is the name of a column in this Message, then this returns that column. Otherwise, this calls __getattr__ on the internal
        DataFrame or Tensor_Message.
        """
        if name in self.__getattribute__('df').keys():
            return self.df.__getattribute__(name)
        elif name in self.__getattribute__('tensor_message').keys():
            return self.tensor_dict.__getattr__(name)
        else:
            raise AttributeError("This message has no attribute or column named {0}.".format(name))

    def __repr__(self):
        return "Message with \n Tensors: \n {0} \n Metadata: \n {1}".format(self.tensor_message, self.df)

    def __copy__(self):
        copy = Message(self.tensor_message, self.df)
        return copy

    def __deepcopy__(self, memo):
        copy = Message(deepcopy(self.tensor_message), deepcopy(self.df))
        return copy

    @property
    def columns(self):
        """
        Returns the names of columns in this Message
        """
        return self.df.columns.append(pd.Index(self.tensor_message.keys()))

    def keys(self):
        """
        Returns names of keys in this Message.
        Note: This is the same as self.columns
        """

        return self.columns

    @property
    def index(self):
        """
        Returns index for internal tensors
        """
        # NOTE: Index currently has no meaning, because messages are currently required to be indexed from 0:length
        return self.df.index

    def append(self, other):
        """
        Compines messages together.
        Should initialize other if not a message already.

        Args:
            other: The message to append to self. Must have the same keys as self so that in the resulting Message,
                every column continues to have the same length as needed.
        """

        other = Message(other)
        # Check if self is empty, and if so, replace it with other.
        if self == empty_message:
            return other

        appended_tensors = self.tensor_message.append(other.tensor_message)
        appended_df = self.df.append(other.df).reset_index(drop=True)
        return Message(appended_tensors, appended_df)

    def enable_gradients(self, columns=None):
        columns = columns or self.tensor_message.columns
        for c in columns:
            self[c].requires_grad=True

    def merge(self, other):
        """
        Combines messages horizontally by producing a message with the keys/values of both.

        Args:
            other: The message to merge with self. Must have different keys and the same length as self to ensure length consistencies.
                Alternatively, if either self or other have an empty TensorMessage or df, then they can be merged together safely as long
                as the resulting Message has a consistent length.
                For example:
                ::
                    message_a = Message({'a': [1,2,3]}) # This is essentially a DataFrame
                    message_b = Message({'b': torch.Tensor([1,2,3])}) # This is essentially a TensorMessage
                    message_c = Message_a.merge(message_b) # This works

        Returns:
            message: The concatenated Message containing columns from self and other.

        """
        other = Message(other)
        return Message({**self.tensor_message, **other.tensor_message}, {**self.df, **other.df})

    def map(self, mapping):
        """
        Applies function mapping to message. If mapping is a dict, then maps will be applied to the corresponding keys as columns, leaving
        columns not present in mapping untouched.
        In otherwords, mapping would be a dict of column_name:functions specifying the mappings.

        Args:

            mapping: Can either be a dict mapping column names to functions that should be applied to those columns, or a single function.
                In the latter case, the mapping function will be applied to every column.

        Returns:

            message: A Message with the column:value pairs produced by the mapping.

        """
        if type(mapping) is dict:
            return Message({key: mapping[key](self[key]) if key in mapping else self[key] for key in self.columns})
        else:
            return Message({key: mapping(self[key]) for key in self.columns})

    def tensors(self, keys = None):
        """
        Return tensors associated with message as a tensormessage.
        If keys are specified, returns tensors associated with those keys, performing conversions as needed.

        Args:
            keys: Keys to get. Default = None, in which case all tensors are returned as a TensorMessage.
                If columns corresponding to requested keys are not tensors, they will be converted.

        Returns:
            tensors (TensorMessage): A TensorMessage containing the tensors requested.

        """
        if keys is None:
            return self.tensor_message
        else:
            return Message(TensorMessage({key:self[key] for key in keys}))

    def to_tensors(self, keys=None): # TODO: Test
        """
        Returns message with columns indicated by keys converted to Tensors. If keys is None, all columns are converted.

        Args:
            keys: Keys to get. Default = None, in which case all columns are mapped to Tensor.

        Returns:
            message: A Message in which the desired columns are Tensors.
        """
        if keys is None:
            keys = list(self.df.keys())

        tensor_message = Message(TensorMessage({key:self[key] for key in keys}))
        self[keys] = tensor_message

        return self

    def dataframe(self, keys = None):
        """
        Returns message as a dataframe. If keys are specified, only returns those keys as a dataframe.

        Args:
            keys: Keys to get. Default = None, in which case all non-tensors are returned as a DataFrame.
                If columns corresponding to requested keys are tensors, they will be converted (to np.arrays).

        Returns:
            df (pd.DataFrame): A DataFrame containing the columns requested.

        """
        if keys is None:
            return self.df
        else:
            return pd.DataFrame({key: np.array(self[key]).tolist() for key in keys})

    def to_dataframe(self, keys = None):
        """
        Returns message with columns indicated by keys converted to DataFrame. If keys is None, all tensors are converted.

        Args:
            keys: Keys to get. Default = None, in which case all tensors are mapped to DataFrame.

        Returns:
            message: A Message in which the desired columns are DataFrames.
        """
        if keys is None:
            keys = list(self.tensor_message.keys())

        new = Message(self.df)

        for key in keys:
            tensor = self[key].cpu().detach().numpy()
            if len(tensor.shape) > 1:
                tensor = tensor.tolist()
            new[key] = tensor

        return new

    def permute(self, index):
        """
        Reorders elements of message based on index.

        Args:
            index: A valid index for the message.

        Returns:
            message: A new Message with the elements arranged according to the input index.
                For example,
                ::
                message_a = Message({'a':[1,2,3]})
                message_b = message_a.permute([2,1,0])
                message_c = Message({'a': [3,2,1]})
                message_b == message_c

                The last statement will evaluate to True
        """
        df = self.df
        tensor_message = self.tensor_message

        if len(df):
            df_index = self.df.index
            df = self.df.reindex(index)
            df.index = df_index
        if len(tensor_message):
            tensor_message = self.tensor_message.permute(index)

        return Message(tensor_message, df)

    def cpu(self, keys = None):
        """
        Moves tensors to system memory. Can specify which ones to move by specifying keys.

        Args:
            keys: Keys to move to system memory. Default = None, meaning all columns are moved.

        Returns:
            message (Message): Moved message
        """
        self.tensor_message = self.tensor_message.cpu(keys)

    def cuda(self, device = 0, keys = None):
        """
        Moves tensors to gpu with given device number. Can specify which ones to move by specifying keys.

        Args:
            device (int): CUDA device number to use. Default = 0.
            keys: Keys to move to GPU. Default = None, meaning all columns are moved.

        Returns:
            message (Message): Moved message

        """
        self.tensor_message = self.tensor_message.cuda(device, keys)

        return self

class TensorMessage:
    """
    A TensorMessage is a class for representing data meant for consumption by pytorch as a dictionary of tensors. It is analogous to a
    Pandas DataFrame, except all elements are PyTorch Tensors (or torch.nn.Parameter). This can be useful when working with PyTorch, as you
    can have your Models use column names to keep track of what each Tensor is for, and by parameterizing column names, you can make your
    Models somewhat generic (instead of taking x as an argument that must have a specific format, you take take a TensorMessage and act on
    columns that can be configured at runtime or in your script.)

    TensorMessages, along with DataFrames, constitute the components of a Message. We recommend using Messages rather than TensorMessages
    directly, as all of the functionality of TensorMessages is encapsulated by a Message, and this gives you the ability to seamlessly move
    data to and from Tensor format, so you can mix your deep learning tasks with other data science tasks that you would use Pandas for.
    """
    def __init__(self, message_dict = None, map_dict = None):
        """
        Initizes TensorMessage, performing type conversions as necessary.
        """
        self.tensor_dict = {}
        self.map_dict = map_dict

        if type(message_dict) is Message:
            self.tensor_dict = message_dict.tensor_message.tensor_dict
            self.length = message_dict.tensor_message.length
        elif type(message_dict) is TensorMessage:
            self.tensor_dict = message_dict.tensor_dict
            self.length = message_dict.length
        elif message_dict is None:
            self.tensor_dict = {}
            self.length = 0
        else:
            for key, value in message_dict.items():
                # Make value list like if not already. For example, for length-1 messages.
                if not hasattr(value, '__len__'):
                    value = [value]
                # Apply (optional) mappings. For example, to specify a torch.BytesTensor instead of a FloatTensor.
                if type(self.map_dict) is dict and key in map_dict:
                    value = self.map_dict[key](value)
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

    def __getitem__(self, index): # TODO: Add support for nonstandard slices (0:, :-k, etc.)

        t = type(index)
        if t is str:
            return self.tensor_dict[index] # This will raise a key error if column not present
        elif t is int:
            if index >= self.length:
                raise IndexError
            else:
                return TensorMessage({key: value[index] for key, value in self.tensor_dict.items()})
        elif t is slice:
            start = index.start or 0
            stop = index.stop or self.length
            if max(start, stop) > self.length:
                raise IndexError
            else:
                return TensorMessage({key: value[index] for key, value in self.tensor_dict.items()})

        elif t is list: # Check if list of column names or indices
            if len(index):
                tt = type(index[0])
                if tt is str:
                    return TensorMessage({key:self.tensor_dict[key] for key in index})
                elif tt is int:
                    return TensorMessage({key: value[index] for key, value in self.tensor_dict.items()})
            else: # Empty list
                return TensorMessage()
        else:
            raise KeyError("{0} is not a column name or a valid index for this message.".format(str(index)))

    def __setitem__(self, index, value):
        """
        Sets the rows specified by the index to the provided value, which can be a dict of Tensors/list-likes, or a Message/TensorMessage/DataFrame
        of the appropriate length. All non-tensor objects will be converted to Tensor if possible (and this will cause an error if it's not possible).
        """
        t = type(index)
        if t is str:
            # Check length
            if len(value) != self.length and self.length:
                raise ValueError("Cannot set column {0} to value {1} with length {2} for message with length {3}".format(index, value, len(value), self.length))
            if not type(value) is torch.Tensor:
                value = torch.Tensor(value)
            self.tensor_dict[index] = value
            if self.length == 0: # Update length if inserting into empty TensorMessage
                self.length = len(value)

        elif t is int or t is slice:
            value = TensorMessage(value)
            for key in self.tensor_dict:
                self.tensor_dict[key][index] = value[key] # BUG: Out of bound slice queries return an empty Message instead of raising an error.
        elif t is list:
            if len(index):
                tt = type(index[0])
                if tt is str: # List of column names
                    for column, tensor in value.items():
                        self[column] = value # This will trigger recursive call to __setitem__
                elif tt is int:
                    value = TensorMessage(value)
                    for key in self.tensor_dict:
                        self.tensor_dict[key][index] = value[key] # Tensors can be indexed by list
                else:
                    raise IndexError("{0} is not a valid index.".format(index))
            else: # Empty list
                raise ValueError("No index specified for setitem.")

    def __delitem__(self, index):
        """
        Deletes the rows corresponding to the provided index.
        """
        t = type(index)
        if t is str:
            del self.tensor_dict[index] # Will automatically throw a KeyError if not in the dict
        elif t is int:
            if index > self.length:
                raise IndexError
            self._delindex(index)
        elif t is slice:
            # if max(index.stop, index.start) > self.length:
            #     raise IndexError
            self._delslice(index)
        elif t is list:
            if len(index):
                tt = type(index[0])
                if tt is str:
                    for column in index:
                        del self.tensor_dict[column]
                elif tt is int:
                    self._dellist(index)
            else:
                raise KeyError

        if len(self.keys()) == 0: # If you've deleted the last column, the length is now 0 #TODO: Test this
            self.length = 0

    def _delindex(self, index):
        """
        Deletes single row corresponding to a single index. This is called by __delitem__.
        """
        new_length = self.length - 1
        if index == 0:
            if self.length == 1: # Deleting the only row of a 1-row Message
                self.tensor_dict = {}
                self.length = new_length
            else: # Deleting first row of an n-row message

                self.tensor_dict = self[1:self.length].tensor_dict
                self.length = new_length
        elif index == self.length-1: # Deleting the last row
            self.tensor_dict = self[:self.length-1].tensor_dict
            self.length = new_length
        else: # Delete an in-between row
            self.tensor_dict = TensorMessage(cat([self[0:index], self[index+1:self.length]])).tensor_dict

    def _delslice(self, orange):
        """
        Deletes rows corresponding to a slice object. This is called by __delitem__.
        This involves finding the complementary set of indices and returning those.
        """
        if orange.step is None or orange.step in [0, 1, -1]: # Unit step size
            # NOTE: We will treat -1 and 1 as the same for now
            if orange.step == -1: # Is backwards
                start = orange.stop
                stop = orange.start
            else:
                start = orange.start
                stop = orange.step

            if start is None:
                start = 0
            if stop is None:
                stop = self.length

            if start >= stop:
                raise IndexError("Invalid slice {0}.".format(str(orange)))
            if start > 0:
                bottom_message = self[0:start]
            else:
                bottom_message = TensorMessage()
            if stop < self.length and stop > start:
                top_message = self[stop:self.length]
            else:
                top_message = TensorMessage()

            # TODO: Have to flip response if a downward slice is requested
            new_message = TensorMessage(cat([bottom_message, top_message]))

            self.tensor_dict = new_message.tensor_dict
            self.length = new_message.length

        else: # Step count is some number that must be accounted for
            listy = slice_to_list(orange) # NOTE: This is low priority so for now we just convert to list and delete that way
            self._dellist(listy)

    def _dellist(self, listy):
        """
        Deletes rows corresponding to indices in a list. This is called by __delitem__.
        This involves finding the complementary set of indices and returning those.
        """
        complement = [i for i in range(self.length) if i not in listy]
        self.tensor_dict = self[complement]
        self.length = len(complement)

    def __copy__(self):

        return TensorMessage(self.tensor_dict, self.map_dict)

    def __deepcopy__(self, memo):

        return TensorMessage(deepcopy(self.tensor_dict), deepcopy(self.map_dict))

    def __contains__(self, key):
        """
        Returns true if the requested key is a column in this TensorMessage.
        """
        if isinstance(key, Hashable):
            return key in self.tensor_dict
        else:
            return False

    def __getattr__(self, name):
        return self.tensor_dict.__getattribute__(name)

    def keys(self,*args, **kwargs):
        """
        Returns the column names of this TensorMessage, which as the keys of the internal tensor_dict.
        """
        return self.tensor_dict.keys(*args, **kwargs)

    def __len__(self):
        """
        The length of all columns in the TensorMessage must be the same to ensure consistency. By length, we mean the first dimension of a
        Tensor's shape (so if a Tensor is 2x4x6, its length is 2, and each 'row' consists of a 4x6 matrix.)
        """
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

    @property
    def tensor_message(self):
        return self

    @property
    def df(self):
        return pd.DataFrame()

    @property
    def columns(self):
        """
        Returns names of tensors in TensorMessage
        """
        return self.tensor_dict.keys()

    @property
    def index(self):
        """
        Returns the index for internal tensors
        """
        # NOTE: Index currently has no meaning, because messages are currently required to be indexed from 0:length
        return pd.RangeIndex(0,self.length)

    def enable_gradients(self, columns=None):
        columns = columns or self.columns
        for c in columns:
            self[c].requires_grad=True

    def disable_gradients(self, columns=None):
        pass

    def append(self, other):
        """
        Note if the target message has additional keys, those will be dropped.
        The target message must also have every key present in this message in
        order to avoid an value error due to length differences.
        """
        other = TensorMessage(other)

        # If self is empty, just replace it with other.
        if self == empty_tensor_message:
            return other
        return TensorMessage({key: torch.cat([self[key], other[key]]) if key in other.keys() else self[key] for key in self.tensor_dict.keys()})

    def merge(self, other):
        """
        Combines self with other into a message with the keys of both.
        self and other must have distinct keys.
        """
        other = TensorMessage(other)
        return TensorMessage({**self, **other})

    def permute(self, index):
        """
        Rearranges elements of TensorMessage to align with new index.
        """
        permutation = self[index]
        return permutation

    def cuda(self, device = 0, keys=None):
        """
        Moves all tensors in TensorMessage to cuda device specified by device number. Specify keys to limit the transformation
        to specific keys only.
        """
        if keys is None:
            keys = self.tensor_dict.keys()
        for key in keys:
            self.tensor_dict[key] = self.tensor_dict[key].cuda(device)

        return self

    def cpu(self, keys=None):
        """
        Moves all tensors in TensorMessage to cpu. Specify keys to limit the transformation
        to specific keys only.
        """
        if keys is None:
            keys = self.tensor_dict.keys()
        for key in keys:
            self.tensor_dict[key] = self.tensor_dict[key].cpu()

        return self

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

def slice_to_list(s):
    """
    Converts a slice object to a list of indices
    """
    step = s.step or 1
    start = s.start
    stop = s.stop
    return [x for x in range(start,stop,step)]

def complement(indices, n):
    """ Given an index, returns all indices between 0 and n that are not in the index. """

    if type(indices) is slice:
        indices = slice_to_list(indices)
    if type(indices) is int:
        indices = [indices]

    complement = [i for i in range(n) if i not in indices]

    return complement

def cat(list_of_args):
    """
    Concatenates messages in list_of_args into one message.
    """

    if list_of_args == []:
        return Message()
    m = Message(list_of_args[0])
    if len(list_of_args) > 1:
        for arg in list_of_args[1:]:
            m = m.append(Message(arg))

    return m

def merge(list_of_args):
    """
    Merges messages in list_of_args into one message with all the keys combined.
    """

    m = Message(list_of_args[0])
    if len(list_of_args) > 1:
        for arg in list_of_args[1:]:
            m = m.merge(Message(arg))

    return m

# These are here for comparison purposes when testing if something is empty and is used by some of the methods in this file.
empty_tensor_message = TensorMessage()
empty_message = Message()
