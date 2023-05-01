from abc import ABC
from .message import Message
from fireworks.utils import index_to_list
from .cache import LRUCache, LFUCache, UnlimitedCache
from types import MethodType
from abc import ABC, abstractmethod

def recursive(accumulate=False):
    """
    Decorator that labels a Pipe method as recursive. This means, that method func will first be called on
    the Pipe's inputs and then on the Pipe itself.
    If accumulate is set to True, then the result from calling the method on a given Pipe will be
    used as input to the next one. If False, then the original arguments will be used when calling
    the method each time.
    """
    def outer(func):

        def wrapper(obj, *args, **kwargs):

            try:
                response = obj.input.__getattribute__(func.__name__)(*args, **kwargs)
            except AttributeError:
                response = None

            if accumulate:
                return func(obj, response)
            else:
                return func(obj, *args, **kwargs)

        return wrapper

    return outer

class Pipe(ABC):
    """
    The core object of computation in fireworks.
    A Pipe can take Pipes as inputs, and its outputs can be streamed to other Pipes.
    All communication is done via Message objects.
    Method calls are deferred to input Pipes recursively until a Pipe that implements the method is reached.

    This is made possible with a recursive function call method. Any Pipe can use this method to call a method on its inputs; this will recursively loop until reaching a Pipe that implements the method and return those outputs (as a Message) or raise an error if there are none. For example, we can do something like this:

    ::

        reader = pipe_for_reading_from_some_dataset(...)
        cache = CachingPipe(reader, type='LRU')
        embedder = CreateEmbeddingsPipe(cache})
        loader = CreateMinibatchesPipe(embedder})

        loader.reset()
        for batch in loader:
        	# Code for training

    Under the hood, the code for loader.__next__() can choose to recursively call a to_tensor() method which is implemented by embedder. Index queries and other magic methods can also be implemented recursively, and this enables a degree of commutativity when stacking Pipes together (changing the order of Pipes is often allowed because of the pass-through nature of recursive calls).

    Note that in order for this to work well, there must be some consistency among method names. If a Pipe expects ‘to_tensor’ to convert batches to tensor format, then an upstream Pipe must have a method with that name, and this should remain consistent across projects to maintain reusability. Lastly, the format for specifying inputs to a Pipe is a dictionary of Pipes. The keys in this dictionary can provide information for the Pipe to use or be ignored completely.

    """

    name = 'base_pipe'
    stateful_attributes = []

    def __init__(self, input = None, *args, **kwargs):

        self.input = input

    def __getitem__(self, *args, **kwargs):

        return self.input.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.input.__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        return self.input.__delitem__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.input.__len__(*args, **kwargs)

    def __next__(self, *args, **kwargs):
        return self.input.__next__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.recursive_call('__call__', *args, **kwargs)

    def __iter__(self, *args, **kwargs):

        if hasattr(self.input, '__iter__'):
            self.input = self.input.__iter__(*args, **kwargs)
        elif hasattr(self.input, '__getitem__'):
            pass
        else:
            raise AttributeError("Input {0} to {1} is not iterable.".format(self.input, self))

        return self

    def __getattr__(self, name):
        """
        If the current pipe does not have the given attribute, this will recursively
        attempt to get the attribute from the input pipe.
        This will also not apply to attributes beginning with underscore. This way, Pipes can have local attributes and methods that do
        not clog up the pipeline namespace.
        This does not intercept special methods (__x__ methods)
        """

        if not name.startswith('_'):
            return self.recursive_call(name, call=False) #self.input.__getattribute__(*args, **kwargs)
        else:
            raise AttributeError("Pipe {0} has no attribute called {1}".format(self, name))

    def get_state(self):
        """
        This returns the current state of the Pipe, which consists of the values of all attributes designated in the list 'stateful_attributes'.
        This can be used to save and load a Pipe's state.

        Args:
            - None

        Returns:
            - A dict of the form {'internal': {...}, 'external': {...}}, where the 'external' subdict is empty. This is so that the representation
            is consistent with the get_state methods of Junctions and Models. We consider all attributes of a Pipe to be internal, and that is
            why the 'external' subdict is empty. See documentation on Component Map for more details on what we mean by that (note that Pipes
            don't use Component_Maps to store state, but simply expose similar methods for compatilibity.)
        """

        return {'internal': {key: getattr(self, key) for key in self.stateful_attributes}, 'external': {}}

    def set_state(self, state, *args, **kwargs):
        """
        Sets the state of the pipe based on the provided state argument.

        Args:
            - state: A dict of the form {'internal': {...}, 'external': {...}}. The 'external' dict will be ignored, because consider all
                     attributes of a Pipe to be in internal (for simplicity). See Component_Map documentation for details.
        """
        for key, value in {**state['internal']}.items():
            setattr(self, key, value)

    def recursive_call(self, attribute, *args, ignore_first = True, call=True,  **kwargs):
        """
        Recursively calls method/attribute on input until reaching an upstream Pipe that implements the method and
        returns the response as a message (empty if response is None).
        Recursive calls enable a stack of Pipes to behave as one entity; any method implemented by any component can be accessed
        recursively.

        Args:
            attribute: The name of the attribute/method to call.
            args: The arguments if this is a recursive method call.
            ignore_first: If True, then ignore whether or not the target attribute is implemented by self. This can be useful if a Pipe
                implements a method and wants to use an upstream call of the same method as well.
            call: If True, and the attribute is a method, the method will be called. Otherwise, it will be returned as is.
            kwargs: The kwargs is this is a recursive method call.

        Returns:
            Responses (dict): A dictionary mapping the name of each input Pipe to the response that was returned.
        """

        if not ignore_first:
            if hasattr(self, attribute):
                try:
                    attr = self.__getattribute__(attribute)
                    if type(attr) is MethodType and call:
                        return attr(*args, **kwargs)
                    else:
                        return attr

                except AttributeError:
                    return self.__getattr__(attribute)

        if not hasattr(self, 'input') or self.input is None:
            raise AttributeError("Pipe {0} does not have method/attribute {1}.".format(self.name, str(attribute)))

        if not isinstance(self.input, Pipe): # If input is not a pipe, just attempt a non-recursive method/attribute call on input.
            if args or kwargs and call: # Is a method call
                try:
                    return self.input.__getattribute__(attribute)(*args, **kwargs)
                except AttributeError:
                    raise AttributeError("Pipe {0} does not have method {1}.".format(self.name, str(attribute)))
            else: # Is an attribute
                try:
                    return self.input.__getattribute__(attribute)
                except AttributeError:
                    try:
                        return self.input.__getattr__(attribute)
                    except AttributeError:
                        raise AttributeError("Pipe {0} does not have attribute {1}".format(self.name, str(attribute)))

        response = self.input.recursive_call(attribute, *args, ignore_first=False, call=call, **kwargs)

        return response

    def _save_hook(self):
        """
        You can override this method to define custom behavior that taks place when a call to save() is made.
        """
        return {}

    def save(self, *args, **kwargs): # TODO: Implement save logic
        """
        This method currently does nothing but pass on a 'save' method call to the input.
        """
        save_dict = self._save_hook(*args, **kwargs)
        if save_dict == {}:
            pass
        else:
            save_df = Message.from_objects(save_dict).to_dataframe().df
            # Save the df using the given method and arguments
            # TODO: Implement

            # Save input

        self.input.save(*args, **kwargs)

class HookedPassThroughPipe(Pipe):
    """
    This Pipe has hooks which can be implemented by subclasses to modify the behavior of
    passed through calls.
    You can define hooks for the following (magic) methods: __getitem__, __call__, and __next__.
    Whenever you call one of these method this will happen:
        1. The method will be recursively called on this Pipes input (if it exists)
        2. The appropriate hook function will be called on the result of that recursive call
        3. This will be the returned value.

    These hooks can make it easy to create pipes that 'do something' every time data is accessed in a certain way. For example, you could have
    the pipe apply some transform to the data.
    """

    name = 'Hooked-passthrough Pipe'

    def _getitem_hook(self, message): return message

    def _call_hook(self, message): return message

    def _next_hook(self, message): return message

    def __getitem__(self, item): # TODO: wrap access methods in try/catch statements

        return self._getitem_hook(Message(self.input.__getitem__(item))) #self.input.__getitem__(*args, **kwargs))

    def __next__(self):
        return self._next_hook(Message(self.input.__next__()))

    def __call__(self, *args, **kwargs): # TODO: Test these hooks more thoroughly.

        try:
            return self._call_hook(self.recursive_call('__call__', *args, **kwargs))
        except AttributeError:
            return self._call_hook(*args, **kwargs)
