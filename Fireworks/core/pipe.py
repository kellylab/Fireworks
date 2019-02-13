from abc import ABC
from .message import Message
from Fireworks.utils import index_to_list
from .cache import LRUCache, LFUCache, UnlimitedCache
from types import MethodType
from abc import ABC, abstractmethod

#TODO: Make methods/attributes beginning with underscore exempt from recursion in order to simplify namespace issues

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
    The core data structure in fireworks.
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

        self.input = self.input.__iter__(*args, **kwargs)
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

                # if args or kwargs: # Is a method call
                #     return self.__getattribute__(attribute)(*args,**kwargs)
                #
                # else: # Is an attribute
                #     try:
                #         return self.__getattribute__(attribute)
                #     except AttributeError:
                #         return self.__getattr__(attribute)

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

        return {}

    def save(self, *args, **kwargs):

        save_dict = self._save_hook(*args, **kwargs)
        if save_dict == {}:
            pass
        else:
            save_df = Message.from_objects(save_dict).to_dataframe().df
            # Save the df using the given method and arguments
            # TODO: Implement

            # Save input

        self.input.save(*args, **kwargs)

    # class recursive_decorator:
    #     """
    #     Decorator that labels a Pipe method as recursive. This means, that method func will first be called on
    #     the Pipe's inputs and then on the Pipe itself.
    #     If accumulate is set to True, then the result from calling the method on a given Pipe will be
    #     used as input to the next one. If False, then the original arguments will be used when calling
    #     the method each time.
    #     """
    #     def __init__(self, outer):
    #         self.outer = outer
    #
    #     def __call__(self, accumulate=True):
    #         def wrapper(func, *args, **kwargs):
            #     response = self.outer.recursive_call(func.__name__, *args, **kwargs)
            #     if accumulate:
            #         return func(response)
            #     else:
            #         return func(*args, **kwargs)
            # return wrapper

        # if response:
        #     if isinstance(responses[0], Pipe):
        #         return Fireworks.merge(responses)
        #     elif len(responses) == 1:
        #         return responses[0]
        #     else:
        #         return {key: response for key, respone in zip(self.inputs.keys(), responses)}

class HookedPassThroughPipe(Pipe): # BUG NOTE: Methods that return self will break the passthrough at the moment
    """
    This Pipe has hooks which can be implemented by subclasses to modify the behavior of
    passed through calls.
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

        # if hasattr(self, 'input') and hasattr(self.input, '__call__'):
        #     return self._call_hook(self.recursive_call('__call__', *args, **kwargs))
        # else:
        #     return self._call_hook(*args, **kwargs)
        try:
            return self._call_hook(self.recursive_call('__call__', *args, **kwargs))
        except AttributeError:
            return self._call_hook(*args, **kwargs)

    # def __iter__(self, *args, **kwargs):
    #
    #     self.input = self.input.__iter__(*args, **kwargs)
    #     return self

# def recursive(pipin, target=None, accumulate=False):
#     """
#     Decorator that labels a Pipe method as recursive. This means, that method func will first be called on
#     the Pipe's inputs and then on the Pipe itself.
#     If accumulate is set to True, then the result from calling the method on a given Pipe will be
#     used as input to the next one. If False, then the original arguments will be used when calling
#     the method each time.
#     """
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             # response = pipe.recursive_call(func.__name__, *args, **kwargs)
#             response = 2
#             assert False
#             if accumulate:
#                 return func(response)
#             else:
#                 return func(*args, **kwargs)
#
#         return wrapper
#     return decorator
