import torch
from .message import Message
from torch.nn import Parameter, Module

class Component_Map(dict):
    """
    Each of the main objects that can be used to construct pipelines (Pipes, Junctions, and Models) have a means for tracking their internal
    state. For Pipes, this is handled by a simple dictionary, but for Junctions and Models we use this class which satisfies the more complex
    needs for these objects.
    In particular, a Component_Map can track whether a variable is internal or external to a given object, and if it's external, whom the
    variable belongs to. This lets us dynamically assign a variable of one object to to a component of a Junction or Model while maintaining
    the distinction that the assigned variable is not internal to Junction or Model.
    This distinction can be useful for variables such as hyperparameters or runtime configurations (eg. whether to use Cuda) that one does
    not want to store alongside variables like model weights. You can also have a Model 'borrow' variables from another Model while maintaining
    this distinciton (eg. use the first two layers from this other model, then use the remainder of the layers using internal weights), and
    this can be useful when training Models (you could have your optimizer operate only on a Model's internal parameters, treating everything
    else as constant.)
    These are a just few examples of how this abstraction can be useful, and in simpler terms, it is essentially a means to deliberately pass
    variables by reference, which is not how Python's memory model operates by default, but it can be extremely helpful when doing machine
    learning.
    The details of the interaction with a Component_Map are abstracted away by Junctions and Models. Hence, you shouldn't have to directly
    interact with a Component_Map. Instead, you can generally just call set_state and get_state on Junctions and Models to get serialized
    representations of the Component_Maps.

    The format of these serialization is a dict of the form {'external': {...}, 'internal': {...}}.
    The 'internal' dict contains a mapping between variable names and those variables.
    The 'external' dict contains a mapping between variable names and the object that those variables belong to. In this way, a Component_Map
    can keep track of the owner of the linked variable and also get its value as needed. Hence, Junctions and Models can simply use that
    variable as if it were internal, and this makes it easy to swap variables around without changing syntax (eg. replace some internal
    component of a Model with an attribute of some other object on the fly.)

    A Component_Map behaves like a dict with the special property that if you assign an tuple of the form (obj, x) to the dict, where
    x is a string, then the Component_Map will treat that as a 'pass by reference' assignment. In other words, it will assume that you
    want to externally link the variable obj.x to the Component_Map. For example, if you do this:

    ::

        A = some_object()
        cm = Component_Map()
        cm['a'] = (A, 'x')

    Now whenever you call cm['a'], you will get whatever is returned by A.x.

    ::

        cm['a'] == A.x # This evaluates to True.
        cm['a'] is A.x # This also evaluates to True, because the assignment is by reference.

    If you cm.get_state(), the 'external' dict will contain a reference to A.

    ::

        state = cm.get_state()
        external = state['external']
        external['a'] == (A, 'x') # This evaluates to True.

    On the other hand, if you do this:

    ::

        cm['a'] = A.x # Don't pass by reference.
        cm['a'] == A.x # This evaluates to True.
        cm['a'] is A.x # This may or may not be True because Python sometimes assigns by reference and sometimes copies data depending on the situation.

    This will be treated as an internal assignment. Note that PyTorch implements logic for enforcing pass-by-reference for torch.nn.Parameter
    objects. Hence, if A.x was a Parameter, then the assignment will be by reference. However, we will have no way of knowing who the
    'owner' of the Parameter is, and by using Component_Maps, we also are able to extend this functionality to any Python object. If
    you now get the state, it will be in the 'internal' dict.

    ::

        state = cm.get_state()
        internal = state['internal']
        internal['a'] == A.x # This evaluates to True. If A.x is vector/tensor-valued, you may get a vector/tensor of 1's.

    """

    def __init__(self, components):
        """
        self._internal_components functions like a normal dictionary. If you call cm['a'] and 'a' is in the internal_components dict, then
        it will return cm._internal_components['a']. However, if 'a' is external, then cm['a'] will return
        getattr(cm._external_modules['a'], c._external_attribute_names['a']).
        """
        dict.__init__(self)
        self._external_modules = {}
        self._external_attribute_names = {}
        self._internal_components = {}

        for key, value in components.items():
            self[key] = value

    def __setitem__(self, key, val):
        """
        This overrides the __setitem__ method of dict so that if you set an item of the form (obj, x) to a key k, where x is a string,
        the Component_Map will 'link' obj.x to k. ie. cm['k'] == obj.x. It will do this by inserting obj into the _external_modules dict
        and x into the _external_attribute_names dict with k as the key for both. Hence, when you call cm['k'], the Component_Map will fetch
        obj and the attribute name x and then call getattr(obj, x).

        Note that you cannot have a key in both the internal and external dicts. If you already have a key in one and you assign to the other,
        the former will be deleted. This prevents ambiguity when accessing elements.
        """
        if type(val) is tuple and len(val) is 2 and type(val[1]) is str and hasattr(val[0], val[1]):
            # Very specific test to check if the intention is to link an attribute inside of another
            # object to this Component Map rather than simply set the value of the key to a tuple.

            if key in self._internal_components:
                # Delete from internal components if this key already exists
                del self[key]
            obj, attribute = val
            value = getattr(obj, attribute)
            # key, value = self.setitem_hook(key, value)
            # self._external_components[key] = value
            self._external_modules[key] = obj
            self._external_attribute_names[key] = attribute
        else:
            if key in self._external_modules:
                # Deelte from external components if this key already exists.
                del self[key]
            value = val
            key, value = self.setitem_hook(key, value)
            self._internal_components[key] = value

        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        """
        This also deletes the key from the the internal and external dicts.
        """

        try:
            dict.__delitem__(self, key)
            for d in [self._external_attribute_names, self._external_modules, self._internal_components]:
                if key in d:
                    del d[key]
        except KeyError:
            raise KeyError

    def __getitem__(self, key):
        """
        This returns the object referenced by the key, whether it is internal or external.
        """
        if key in self._internal_components:
            return self._internal_components[key]
        elif key in self._external_modules:
            return getattr(self._external_modules[key], self._external_attribute_names[key])
        else:
            raise AttributeError()

    def setitem_hook(self, key, value):
        """
        This can be overridden by a subclass in order to implement specific actions
        that should take place before an attribute is set.
        """
        return key, value

    def set_state(self, state):
        """
        This method can be used to apply a serialized representation of state to a Component_Map at once. This is used for loading in saved
        data.

        Args:
            state: A dict of the form {'external': {...}, 'internal': {...}}. The elements of this dict will be assigned to the Component_Map.
                   Note that this will not reset the Component_Map, so if there were previous elements already present, those will remain
                   in the Component_Map.
        """
        for key, value in {**state['internal'], **state['external']}.items():

            self[key] = value

    def get_state(self):
        """
        This returns a serialized representation of the state of the Component_Map.

        Args:
            None

        Returns:
            state: A dict of the form {'external': {...}, 'internal': {...}}. See above documentation for more information.
        """
        internal = self._internal_components
        external = {key: (self._external_modules[key], self._external_attribute_names[key]) for key in self._external_modules}

        return {'internal': internal, 'external': external}

class PyTorch_Component_Map(Component_Map):
    """
    This is a subclass of Component_Map with additional functionality for dealing with PyTorch data structures. PyTorch has a lot of logic
    in the background to keep track of Parameters and gradients and where objects are located in memory. The PyTorch_Component_Map has
    a modified __setitem__ method which ensures that there are no conflicts with any of these background operations by PyTorch.
    In particular, a PyTorch_Component_Map can have a (PyTorch) Model assigned to it, and whenever __setitem__ is called, the item is
    1) Converted to a torch.nn.Parameter object if possible. This is essential for computing gradients and training the parameter.
    2) Recursively assigned if necessary. This concept is best explained with an example. Say you have a neural network with a convolutional
       layer,

       ::

            model = some_pytorch_model()
            model.conv1 = torch.nn.Conv2d(4,4,4) # This represents a 4x4 convolutional layer with 4 channels.

        'model.conv1' is itself a PyTorch Module with its own internal state, and in general, models can have models that have models,
        and so on. In other words, 'model.conv1' could itself have variables that are Modules/Models and so on. When you get the state
        dict for the original model, you will get nested dictionaries. These can still be serialized and saved to a file like normal, but
        when we call set_state, we want to make sure that we assign these nested dictionary elements to the correct submodules.

        ::

            state = model.get_state()
            internal = state['internal']
            internal['conv1'] == {'weights': ['This is some Tensor'], 'bias': ['This is some vector']}

        If we naively called model.set_state(state) to load some other state from a file, then we would end up assigning a nested
        dictionary to the value of model.conv1. What we actually want is:

        ::

            model.set_state(state)
            print(model.conv1) # This is a PyTorch Module
            print(model.conv1.weights) # This is a Tensor
            print(model.conv1.bias) # This is a Tensor

        PyTorch_Component_Map checks if the attribute being assigned to is a PyTorch_Model or (PyTorch) Module and performs this type
        of assignment.

    3) 'Registered' to the Model. This is something that PyTorch does whenever you assign a value to a PyTorch Module and is essential
        for proper functioning of PyTorch methods/functions, such as getting a state_dict, submodules, etc.

    This additional logic is important, because in general, all of the layers of a Neural Network are implemented as Modules and PyTorch
    Modules inherently has a nested structure.
    """

    def __init__(self, components, model=None):
        """
        If a model is provided, then the PyTorch_Component_Map can register values with that model, which is essential for proper usage with
        PyTorch Modules.
        """
        self.model = model
        Component_Map.__init__(self, components)

    def __setitem__(self, key, val):
        """
        This method has additional logic to __setitem__ that is described above.
        """
        # This allows the Module superclass to register the parameters.
        if self.model is not None and key != 'components' and hasattr(self.model, key) and isinstance(val, dict): # If setting the state dict for a submodule.
                submodule = getattr(self.model, key)
                if hasattr(submodule, 'set_state'): # Is a Model
                    submodule.set_state(val)
                elif isinstance(submodule, Module):
                    # Convert state dict to a dict of tensors
                    val = {k:torch.Tensor(v) for k,v in val.items()}
                    submodule.load_state_dict(val)
                elif isinstance(submodule, dict): # It was supposed to be a dict
                    Component_Map.__setitem__(self, key, val)
                    if self.model is not None:
                        val = self[key]
                        i = self.model._flags['components_initialized']
                        self.model._flags['components_initialized'] = 0
                        setattr(self.model, key, val)
                        self.model._flags['components_initialized'] = i
        else:
            Component_Map.__setitem__(self, key, val)
            if self.model is not None:
                val = self[key]
                i = self.model._flags['components_initialized']
                self.model._flags['components_initialized'] = 0
                setattr(self.model, key, val)
                self.model._flags['components_initialized'] = i

    def hook(self, key, value):
        """
        This is used to (try to) convert objects to torch.nn.Parameter objects upon assignment. If a value has tensor-like structure (ie. is
        a list or ndarray), then it will automatically be converted.
        """

        if not isinstance(value, Parameter) and not isinstance(value, Module) and hasattr(value, '__len__'): # Convert to Parameter
            try:
                value = Parameter(torch.Tensor(value))
            except:
            # If the component is not a tensor-like, Parameter, or Module, then it is some other object that we simply attach to the model
            # For example, it could be a Pipe or Junction that the model can call upon.
                pass
        return key, value

    def setitem_hook(self, key, value):
        """
        This assigns the above hook to setitem_hook so that it will be triggered upon every __setitem__ call.
        """
        return self.hook(key, value)
