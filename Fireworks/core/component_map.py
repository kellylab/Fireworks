import torch
from .message import Message
from torch.nn import Parameter, Module

class Component_Map(dict):
    """
    This is a bank.
    """

    def __init__(self, components):

        dict.__init__(self)
        self._external_modules = {}
        self._external_attribute_names = {}
        self._internal_components = {}

        for key, value in components.items():
            self[key] = value

    def __setitem__(self, key, val):

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
        Also deletes the key from the the internal and external dicts.
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
        If the key refers to an attribute of another Model, then returns that value.
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

        for key, value in {**state['internal'], **state['external']}.items():
            self[key] = value

    def get_state(self):

        internal = self._internal_components
        external = {key: (self._external_modules[key], self._external_attribute_names[key]) for key in self._external_modules}

        return {'internal': internal, 'external': external}

    # def getitem_hook(self, value):
    #     """
    #     This can be overridden by a subclass in order to implement specific actions
    #     that should take place before an attribute is accessed.
    #     """
    #     return value

class PyTorch_Component_Map(Component_Map):

    def __init__(self, components, model=None):
        self.model = model
        Component_Map.__init__(self, components)

    def __setitem__(self, key, val):

        # This allows the Module superclass to register the parameters.
        if self.model is not None and key != 'components' and hasattr(self.model, key) and isinstance(val, dict): # If setting the state dict for a submodule.
                submodule = getattr(self.model, key)
                if hasattr(submodule, 'set_state'): # Is a Model
                    submodule.set_state(val)
                elif isinstance(submodule, Module):
                    # Convert state dict to a dict of tensors
                    val = Message(val).to_tensors()
                    submodule.load_state_dict({key: val[key] for key in val.keys()})
        else:
            Component_Map.__setitem__(self, key, val)
            if self.model is not None:
                val = self[key]
                i = self.model._flags['components_initialized']
                self.model._flags['components_initialized'] = 0
                setattr(self.model, key, val)
                self.model._flags['components_initialized'] = i

    def hook(self, key, value):

        if not isinstance(value, Parameter) and not isinstance(value, Module) and hasattr(value, '__len__'): # Convert to Parameter
            try:
                value = Parameter(torch.Tensor(value))
            except:
            # If the component is not a tensor-like, Parameter, or Module, then it is some other object that we simply attach to the model
            # For example, it could be a Pipe or Junction that the model can call upon.
                pass
        return key, value

    def setitem_hook(self, key, value):
        return self.hook(key, value)

    # def getitem_hook(self, value):
    #     return self.hook('_', value)[1]
