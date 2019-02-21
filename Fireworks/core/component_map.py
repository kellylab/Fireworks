
class Component_Map(dict):
    """
    This is a bank.
    """

    def __init__(self, kwargs):

        dict.__init__(self)
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, val):

        if type(val) is tuple and len(val) is 2 and type(val[1]) is str and hasattr(val[0], val[1]):
            # Very specific test to check if the intention is to link an attribute inside of another
            # object to this Component Map rather than simply set the value of the key to a tuple.
            obj, attribute = val
            value = getattr(obj, attribute)
            # key, value = self.setitem_hook(key, value)
            # self._external_components[key] = value
            self._external_modules[key] = obj
            self._external_attribute_names[key] = attribute
        else:
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
            for d in [self._external_attribute_names, self._extrnal_modules, self._internal_components]:
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

    def setitem_hook(self, key, value):
        """
        This can be overridden by a subclass in order to implement specific actions
        that should take place before an attribute is set.
        """
        return key, value

    # def getitem_hook(self, value):
    #     """
    #     This can be overridden by a subclass in order to implement specific actions
    #     that should take place before an attribute is accessed.
    #     """
    #     return value

class PyTorch_Component_Map(Component_Map):

    def hook(self, key, value):

        if not isinstance(value, Parameter) and not isinstance(value, Module): # Convert to Parameter
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
