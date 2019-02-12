from abc import ABC, abstractmethod, abstractproperty
import torch
from torch.nn import Module, Parameter
from abc import abstractmethod
from Fireworks.utils.exceptions import ParameterizationError
from .message import Message
from .junction import Junction
from .pipe import HookedPassThroughPipe
import os
import pandas as pd

class Model(Module, HookedPassThroughPipe, Junction, ABC):
    """
    Represents a statistical model which has a set of components, and a
    means for converting inputs into outputs. The model functions like a Pipe
    with respect to the input/output stream, and it functions like a Junction
    with respect to the parameterization. components can be provided via multiple
    different sources in this way, providing flexibility in model configuration.
    Models can also provide components for other Models, enabling one to create
    complex graphs of Models that can be trained simultaneously or individually.
    """

    # TODO: Add methods to move model components to and from GPU

    def __init__(self, components = {}, *args, input = None, skip_module_init=False, **kwargs):
        """
        Args:
            components: A dict of components that the model can call on.
        """
        self._flags = {'recursive_get': 1, 'components_initialized': 0} # Used for controlling recursion. Don't mess with this.
        self.components = {}

        if not skip_module_init: # This is so the ModelFromModule Class can work.
            Module.__init__(self)
        HookedPassThroughPipe.__init__(self, input = input)

        self.init_default_components()
        self.update_components(components)
        self.update_components()
        # self.components.update(components) # Overwrite and/or adds params in the argument.
        self.check_components()
        self.update_hook = self.update
        self.forward_hook = self.forward
        self._flags['components_initialized'] = 1

    def init_default_components(self):
        """
        This method can optionally be implemented in order for the model to provide a default initialization for some or all of its
        required components.
        """
        pass

    def update_components(self, components = None):

        self._flags['components_initialized'] = 0
        self.components = {**self.components, **self._modules, **self._parameters}

        if components is None:
            components = self.components

        for key, component in components.items():
            if not isinstance(component, Parameter) and not isinstance(component, Module): # Convert to Parameter
                try:
                    component = Parameter(torch.Tensor(component))
                except: # If the component is not a tensor-like, Parameter, or Module, then it is some other object that we simply attach to the model
                # For example, it could be a Pipe or Junction that the model can call upon.
                    pass
            self.components[key] = component
            setattr(self, key, component)

        self.components = {**self.components, **self._modules, **self._parameters}
        self._flags['components_initialized'] = 1

    @abstractmethod
    def forward(self, message):
        """
        Represents a forward pass application of the model to an input. Must be implemented by a subclass.
        This should return a Message.
        """
        pass

    def save(self, *args, method='json', **kwargs):
        """
        Aggregates this model's components into a single Message and saves them using the chosen method.
        You can use any method that Messages support writing to via the to_ method, and you can
        provide additional key-word arguments as needed to support this.
        If the save method involves a path, then the path will be modified for each component and the state_dict.
        For the state_dict, the path will be torch_{name}-{path}, and for each component it will be {key}_{path}
        where name is either the name of the class or self.__name__ if it is defined. key is the value of the key in
        the components dictionary.
        """
        # Get name of Model
        if hasattr(self, '__name__'):
            name = self.__name__
        else:
            name = type(self).__name__

        # Parse save path if provided
        if 'path' in kwargs:
            kwargs = kwargs.copy()
            path = kwargs['path']
            paths = path.split('/')
            paths[-1]="{0}-{1}".format(name, paths[-1])

        # Get params
        pytorch_parameters = {
            key: value.clone().detach() for key, value in self.components.items() if
            isinstance(value, torch.nn.Parameter) or isinstance(value, torch.Tensor)
            }
        pytorch_modules = [
            {key: value.state_dict()} for key, value in self.components.items() if
            isinstance(value, torch.nn.Module) and not isinstance(value, Model)
            ]
        pytorch_modules = {
            "{0}.{1}".format(outerkey, innerkey): value.clone().detach()
            for outerdict in pytorch_modules
            for outerkey, innerdict in outerdict.items()
            for innerkey, value in innerdict.items()
            }
        pytorch_components = {**pytorch_parameters, **pytorch_modules}
        # pytorch_components = self.state_dict() # Modules and parameters

        # other_keys = self.components.keys() - self._parameters.keys()
        # other_components = {key: self.components[key] for key in other_keys}
        other_components = {
            key: value for key, value in self.components.items() if
            not isinstance(value, torch.nn.Parameter) and
            not isinstance(value, torch.Tensor) and
            not isinstance(value, torch.nn.Module) or
            isinstance(value, Model)
        }

        # Save pytorch state_dict
        if 'path' in kwargs:
            kwargs['path'] = os.path.join(*paths[:-1], 'torch_'+paths[-1])

        pytorch_as_message = Message.from_objects(pytorch_components)
        pytorch_as_message.to(method=method, **kwargs)

        # Save other components. This should recursively trigger the same action on the components if necesssary.
        for key, component in other_components.items():
            if hasattr(component, 'save'):
                if 'path' in kwargs:
                    kwargs['path'] = os.path.join(*paths[:-1], key+'_'+paths[-1])
                component.save(method=method, **kwargs)

        # Save inputs. This is done by default and can be disabled by providing 'recursive=False' as an argument.
        if ('recursive' in kwargs and kwargs['recursive']) or 'recursive' not in kwargs:
            try:
                self.recursive_call('save',method=method, **kwargs)
            except: #TODO: Make a custom RecursionEnd error to indicate that the recursion has ended in order to provide tighter error checking.
                pass

    def load(self, *args, **kwargs):

        # Determine file access scheme for inputs/components

        # Load state_dict for this object

        # Recurse

        pass

    def load_state_dict(self, *args, method='json', **kwargs):
        """
        Loads the data in the given save file into the state dict.
        """
        if 'path' in kwargs:
            kwargs = kwargs.copy()
            kwargs['filepath'] = kwargs['path']
            del kwargs['path']

        methods = {
            'json': pd.read_json,
            'pickle': pd.read_pickle,
            'csv': pd.read_csv,
        }

        df = methods[method](*args, **kwargs) # Load parameters in
        state_dict = dict(Message(df).to_tensors()) # Convert to tensors
        state_dict = {key: value.reshape(*value.shape[1:]) for key, value in state_dict.items()} # Unflatten
        Module.load_state_dict(self, state_dict)

    def update(self, batch, method=None): pass

    def compile(self): pass

    def _change_temperature(self, boo, components = None):

        self.update_components() # This is here so that the model checks to see if Parameters/Modules have changed before freezing/unfreezing

        if isinstance(components, str): # eg. a single component is provided as a string
            components = [components]

        if components is None:
            components = self.components

        for component in components:
            if isinstance(getattr(self, component), Parameter):
                getattr(self, component).requires_grad = boo
            elif isinstance(getattr(self,component), Model):
                getattr(self,component)._change_temperature(boo) # Recursively freezes Models
            elif isinstance(getattr(self,component), Module): # Is a PyTorch module but not a model.
                _change_temperature(boo, getattr(self,component))

    def all_parameters(self):
        """
        Returns a list of every parameter that this Model depends on that is unfrozen. This is useful for providing a parameters list to
        an optimizer.
        """
        all_params = []
        # Get parameters
        all_params.extend([param for param in self._parameters.values() if param.requires_grad])
        # Get submodules
        all_params.extend([list(module.parameters()) for module in self._modules.values() if module is not self and not isinstance(module, Model)]) # TODO: Test this
        # Get components
        for component in self.components.values():
            try:
                all_params.extend(component.all_parameters())
            except AttributeError:
                pass
        # Get from input
        try:
            all_params.extend(self.input.all_parameters())
        except AttributeError:
            pass

        return all_params

    def freeze(self, components = None):
        """
        Freezes the given components of the model (or all of them if none are specified) in order to prevent gradient updates.
        This means setting requires_grad to false for specified components so that these components
        are not updated during training.
        """
        self._change_temperature(False, components)

    def unfreeze(self, components = None):
        """
        Unfreezes the given components of the model (or all of them if none are specified) in order to prevent gradient updates.
        This means setting requires_grad to true for specified components so that these components
        are updated during training.
        """
        self._change_temperature(True, components)

    # TODO: Figure out model i/o
    # TODO: Implement description methods

    def _getitem_hook(self, message):

        self.update_hook(message, method='get')
        return self.forward_hook(message)

    def _next_hook(self, message):

        self.update_hook(message, method='next')
        return self.forward_hook(message)

    def _call_hook(self, message, *args, **kwargs):

        self.update_hook(message, method='call')
        return self.forward_hook(message, *args, **kwargs)
        # try: # This will trigger a recursive call if possible.
        #     message = self.recursive_call('__call__')(message, *args, **kwargs)
        # except:
        #     pass

        # return self.forward(message, *args, **kwargs)

    def __call__(self, *args, **kwargs):

        return HookedPassThroughPipe.__call__(self, *args, **kwargs)

    def __getattr__(self, name):

    #     return self._recursed_getattr(name)
    #
    # def _recursed_getattr(self, name):

        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]

        if not name.startswith('_') and self._flags['recursive_get']:
            try:
                return HookedPassThroughPipe.__getattr__(self, name)
            except:
                pass

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        """
        Attribute modifications ignore the recursive aspect of Pipes.
        """

        if name in ['_flags', 'input']:
            object.__setattr__(self, name, value)
        else:
            self._flags['recursive_get'] = 0
            Module.__setattr__(self, name, value)
            self._flags['recursive_get'] = 1
            if self._flags['components_initialized']:
                self.update_components()

    def enable_inference(self):
        self.forward_hook = self.forward
        try:
            self.recursive_call('enable_inference')
        except AttributeError:
            pass

    def disable_inference(self): #TODO: test this
        self.forward_hook = identity
        try:
            self.recursive_call('disable_inference')
        except AttributeError:
            pass

    def enable_updates(self):
        self.update_hook = self.update
        try:
            self.recursive_call('enable_updates')
        except AttributeError:
            pass

    def disable_updates(self):
        self.update_hook = identity
        try:
            self.recursive_call('disable_updates')
        except AttributeError:
            pass

def identity(*args, **kwargs):

    return args, kwargs

def freeze_module(module, parameters = None, submodules = None):
    """
    Recursively freezes the parameters in a PyTorch module.
    """
    _change_temperature(False, module, parameters, submodules)

def unfreeze_module(module, parameters = None, submodules = None):
    """
    Recursively unfreezes the parameters in a PyTorch module.
    """
    _change_temperature(True, module, parameters, submodules)

def _change_temperature(boo, module, parameters = None, submodules = None):
    """
    Changes the temperature of a PyTorch module.
    """
    parameters = parameters or module.parameters()
    submodules = submodules or module.modules()

    for parameter in parameters:
        parameter.requires_grad = boo
    for submodule in submodules:
        if submodule is not module:
            change_temperature(boo, submodule)

def model_from_module(module_class):
    """
    Given the class definition for a pytorch module, returns a model that
    encapsulates that module.
    """
    class ModelFromModule(module_class, Model):

        def __init__(self, components={}, *args, input=None, **kwargs):
            self._flags = {'recursive_get': 1, 'components_initialized': 0}
            module_class.__init__(self, *args, **kwargs)
            Model.__init__(self, components, input=input, skip_module_init=True)

        def __call__(self, message, *args, **kwargs):
            try: # This will trigger a recursive call if possible.
                message = self.input(message, *args, **kwargs)
            except:
                pass

            return self.forward(message, *args, **kwargs)

    return ModelFromModule
