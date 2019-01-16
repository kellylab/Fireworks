from abc import ABC, abstractmethod, abstractproperty
import torch
from torch.nn import Module, Parameter
from abc import abstractmethod
from Fireworks.exceptions import ParameterizationError
from Fireworks.pipeline import HookedPassThroughPipe

class Model(Module, HookedPassThroughPipe, ABC):
    """
    Represents a statistical model which has a set of components, and a
    means for converting inputs into outputs. The model functions like a Pipe
    with respect to the input/output stream, and it functions like a Junction
    with respect to the parameterization. components can be provided via multiple
    different sources in this way, providing flexibility in model configuration.
    Models can also provide components for other Models, enabling one to create
    complex graphs of Models that can be trained simultaneously or individually.
    """

    def __init__(self, components={}, *args, inputs=None, **kwargs): # QUESTION: Should a model have a list of required components?
        """
        Args:
            components: A dict of components that the model can call on.
        """

        HookedPassThroughPipe.__init__(self, inputs=inputs)
        Module.__init__(self)
        # self.components = {}
        self.init_default_components()
        self.update_components(components)
        # self.components.update(components) # Overwrite and/or adds params in the argument.
        self.check_components()

        #TODO: Implement pytorch module methods to mirror pytorch style

    def init_default_components(self):
        """
        This method can optionally be implemented in order for the model to provide a default initialization for some or all of its
        required components.
        """
        pass

    def update_components(self, components):

        for key, component in components.items():
            if not isinstance(component, Parameter) and not isinstance(component, Module): # Convert to Parameter
                component = Parameter(torch.Tensor(component))
            setattr(self, key, component)

    def check_components(self, components = None):
        """
        Checks to see if the provided components dict provides all necessary params for this model to run.
        """
        if components is None:
            components = self.components
        missing_components = []
        error = False
        for key in self.required_components:
            if key not in components:
                missing_components.append(key)
                error = True
        if error:
            raise ParameterizationError("Missing required components {0}".format(missing_components))

    @property
    def components(self):
        return list(self._modules.keys()) + list(self._parameters.keys())

    @property
    def required_components(self):
        """
        This should be overridden by a subclass in order to specify components that should be provided during initialization. Otherwise,
        this will default to just return the components already present within the Model.
        """
        return self.components

    @abstractmethod
    def forward(self, message):
        """
        Represents a forward pass application of the model to an input. Must be implemented by a subclass.
        This should return a Message.
        """
        pass

    def freeze(self, components = None):
        """
        Freezes the given components of the model (or all of them if none are specified) in order to prevent gradient updates.
        """
        pass # TODO: Implement

    # TODO: Figure out model i/o
    # TODO: Implement description methods
    # TODO: Implement method to access components by key

    def __getitem_hook(self, message):

        return self.forward(message)

    def __next_hook(self, message):

        return self.forward(message)

    def __call__(self, *args, **kwargs):

        return self.forward(*args, **kwargs)

# class ModelFromModule(Model):
#     """
#     Converts a PyTorch Module into a Fireworks Model.
#     """
#     def __init__(self, module, parameters, *args, inputs=None, **kwargs):
#
#         super.__init__(self, parameters, *args, inputs=inputs, **kwargs)
#         self.module = module(*args, **parameters) # TODO: Test that these parameters stay linked.
#
#     def forward(self, message):
#
#         return self.module(message)

# class TrainableModel(Module, Model): # This should call Module supermethods before any Fireworks methods.
#
#     def __init__(self, parameters, *args, inputs=None, **kwargs):
#         super(Module, self).__init__()
#         super(Model, self).__init__()
#         for key, param in self.parameters.items():
#             setattr(self.module, key, param)

def model_from_module(module_class):
    """
    Given the class definition for a pytorch module, returns a model that
    encapsulates that module.
    """
    class ModelFromModule(module_class, Model):

        def __init__(self, components={}, *args, inputs=None, **kwargs):
            Model.__init__(self, components, inputs=inputs)
            module_class.__init__(self, *args, **kwargs)

        # def forward(self, message):
        #
        #     try:
        #         message = self.inpu

    return ModelFromModule
