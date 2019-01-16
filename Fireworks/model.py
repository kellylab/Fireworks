from abc import ABC, abstractmethod, abstractproperty
import torch
from torch.nn import Module
from abc import abstractmethod
from Fireworks.exceptions import ParameterizationError
from Fireworks.pipeline import HookedPassThroughPipe

class Model(HookedPassThroughPipe, ABC):
    """
    Represents a statistical model which has a set of parameters, and a
    means for converting inputs into outputs. The model functions like a Pipe
    with respect to the input/output stream, and it functions like a Junction
    with respect to the parameterization. Parameters can be provided via multiple
    different sources in this way, providing flexibility in model configuration.
    Models can also provide parameters for other Models, enabling one to create
    complex graphs of Models that can be trained simultaneously or individually.
    """

    def __init__(self, parameters={}, *args, inputs=None, **kwargs): # QUESTION: Should a model have a list of required parameters?
        """
        Args:
            parameters: A dict of parameters that the model can call on.
        """

        self.parameters = {}
        self.init_default_parameters()
        self.parameters.update(parameters) # Overwrite and/or adds params in the argument.
        self.check_parameters()

        #TODO: Implement pytorch module methods to mirror pytorch style

    def init_default_parameters(self):
        """
        This method can optionally be implemented in order for the model to provide a default initialization for some or all of its
        required parameters.
        """
        pass

    def check_parameters(self, parameters = None):
        """
        Checks to see if the provided parameters dict provides all necessary params for this model to run.
        """
        if parameters is None:
            parameters = self.parameters
        missing_params = []
        error = False
        for key in self.required_parameters:
            if key not in parameters:
                missing_params.append(key)
                error = True
        if error:
            raise ParameterizationError("Missing required parameters {0}".format(missing_params))

    @property
    def required_parameters(self): return {}

    @abstractmethod
    def forward(self, message):
        """
        Represents a forward pass application of the model to an input. Must be implemented by a subclass.
        This should return a Message.
        """
        pass

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

class TrainableModel(Module, Model): # This should call Module supermethods before any Fireworks methods.

    def __init__(self, parameters, *args, inputs=None, **kwargs):
        super(Module, self).__init__()
        super(Model, self).__init__()
        for key, param in self.parameters.items():
            setattr(self.module, key, param)

def model_from_module(module_class):
    """
    Given the class definition for a pytorch module, returns a model that
    encapsulates that module.
    """
    class TrainableModelFromModule(module_class, TrainableModel):

        def __init__(self, parameters, *args, inputs=None, **kwargs):
            # self.required_parameters = [] # QUESTION: How can we infer this?
            module_class.__init__(self, *args, **kwargs)
            TrainableModel.__init__(self, parameters, inputs=inputs)

    return TrainableModelFromModule
