
@abstractclass
class Model(HookedPassThroughPipe):
    """
    Represents a statistical model which has a set of parameters, and a
    means for converting inputs into outputs. The model functions like a Pipe
    with respect to the input/output stream, and it functions like a Junction
    with respect to the parameterization. Parameters can be provided via multiple
    different sources in this way, providing flexibility in model configuration.
    Models can also provide parameters for other Models, enabling one to create
    complex graphs of Models that can be trained simultaneously or individually.
    """

def __init__(self, parameters, *args, inputs=None, **kwargs):
    """
    Args:
        parameters: A dict of parameters that the model can call on.
    """

    self.parameters = parameters

@abstractmethod
def forward(self, message):
    """
    Represents a forward pass application of the model to an input. Must be implemented by a subclass.
    """
    pass

def __getitem_hook(self, message):

    return self.forward(message)

def __next_hook(self, message):

    return self.forward(message)

class ModelFromModule(Model):
    """
    Converts a PyTorch Module into a Fireworks Model.
    """
    def __init__(self, module, parameters, *args, inputs=None, **kwargs):

        super.__init__(self, parameters, *args, inputs=inputs, **kwargs)
        self.module = module(*args, **parameters) # TODO: Test that these parameters stay linked.

    def forward(self, message)

        return self.module(message)
