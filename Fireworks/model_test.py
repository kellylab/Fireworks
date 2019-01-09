#TODO: Test the following scenarios: hardcoded params, GUI params, database params, params from a pipe, from another model, and param updating

#TODO: Test that gradients and updates behave correctly when multiple models are linked together

#TODO: Demonstrate the following applications: training and inferencing models, interactive models, auto-generated models

from Fireworks.model import Model
from Fireworks.exceptions import ParameterizationError
from Fireworks import Message
from torch.nn import Module

class DummyModel(Model):
    """ Implements y = m*x + b """
    required_parameters = ['m', 'b']

    def init_default_parameters(self):
        """ Default y-intercept to 0 """
        self.parameters['b'] = 0

    def forward(self, message):

        m = self.parameters['m']
        b = self.parameters['b']
        y = m*message['x']+b
        message['y'] = y
        return message

class porcupine(Module):
    """ Dummy PyTorch Module. """
    def __init__(self, input_dim, hidden_dim, output_dim):
        pass

def test_Model_init():

    # Confirm that the model detects that parameters are missing.
    try:
        damura = DummyModel()
        assert False
    except ParameterizationError as error:
        assert True

    # Confirm that the model can be initialized
    damura = DummyModel({'m': 3.})
    assert damura.parameters['m'] == 3.
    assert damura.parameters['b'] == 0 # Default for 'b'

    damura = DummyModel({'m':4., 'b':5.})
    assert damura.parameters['m'] == 4.
    assert damura.parameters['b'] == 5.

def test_Model_inferencing():

    damura = DummyModel(parameters={'m': 2.})
    x = Message({'x':[1,2,3]})
    y = damura(x)
    assert y == x
    assert (y['x'] == [1,2,3]).all()
    assert (y['y'] == [2.,4.,6.]).all()

def test_ModelFromModule(): pass
