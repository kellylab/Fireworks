#TODO: Test the following scenarios: hardcoded params, GUI params, database params, params from a pipe, from another model, and param updating

#TODO: Test that gradients and updates behave correctly when multiple models are linked together

#TODO: Demonstrate the following applications: training and inferencing models, interactive models, auto-generated models

from Fireworks.model import Model
from Fireworks.exceptions import ParameterizationError
from Fireworks import Message
from torch.nn import Module
from random import randint
import numpy as np

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

def generate_linear_model_data(n=1000):
    """
    Generates n samples from a linear model with a small variability.
    """
    m = randint(-3,3)
    b = randint(-10,10)
    x = np.random.rand(n)*100
    errors = np.random.normal(0,.5,n) # Gaussian samples for errors
    y = m*x+b + errors

    return Message({'x':x, 'y':y}), {'m': m, 'b': b, 'errors': errors} # Second dict is for debugging

def train_model(model, data):

    # Initialize model for training
    # Define loss function and learning algorithm
    optimizer = optim.SGD(model.parameters(), lr=.1)
    # Training loop
    num_epochs = 3
    for epoch in num_epochs:
        for batch in data:
            optimizer.zero_grad()
            l = loss(model(data['x']), data['y'])
            l.backward()
            optimizer.step()

    return model

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

def test_one_Model_training(): pass

def test_multiple_Models_inferencing(): pass

def test_multiple_Models_training(): pass
