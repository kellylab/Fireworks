#TODO: Test the following scenarios: hardcoded params, GUI params, database params, params from a pipe, from another model, and param updating

#TODO: Test that gradients and updates behave correctly when multiple models are linked together

#TODO: Demonstrate the following applications: training and inferencing models, interactive models, auto-generated models

from Fireworks.model import Model, model_from_module
from Fireworks.exceptions import ParameterizationError
from Fireworks.pipeline import BatchingPipe, LoopingPipe, ShufflerPipe, RepeaterPipe
from Fireworks import Message
import torch
from torch.nn import Parameter
from random import randint
import numpy as np

class DummyModel(Model):
    """ Implements y = m*x + b """
    required_components = ['m', 'b']

    def init_default_components(self):
        """ Default y-intercept to 0 """
        self.b = Parameter(torch.Tensor([0.]))

    def forward(self, message):

        y = self.m*message['x']+self.b
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

loss = torch.nn.MSELoss()

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

class LinearModule(torch.nn.Module):
    """ Dummy PyTorch Module. """
    def __init__(self):
        super().__init__()
        self.m = Parameter(torch.randn(1))
        self.b = Parameter(torch.randn(1))
        self.conv1 = torch.nn.Conv2d(1, 20, 5)

    def forward(self, message):

        return Message({'x': message['x'], 'y': self.m*message['x']+self.b})

def test_Model_init():

    # Confirm that the model detects that parameters are missing.
    try:
        damura = DummyModel()
        assert False
    except ParameterizationError as error:
        assert True

    # Confirm that the model can be initialized
    damura = DummyModel({'m': [3.]})

    assert (damura.m == 3.).all()
    assert (damura.b == 0).all() # Default for 'b'

    damura = DummyModel({'m': [4.], 'b': [5.]})
    assert (damura.m == 4.).all()
    assert (damura.b == 5.).all()

def test_Model_inferencing():

    damura = DummyModel({'m': [2.]})
    x = Message({'x': torch.Tensor([1,2,3])})
    y = damura(x)
    assert y == x
    assert (y['x'] == torch.Tensor([1,2,3])).all()
    assert (y['y'] == torch.Tensor([2.,4.,6.])).all()

def test_ModelFromModule():

    pop = LinearModule()
    mom = model_from_module(LinearModule)
    messi = Message({'x':torch.Tensor([1,2,3])})
    pom = mom()
    assert set(pom.components) == set(['conv1', 'm', 'b'])
    result = pom(messi)
    assert 'y' in result and 'x' in result

def test_one_Model_training():

    A = DummyModel({'m': [0.]})
    B = model_from_module(LinearModule)
    training_data = generate_linear_model_data()
    repeater = RepeaterPipe(training_data[0])
    lol = LoopingPipe(repeater)
    assert False 
    # minibatcher = BatchingPipe(ShufflerPipe(LoopingPipe(training_data)))
    assert False

def test_multiple_Models_inferencing(): pass

def test_multiple_Models_training(): pass
