#TODO: Test the following scenarios: hardcoded params, GUI params, database params, params from a pipe, from another model, and param updating

#TODO: Test that gradients and updates behave correctly when multiple models are linked together

#TODO: Demonstrate the following applications: training and inferencing models, interactive models, auto-generated models

from Fireworks.model import Model, model_from_module
from Fireworks.exceptions import ParameterizationError
from Fireworks.pipeline import BatchingPipe, LoopingPipe, ShufflerPipe, RepeaterPipe
from Fireworks import Message, Junction
import random
import torch
from torch.nn import Parameter
from random import randint
import numpy as np

loss = torch.nn.MSELoss()

class DummyModel(Model):
    """ Implements y = m*x + b """
    required_components = ['m', 'b']

    def __init__(self, components = {}, input = None, in_column = 'x', out_column = 'y'):
        Model.__init__(self, components, input = input)
        self.in_column = in_column
        self.out_column = out_column

    def init_default_components(self):
        """ Default y-intercept to 0 """
        self.b = Parameter(torch.Tensor([0.]))

    def forward(self, message):

        y = self.m*message[self.in_column]+self.b
        message[self.out_column] = y

        return message

class DummyMultilinearModel(Model):
    """ Implements y = m1*x1 +m2*x1 + b """
    required_components = ['m1', 'm2', 'b']

    def init_default_components(self):
        """ Default y-intercept to 0 """
        self.b = Parameter(torch.Tensor([0.]))

    def forward(self, message):

        y = self.m1*message['x1']+ self.m2*message['x2'] + self.b
        message['y'] = y
        return message

class LinearJunctionModel(Model):
    """ Implements y = f(x) + b, where the function f(x) is provided as a junction input. """

    required_components = ['b', 'f']

    def init_default_components(self):
        """ Default y-intercept to 0 """
        self.b = Parameter(torch.Tensor([0.]))

    def forward(self, message):

        y = self.f(message)['z'] + self.b
        message['y'] = y
        return message

class LinearModule(torch.nn.Module):
    """ Dummy PyTorch Module. """
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.m = Parameter(torch.randn(1))
        self.b = Parameter(torch.randn(1))
        self.conv1 = torch.nn.Conv2d(1, 20, 5)

    def forward(self, message):

        message['y'] = self.m*message['x']+self.b

        return message

class RandomJunction(Junction):

    def __call__(self, *args, **kwargs):

        target = random.sample(self.components.keys(),1)[0]
        return self.components[target](*args, **kwargs)

def generate_linear_model_data(n=300):
    """
    Generates n samples from a linear model with a small variability.
    """
    m = randint(-3,3)
    b = randint(-10,10)
    x = np.random.rand(n)*100
    errors = np.random.normal(0, .4, n) # Gaussian samples for errors
    y = m*x+b + errors

    return Message({'x':x, 'y_true':y}), {'m': m, 'b': b, 'errors': errors} # Second dict is for debugging

def generate_multilinear_model_data(n=550):
    """
    Generates n samples from a multilinear model y = m1*x1 + m2*x2 +b with a small variability.
    """

    m1 = randint(-5,5)
    m2 = randint(-5,5)
    b = randint(-10,10)
    x1 = np.random.rand(n)*100
    x2 = np.random.rand(n)*100
    errors = np.random.normal(0,.1,n) # Gaussian samples for errors
    y = m1*x1 + m2*x2 + b + errors

    return Message({'x1':x1, 'x2': x2, 'y_true':y}), {'m1': m1, 'm2':m2, 'b': b, 'errors': errors} # Second dict is for debugging

def train_model(model, data, models = None, predicted='y', label='y_true'):

    # Initialize model for training
    # Define loss function and learning algorithm
    models = models or [model]
    parameters = [filter(lambda p: p.requires_grad, m.parameters()) for m in models]
    parameters = [x for y in parameters for x in y]
    optimizer = torch.optim.SGD(parameters, lr=.00015)
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        for batch in data:
            optimizer.zero_grad()
            result = model(batch.to_tensors())
            lo = loss(result[predicted], result[label])
            lo.backward()
            optimizer.step()

    return model

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
    y = result['y']
    x = result['x']
    assert 'y' in result and 'x' in result and 'z' not in result
    # Test that Model calls are recursive on inputs
    hom = DummyModel({'m': [3.]}, out_column='z')
    pom.input = hom
    result = pom(messi)
    assert (x == result['x']).all()
    assert (y == result['y']).all()
    assert 'y' in result and 'x' in result and 'z' in result

def test_freeze_and_unfreeze():

    A = DummyModel({'m': [0.]})
    assert A.m.requires_grad == True
    assert A.b.requires_grad == True
    A.freeze()
    assert A.m.requires_grad == False
    assert A.b.requires_grad == False
    A.unfreeze()
    assert A.m.requires_grad == True
    assert A.b.requires_grad == True
    A.freeze('m')
    assert A.m.requires_grad == False
    assert A.b.requires_grad == True
    A.unfreeze('m')
    assert A.m.requires_grad == True
    assert A.b.requires_grad == True
    A.freeze('b')
    assert A.m.requires_grad == True
    assert A.b.requires_grad == False
    A.unfreeze('b')
    assert A.m.requires_grad == True
    assert A.b.requires_grad == True

    B = model_from_module(LinearModule)() # This one has a PyTorch module Conv1 in it as well.
    assert B.m.requires_grad == True
    assert B.b.requires_grad == True
    assert B.conv1.weight.requires_grad == True
    assert B.conv1.weight.requires_grad == True
    B.freeze()
    assert B.m.requires_grad == False
    assert B.b.requires_grad == False
    assert B.conv1.weight.requires_grad == False
    assert B.conv1.weight.requires_grad == False
    B.unfreeze('conv1')
    assert B.m.requires_grad == False
    assert B.b.requires_grad == False
    assert B.conv1.weight.requires_grad == True
    assert B.conv1.weight.requires_grad == True
    B.freeze('conv1')
    assert B.m.requires_grad == False
    assert B.b.requires_grad == False
    assert B.conv1.weight.requires_grad == False
    assert B.conv1.weight.requires_grad == False
    B.unfreeze(['m','b'])
    assert B.m.requires_grad == True
    assert B.b.requires_grad == True
    assert B.conv1.weight.requires_grad == False
    assert B.conv1.weight.requires_grad == False

def get_minibatcher(training_data):

    repeater = RepeaterPipe(training_data)
    lol = LoopingPipe(repeater)
    shuffler = ShufflerPipe(lol)
    minibatcher = BatchingPipe(shuffler, batch_size=50)

    return minibatcher

def test_one_Model_training():

    A = DummyModel({'m': [0.]})
    B = model_from_module(LinearModule)()
    training_data = generate_linear_model_data()
    m = training_data[1]['m']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    train_model(A, minibatcher)
    # For some reason, this model struggles to learn the y-intercept.
    assert (m-A.m < .4).all()
    train_model(B, minibatcher)
    assert (m - B.m < .4).all()

    assert (A.m - B.m < .4).all() # Test precision between models

def test_multiple_Models_training():
    """
    Here, we compose a model multilinear = A + B + C, where A, B, and C represent
    different components of the overall model.
    """
    A = DummyModel({'m': [0.]})
    A.freeze('b')
    B = model_from_module(LinearModule)()
    B.freeze('b')
    C = DummyModel({'m': [0.]})
    C.freeze('m')
    multilinear = DummyMultilinearModel({'m1':A.m, 'm2': B.m, 'b': C.b})
    assert multilinear.m1 is A.m
    assert multilinear.m2 is B.m
    assert multilinear.b is C.b
    training_data = generate_multilinear_model_data()
    m1 = training_data[1]['m1']
    m2 = training_data[1]['m2']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    train_model(multilinear, minibatcher)
    assert (A.m - m1 < .4).all()
    assert (B.m - m2 < .4).all()
    assert (A.b == 0).all()
    assert (C.m == 0.).all()

def test_multiple_Models_training_in_pipeline():
    """
    Here, model A pipes its output into B
    """
    A = DummyModel({'m': [3.]}, out_column='y1')
    B = DummyModel({'m': [1.], 'b': [2.]}, input=A, in_column='y1', out_column='y')
    A.freeze('b')
    B.freeze('m')
    training_data = generate_linear_model_data()
    m = training_data[1]['m']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    assert (A.m == 3.).all()
    assert (B.m == 1).all()
    assert (A.b == 0).all()
    assert (B.b == 2.).all()
    train_model(B, minibatcher, models = [B])
    assert (A.m - m < .4).all()
    assert (B.b != 2).all()
    assert (B.m == 1).all()
    assert (A.b == 0).all()

def test_multiple_Models_training_in_junction():
    """
    Here, model A is provided as a component of B
    """
    A = DummyModel({'m': [1.],'b':[3.]}, out_column='z')
    B = LinearJunctionModel(components={'b':[.5], 'f':A})
    training_data = generate_linear_model_data()
    m = training_data[1]['m']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    batch = minibatcher.__next__()
    batch.to_tensors()
    B(batch)
    assert (A.m == 1.).all()
    train_model(B, minibatcher, models=[A, B])
    assert (A.m != 1).all()

def test_multiple_Models_training_via_junction():
    """
    Here, model B takes a junction A as input that randomly calls one of C, D, or E
    Hence, B implements y = f(x) + b, where f is randomly C, D, or E
    """
    C = DummyModel({'m': [1.],'b':[0.]}, out_column='z')
    D = DummyModel({'m': [2.],'b':[0.]}, out_column='z')
    E = DummyModel({'m': [3.],'b':[0.]}, out_column='z')
    C.freeze('b')
    D.freeze('b')
    E.freeze('b')
    A = RandomJunction(components={'C':C, 'D': D, 'E':E})
    B = LinearJunctionModel(components={'b': [3.], 'f':A})
    training_data = generate_linear_model_data(n=750)
    m = training_data[1]['m']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    batch = minibatcher.__next__()
    batch.to_tensors()
    banana = B(batch)['y']
    rambo = False
    for i in range(20):
        bonana = B(batch)['y']
        if (banana != bonana).all():
            rambo = True
            break
    assert rambo
    train_model(B, minibatcher, models=[B, C, D, E])
    # Test that all Models trained
    assert (C.m - m < .4).all()
    assert (D.m - m < .4).all()
    assert (E.m - m < .4).all()
    assert (C.m != D.m).all()
    assert (D.m != E.m).all()
    assert (E.m != C.m).all()
