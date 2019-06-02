import torch
from fireworks import Message, PyTorch_Model
from fireworks.extensions import IgniteJunction
from fireworks.toolbox import ShufflerPipe, BatchingPipe, TensorPipe
from fireworks.toolbox.preprocessing import train_test_split, Normalizer

import numpy as np
from random import randint

def generate_data(n=1000):

    a = randint(-10,10)
    b = randint(-10,10)
    c = randint(-10,10)
    errors = np.random.normal(0, .15, n)
    x = np.random.rand(n) * 100 - 50
    y = a + b*x + c*x**2 + errors

    return Message({'x': x, 'y': y, 'errors': errors}), {'a': a, 'b': b, 'c': c}

# Construct data, split into train/eval/test, and get get_minibatches
def get_data(n=1000):

    data, params = generate_data(n)
    train, test = train_test_split(data, test=.25)

    shuffler = ShufflerPipe(train)
    minibatcher = BatchingPipe(shuffler, batch_size=25)
    train_set = TensorPipe(minibatcher, columns=['x','y'])

    test_set = TensorPipe(test, columns=['x','y'])

    return train_set, test_set, params

class NonlinearModel(PyTorch_Model):

    required_components = ['a','b', 'c', 'd', 'e']

    def init_default_components(self):

        for letter in ['a', 'b', 'c', 'd', 'e']:
            self.components[letter] = torch.nn.Parameter(torch.Tensor(np.random.normal(0,1,1)))

        self.in_column = 'x'
        self.out_column = 'y_pred'

    def forward(self, message):

        x = message[self.in_column]
        message[self.out_column] = (self.a + self.b*x + self.c*x**2 + self.d*x**3 + self.e*x**4)

        return message
