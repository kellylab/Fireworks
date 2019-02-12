import Fireworks
from Fireworks import Message
from Fireworks.toolbox import ShufflerPipe, BatchingPipe, TensorPipe
import numpy as np
from random import randint

def generate_data(n=1000):

    a = randint(-10,10)
    b = randint(-10,10)
    c = randint(-10,10)
    errors = np.random.normal(0, .5, n)
    x = np.random.rand(n)*100
    y = a + b*x + c*x**2

    return Message({'x': x, 'y': y, 'errors': errors}), {'a': a, 'b': b, 'c': c}

class NonlinearModel(Model):

    required_components = ['a','b', 'c', 'd', 'e']

    def init_default_components(self):

        for letter in ['a', 'b', 'c', 'd', 'e']:
            self.components[letter] = np.random.normal(0,1)

    def forward(self, message):

        x = message[self.in_column]
        message[self.out_column] = self.a + self.b*x + self.c*x**2 + self.d*x**3 + self.e*x**4

        return message

# Construct data, split into train/eval/test, and get get_minibatcher
data, params = generate_data(n=1000)
training = data[0:800]
evaluation = data[800:900]
testing = data[900:]

shuffler = ShufflerPipe(training)
minibatcher = BatchingPipe(shuffler, batch_size=25)
# TODO: Add a normalizer here
dataset = TensorPipe(minibatcher)

# Construct training closure and train using ignite
base_loss = torch.nn.MSELoss()
loss = lambda batch: base_loss(batch['y'], batch['y_true'])

model = NonlinearModel()
trainer = IgniteJunction({'model': model, 'dataset': dataset})

# Specify parameters and metrics construction. Initialize Experiment(s).
trainer.run()

# Specify hyperparameter optimization scheme using Factory
