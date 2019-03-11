import torch
import Fireworks
from Fireworks import Message, PyTorch_Model
from Fireworks.extensions import IgniteJunction
from Fireworks.toolbox import ShufflerPipe, BatchingPipe, TensorPipe, GradientPipe
from Fireworks.toolbox.preprocessing import train_test_split, Normalizer
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

# Construct data, split into train/eval/test, and get get_minibatches
data, params = generate_data(n=1000)
train, test = train_test_split(data, test=.25)

shuffler = ShufflerPipe(train)
minibatcher = BatchingPipe(shuffler, batch_size=25)
normalizer = Normalizer(input=minibatcher)
# dataset = GradientPipe(TensorPipe(normalizer, columns=['x','y']), columns=['x','y'])
dataset = TensorPipe(normalizer, columns=['x','y'])
# Compute means and variances
normalizer.enable_updates()
for batch in normalizer:
    pass
normalizer.disable_updates()
normalizer.enable_inference()

# Construct training closure and train using ignite
base_loss = torch.nn.MSELoss()
loss = lambda batch: base_loss(batch['y_pred'], batch['y'])

model = NonlinearModel()

trainer = IgniteJunction({'model': model, 'dataset': dataset}, loss=loss, optimizer='Adam')

# Specify parameters and metrics construction. Initialize Experiment(s).
test_set = Normalizer(input=TensorPipe(test, columns=['x','y']))
test_set.set_state(normalizer.get_state())
initial_loss = loss(model(test_set[0:250]))

trainer.train(max_epochs=50)

final_loss = loss(model(test_set[0:250]))

# Specify hyperparameter optimization scheme using Factory
