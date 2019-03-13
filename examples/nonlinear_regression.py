import torch
import Fireworks
from Fireworks import Message, PyTorch_Model
from Fireworks.extensions import IgniteJunction
from Fireworks.toolbox import ShufflerPipe, BatchingPipe, TensorPipe, GradientPipe
from Fireworks.toolbox.preprocessing import train_test_split, Normalizer
from Fireworks.utils.exceptions import EndHyperparameterOptimization
from ignite.engine import Events
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sqlalchemy import Table, Column, Integer, String, JSON, create_engine
from Fireworks.extensions.database import TablePipe, create_table
from itertools import combinations

def generate_data(n=1000):

    a = randint(-10,10)
    b = randint(-10,10)
    c = randint(-10,10)
    errors = np.random.normal(0, .5, n)
    x = np.random.rand(n)*100 - 50
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
dataset = TensorPipe(normalizer, columns=['x','y'])
# Compute means and variances
# normalizer.enable_updates()
normalizer.disable_updates()
for batch in normalizer:
    pass
normalizer.disable_updates()
normalizer.enable_inference()

# Construct training closure and train using ignite
base_loss = torch.nn.MSELoss()
loss = lambda batch: base_loss(batch['y_pred'], batch['y'])

model = NonlinearModel(components={'d': [0], 'e':[0]})
# model.freeze(['d','e'])

model_state = Message.from_objects(model.get_state())
model_state['iteration'] = [0]

# engine = create_engine('sqlite:///:memory:')
# table = create_table('ok', columns=[Column('state', JSON), Column('iteration', Integer)])
# bob = TablePipe(table, engine)

trainer = IgniteJunction({'model': model, 'dataset': dataset}, loss=loss, optimizer='Adam', lr=.05)

# Specify parameters and metrics construction. Initialize Experiment(s).
test_set = Normalizer(input=TensorPipe(test, columns=['x','y']))
test_set.set_state(normalizer.get_state())

x = Message({'x':np.arange(-10,10,.2)}).to_tensors()
y_initial = model(x)['y_pred'].detach().numpy()

initial_loss = loss(model(test_set[0:250]))

trainer.train(max_epochs=50)

final_loss = loss(model(test_set[0:250]))

# Visualize functions
true_model = NonlinearModel(components={'a':[params['a']], 'b': [params['b']], 'c': [params['c']], 'd': [0], 'e': [0]})

true = true_model(x)
y_true = true['y_pred'].detach().numpy()
predicted = model(x)
y_final = predicted['y_pred'].detach().numpy()
x_base = x['x'].detach().numpy()

fig, ax = plt.subplots()
def animate(frame):

    current_state = {'internal': frame['internal'][0], 'external': {}}
    print(frame)
    model.set_state(current_state)
    print(model.get_state()['internal'])
    y_predicted = model(x)['y_pred'].detach().numpy()
    xdata = list(x_base)
    ydata = list(y_predicted)
    ax.clear()
    ax.plot(xdata, list(y_true), 'r')
    ax.plot(xdata, ydata, 'g')
    title = "Iteration: {0}".format(frame['iteration'][0])
    print(title)
    ax.set_title(title)

ani = FuncAnimation(fig, animate, trainer.model_state, interval=1000)
plt.show()


# Specify hyperparameter optimization scheme using Factory
def make_model(parameters):
    include = [letter for letter in ['a','b','c','d','e'] if letter in parameters]
    exclude = [letter for letter in ['a','b','c','d','e'] if letter not in parameters]
    for letter in exclude:
        parameters[letter] =  [0]
    model = NonlinearModel(parameters)
    for letter in exclude: # Prevent training from taking place for these parameters
        model.freeze(letter)

    return model

def get_trainer(ignite_junction):

    def train_from_params(parameters):

        model = make_model(parameters)
        trainer.components['model'] = model
        ignite_junction.train(max_epochs=50)
        return model

    return train_from_params

class Parameterizer:

    def __init__(self):
        possible_params = ['a','b','c','d','e']
        def generator():
            for i in reversed(range(5)):
                for combination in combinations(possible_params,i):
                    params = {param: [0] for param in combination}
                    yield params
        self.generator = generator()

    def __call__(self,past_params, metrics):
        try:
            return self.generator.__next__()
        except StopIteration:
            raise EndHyperparameterOptimization

# factory = LocalMemoryFactory(components={'trainer': get_trainer(trainer), 'eval_set': test', })
# Alter regularization weight, number of parameters
