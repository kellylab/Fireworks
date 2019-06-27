import torch
from copy import deepcopy
from fireworks import Message
from fireworks.extensions import IgniteJunction
from fireworks.utils.exceptions import EndHyperparameterOptimization
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from fireworks.extensions import LocalMemoryFactory, Experiment
from fireworks.extensions.training import default_training_closure, default_evaluation_closure
from itertools import combinations, count

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from nonlinear_regression_utils import NonlinearModel, get_data
from nonlinear_regression import base_loss, ModelSaverMetric

train_set, test_set, params = get_data(n=1000)
loss = lambda batch: base_loss(batch['y_pred'], batch['y'])

# Specify hyperparameter optimization scheme using Factory
def make_model(parameters):
    temp_parameters = deepcopy(parameters)
    include = [letter for letter in ['a','b','c','d','e'] if letter in parameters]
    exclude = [letter for letter in ['a','b','c','d','e'] if letter not in parameters]
    for letter in exclude:
        temp_parameters[letter] =  [0]
    model = NonlinearModel(temp_parameters)
    for letter in exclude: # Prevent training from taking place for these parameters
        model.freeze(letter)
    return model

def get_trainer(train_set, loss, optimizer, **kwargs):

    def train_from_params(parameters):

        model = make_model(parameters)
        trainer = IgniteJunction(components={'model': model, 'dataset': train_set}, loss=loss, optimizer=optimizer, visdom=False, **kwargs)
        print("Now training model for parameters {0}".format(parameters))
        trainer.train(max_epochs=10)
        evaluator = IgniteJunction(components={'model': model, 'dataset': train_set}, loss=loss, optimizer=optimizer, update_function=default_evaluation_closure, visdom=False, **kwargs)
        print("Now evaluating trained model.")
        return trainer

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
            params = self.generator.__next__()
            if params == {}:
                raise EndHyperparameterOptimization
            return params

        except StopIteration:
            raise EndHyperparameterOptimization

class AccuracyMetric(Metric):

    def __init__(self, output_transform = lambda x:x):
        Metric.__init__(self, output_transform=output_transform)
        self.reset()

    def reset(self):
        self.l2 = 0.
        self.num_examples = 0

    def update(self, output):
        self.l2 += output['loss']
        self.num_examples += len(output['output'])

    def compute(self):

        if self.num_examples == 0:
            raise NotComputableError(
                "Metric must have at least one example before it can be computed."
            )
        return Message({'average-loss': [self.l2 / self.num_examples]}).to_dataframe()

if __name__=="__main__":

    description = "In this experiment, we will compare the performance of different polynomial models when regressed against data generated from a random polynomial."
    experiment = Experiment("model_selection", description=description)

    factory = LocalMemoryFactory(components={
        'trainer': get_trainer(train_set, loss, optimizer='Adam', lr=.1),
        'eval_set': test_set,
        'parameterizer': Parameterizer(),
        'metrics': {'accuracy': AccuracyMetric(), 'model_state': ModelSaverMetric()}
        })

    factory.run()

    # Plot the different runs
    fig, ax = plt.subplots()
    model = NonlinearModel()
    true_model = NonlinearModel(components={'a':[params['a']], 'b': [params['b']], 'c': [params['c']], 'd': [0], 'e': [0]})
    x = Message({'x':np.arange(-10,10,.2)}).to_tensors()
    y_true = true_model(x)['y_pred'].detach().numpy()

    def animate(frame):

        current_state = {'internal': frame[1]['internal'][0], 'external': {}}
        model.set_state(current_state)

        y_predicted = model(x)['y_pred'].detach().numpy()
        xdata = list(x['x'].detach().numpy())
        ydata = list(y_predicted)
        ax.clear()
        ax.plot(xdata, list(y_true), 'r')
        ax.plot(xdata, ydata, 'g')

        # Set up titles
        substrings = []
        if (frame[0]['a'] == 0).all():
            substrings.append("a")
        if (frame[0]['b'] == 0).all():
            substrings.append("bx")
        for key, i in zip(['c','d','e'], count()):
            if (frame[0][key] == 0).all():
                substrings.append("{0}x^{1}".format(key,i+2))
        title = "Model: {0}".format(" + ".join(substrings))

        ax.set_title(title)

    params, metrics = factory.read()
    accuracy_file = experiment.open('accuracy.csv', string_only=True)
    metrics['accuracy'].to('csv', path=accuracy_file)
    model_state_file = experiment.open('model_states.csv', string_only=True)
    metrics['model_state'].to('csv', path=model_state_file)
    params_file = experiment.open('params.csv', string_only=True)
    params.to('csv', path=params_file)

    ani = FuncAnimation(fig, animate, zip(factory.params, factory.computed_metrics['model_state']), interval=3000)
    ani.save(experiment.open("models.mp4", string_only=True)) # This will only work if you have ffmpeg installed.
    plt.show()
