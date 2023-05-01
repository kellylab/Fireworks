from fireworks.core import Junction, Model, Message
from fireworks.utils import subset_dict
import torch
from torch import optim
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss
import visdom
import datetime
import numpy as np
import types
import logging
from copy import deepcopy

logging.getLogger(name=__name__)

def default_training_closure(model, optimizer, loss_fn):
    """
    This function produces a simple training loop that can be used for many situations. During each loop, the model is applied to the
    current batch, the loss_fn computes a loss, gradients are computed on the loss, and the optimizer updates its parameters using those
    gradients. The update function returns a dictionary containing the loss, the current value of the optimizer (ie. the gradients), and
    the output of the model.
    """
    def update_function(engine, batch):

        model.train()
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()

        return {'loss': loss.detach().cpu().numpy(), 'optimizer': optimizer.state_dict(), 'state': model.get_state(), 'output': output}

    return update_function

def default_evaluation_closure(model, optimizer, loss_fn):
    """
    This function produces an evaluation loop that evaluates the model on the input data and computes the loss function but does not update
    the model. The optimizer is ignored.
    """
    def eval_function(engine, batch):
        output = model(batch)
        loss = loss_fn(output)

        return {'loss': loss.detach().cpu().numpy(), 'state': model.get_state(), 'output': output}

    return eval_function

class IgniteJunction(Junction):
    """
    This abstracts the functionality of an Ignite Engine into Junction format. See the Ignite documentation (https://github.com/pytorch/ignite)
    for more details.
    """
    required_components = {'model': Model, 'dataset': object}

    # These dictionaries describe the allowed optimizers and learning rate schedulers, along with the keyword arguments that each can accept.
    # These are all the optimizers/schedulers that PyTorch includes.

    optimizers = {
        'Adadelta':  optim.Adadelta,
        'Adagrad': optim.Adagrad,
        'Adam': optim.Adam,
        'SparseAdam': optim.SparseAdam,
        'Adamax': optim.Adamax,
        'ASGD': optim.ASGD,
        'LBFGS': optim.LBFGS,
        'RMSprop': optim.RMSprop,
        'Rprop': optim.Rprop,
        'SGD': optim.SGD,
    }
    optimizer_kwargs = {
        'Adadelta':  ['lr', 'rho', 'eps', 'weight_decay'],
        'Adagrad': ['lr', 'lr_decay', 'weight_decay', 'initial_accumulator_value'],
        'Adam': ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'],
        'SparseAdam': ['lr', 'betas', 'eps'],
        'Adamax': ['lr', 'betas', 'eps', 'weight_decay'],
        'ASGD': ['lr', 'lambd', 'alpha', 't0', 'weight_decay'],
        'LBFGS': ['lr', 'max_iter', 'max_eval', 'tolerance_grad', 'tolerance_change', 'history_size', 'line_search_fn'],
        'RMSprop': ['lr', 'alpha', 'eps', 'weight_decay', 'momentum', 'centered'],
        'Rprop': ['lr', 'etas', 'step_sizes'],
        'SGD': ['lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'],
    }
    schedulers = {
        'LambdaLR': optim.lr_scheduler.LambdaLR,
        'StepLR': optim.lr_scheduler.StepLR,
        'MultiStepLR': optim.lr_scheduler.MultiStepLR,
        'ExponentialLR': optim.lr_scheduler.ExponentialLR,
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
    }
    scheduler_kwargs = {
        'LambdaLR': ['lr_lambda', 'last_epoch'],
        'StepLR': ['step_size', 'gamma', 'last_epoch'],
        'MultiStepLR': ['milestones', 'gamma', 'last_epoch'],
        'ExponentialLR': ['gamma', 'last_epoch'],
        'CosineAnnealingLR': ['T_max', 'eta_min', 'last_epoch'],
        'ReduceLROnPlateau': ['mode', 'factor', 'patience', 'verbose', 'threshold', 'threshold_mode', 'cooldown', 'min_lr', 'eps'],
    }

    def __init__(self, components, loss, optimizer, scheduler=None, update_function=default_training_closure, visdom=True, environment='default', description='', **kwargs):

        Junction.__init__(self, components = components)
        parameters = [x for x in self.model.all_parameters() if x.requires_grad]
        # Initialize engine
        self.optimizer = self.optimizers[optimizer](parameters, **subset_dict(kwargs, self.optimizer_kwargs[optimizer]))
        if scheduler is not None:
            self.optimizer = self.schedulers[scheduler](self.optimizer, **subset_dict(kwargs, self.scheduler_kwargs[scheduler]))
        self.loss = loss
        self.update_function = update_function(self.model, self.optimizer, self.loss)
        self.engine = Engine(self.update_function)

        # Configure metrics and events
        if not visdom:
            environment = None
        self.attach_events(environment=environment, description=description)

    def train(self, dataset = None, max_epochs=10):

        dataset = dataset or self.dataset
        self.engine.run(dataset, max_epochs=max_epochs)

    def run(self, *args, **kwargs):
        """ Alias for train """
        self.train(*args, **kwargs)

    def add_event_handler(self, *args, **kwargs):

        self.engine.add_event_handler(*args, **kwargs)

    def has_event_handler(self, *args, **kwargs):

        return self.engine.has_event_handler(*args, **kwargs)

    def attach_events(self, description, environment=None, save_file = None):

        tim = Timer()
        tim.attach( self.engine,
                    start=Events.STARTED,
                    step=Events.ITERATION_COMPLETED,
        )

        log_interval = 100
        plot_interval = 10

        @self.engine.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration -1)
            if iter % log_interval == 0:
                logging.info("Epoch[{}] Iteration: {} Time: {} Loss: {:.2f}".format(
                    engine.state.epoch, iter, str(datetime.timedelta(seconds=int(tim.value()))), engine.state.output['loss']
                ))
        if environment:

            vis = visdom.Visdom(env=environment)

            def create_plot_window(vis, xlabel, ylabel, title):
                return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

            train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss {0}'.format(description))

            @self.engine.on(Events.ITERATION_COMPLETED)
            def plot_training_loss(engine):
                iter = (engine.state.iteration -1)
                if iter % plot_interval == 0:
                    vis.line(X=np.array([engine.state.iteration]),
                            Y=np.array([engine.state.output['loss']]),
                            update='append',
                            win=train_loss_window)


        # @self.engine.on(Events.ITERATION_COMPLETED)
        # def log_model_state(engine):
        #     iter = (engine.state.iteration-1)
        #     if iter % log_interval == 0:
        #         current_state = Message.from_objects(deepcopy(engine.state.output['state']))
        #         current_state['iteration'] = [iter]
        #         self.model_state = self.model_state.append(current_state)

        # if save_file is not None:
        #     save_interval = 50
        #     handler = ModelCheckpoint('/tmp/models', save_file, save_interval = save_interval, n_saved=5, create_dir=True, require_empty=False)
        #     self.engine.add_event_handler(Events.ITERATION_COMPLETED, handler, {'model': model})
