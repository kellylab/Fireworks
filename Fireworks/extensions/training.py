from Fireworks import Junction, Model
from Fireworks.utils import subset_dict
import torch
from torch import optim
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss
import visdom
import datetime
import numpy as np
import types

def default_training_closure(model, optimizer, loss_fn):

    def update_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(batch)
        optimizer.step()

        return {'loss': loss, 'optimizer': optimizer.state_dict(), 'output': output}

class IgniteJunction(Junction):

    required_components = {'model': Model, 'dataset': object}
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

    def __init__(self, components, loss, optimizer, scheduler=None, update_function=default_training_closure, **kwargs):

        Junction.__init__(self, components)

        # Initialize engine
        self.optimizer = optimizers[optimizer](self.model.all_parameters(), **subset_dict(kwargs, optimzer_kwargs[optimizer]))
        if scheduler is not None:
            self.optimizer = schedulers[scheduler](self.optimizer, **subset_dict(kwargs, scheduler_kwargs[scheduler]))
        self.loss = loss_fn
        self.update_function = update_function(self.model, self.optimizer, self.loss)
        self.engine = Engine(self.update_function)

        # Configure metrics and events
        self.attach_events()

    def train(self, dataset = None):

        dataset = dataset or self.dataset
        self.engine.run(dataset)

    def attach_events(self):

        tim = Timer()
        tim.attach( trainer,
                    start=Events.STARTED,
                    step=Events.ITERATION_COMPLETED,
        )

        vis = visdom.Visdom(env=environment) # TODO: Specify which workspace

        def create_plot_window(vis, xlabel, ylabel, title):
            return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

        train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss {0}'.format(description))
        log_interval = 100

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def plot_training_loss(engine):
            iter = (engine.state.iteration -1)
            if iter % log_interval == 0:
                print("Epoch[{}] Iteration: {} Time: {} Loss: {:.2f}".format(
                    engine.state.epoch, iter, str(datetime.timedelta(seconds=int(tim.value()))), engine.state.output['loss']
                ))
            vis.line(X=np.array([engine.state.iteration]),
                     Y=np.array([engine.state.output['loss']]),
                     update='append',
                     win=train_loss_window)

        if save_file is not None:
            save_interval = 50
            handler = ModelCheckpoint('/tmp/models', save_file, save_interval = save_interval, n_saved=5, create_dir=True, require_empty=False)
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED, handler, {'model': model})
