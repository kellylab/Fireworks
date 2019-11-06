from functools import lru_cache
import math
import numpy as np
from fireworks.toolbox import pipes as pl
from fireworks.core import Model, PyTorch_Model
from collections import defaultdict
import torch 
import random

"""
This file contains models that can perform common preprocessing tasks, such as batch normalization.
"""

def train_test_split(pipe, test=.2):
    """
    Splits input pipe into a training pipe and a test pipe. The indices representing the input pipe are shuffled, and assigned to the training
    and test sets randomly based on the proportions specified.

    Args:
        - pipe: A pipe which represents the data to be split up.
        - test: The proportion of the set that should be returns as test set. This should be between 0 and 1.

    Returns:
        - train_pipe: A pipe that represents the training data. You can call __getitem__, __next__, etc. on this pipe and it will transparently
                      provide elements from the shuffled training set.
        - test_pipe: Analogous to the train_pipe, this represents the test data, which is shuffled and disjoint from the training data.
    """
    if not hasattr(pipe, '__getitem__'):
        raise ValueError("Input pipe must be indexable via __getitem__")

    l = len(pipe)
    num_test = math.floor(l*test)
    indices = random.sample(range(l), l)    
    train_indices = indices[0:l-num_test]
    test_indices = indices[l-num_test:]

    test_pipe = pl.IndexMapperPipe(input=pipe, input_indices=range(0,len(test_indices)), output_indices=test_indices)
    train_pipe = pl.IndexMapperPipe(input=pipe, input_indices=range(0,len(train_indices)), output_indices=train_indices)

    return train_pipe, test_pipe

def oversample(): pass

def apply_noise(): pass

#IDEA: Instead of implementing this, what if we couple a LoopingSource + CachingSource to a SKLearn estimator?
# Doing so would create a path to seamless insertion of SKlearn modules into pipelines.

class Normalizer(PyTorch_Model):
    """
    Normalizes Data by Mean and Variance. Analogous to sklearn.preprocessing.Normalizer
    This Model uses a one-pass method to estimate the sample variance which is not guaranteed to be numerically stable.

    The functionality is implemented using hooks. Every time data is accessed from upstream pipes, this Model updates its estimate of the
    population mean and variance using the update() method. If self._inference_enabled is set to True, then the data will also be normalized
    based on those estimates. Means and variances are calculated on a per-column basis. You can also disable/enable the updating of these
    estimate by calling self.enable_updates / self.disable_updates.
    """

    required_components = ['mean', 'variance', 'count', 'rolling_sum', 'rolling_squares']

    def __init__(self, *args, **kwargs):

        PyTorch_Model.__init__(self, *args, **kwargs)
        self.freeze(['mean', 'variance', 'count', 'rolling_sum', 'rolling_squares'])

    def init_default_components(self):

        self.components['mean'] = defaultdict(lambda : [0])
        self.components['variance'] = defaultdict(lambda : 1)
        self.components['count'] = [0.]
        self.components['rolling_sum'] = defaultdict(lambda : 0.)
        self.components['rolling_squares'] = defaultdict(lambda : 0.)

    def forward(self, batch):
        """
        Uses computed means and variances in order to transform the given batch.
        """

        keys = self.mean.keys()
        for key in keys:
            if key in batch:
                if self.variance[key].device.type == 'cuda':
                    batch[key] = (batch[key] - self.mean[key])/torch.sqrt(self.variance[key])
                else:
                    batch[key] = (batch[key] - self.mean[key])/np.sqrt(self.variance[key])

        return batch

    def update(self, batch, method=None):
        """
        Updates internal tracking of mean and variance given a batch.
        """
        if method == 'next' or method is None:
            self.count += len(batch)
            for key in batch.keys(): #WARNING: This is numerically unstable
                self.rolling_sum[key] += sum(batch[key])
                self.rolling_squares[key] += sum((batch[key]-self.mean[key][0])**2)
                # self.rolling_squares[key] += np.var(batch[key])
                self.compile()

    def compile(self):
        """
        Computes mean and variance given internal rolling sum and squares.
        """
        for key in self.rolling_sum:
            self.mean[key] = self.rolling_sum[key] / self.count
            self.variance[key] = (self.rolling_squares[key] - (self.rolling_sum[key])**2/self.count) / self.count + self.mean[key]**2
            self.variance[key] = self.rolling_squares[key] / self.count

    def get_state(self):
        """
        Since defaultdicts are not picklable, we implement custom serialization logic for a normalizer.
        """
        state = super().get_state()
        for key in ['mean', 'variance', 'rolling_sum', 'rolling_squares']:
            state['internal'][key] = dict(state['internal'][key])
        
        return state

    def set_state(self, state, reset=False):
        """
        Since defaultdicts are not picklable, we implement custom serialization logic for a normalizer.
        """
        super().set_state(state, reset=reset)
        self.init_default_components()
        for key in ['mean', 'variance', 'rolling_sum', 'rolling_squares']:
            for k,v in state['internal'][key].items():
                self.components[key][k] = v