from functools import lru_cache
import math
import numpy as np
from Fireworks.toolbox import pipes as pl
from Fireworks import Model
from collections import defaultdict

def train_test_split(pipe, test=.2):
    """
    Splits input pipe into a training pipe and a test pipe.
    """
    if not hasattr(pipe, '__getitem__'):
        raise ValueError("Input pipe must be indexable via __getitem__")

    l = len(pipe)
    num_test = math.floor(l*test)
    indices = [i for i in range(l)]
    test_indices = sorted(np.random.choice(indices, num_test, replace=False))
    train_indices = [i for i in indices if i not in test_indices]

    test_pipe = pl.IndexMapperPipe(inputs={'data': pipe}, input_indices=range(0,len(test_indices)), output_indices=test_indices)
    train_pipe = pl.IndexMapperPipe(inputs={'data': pipe}, input_indices=range(0,len(train_indices)), output_indices=train_indices)

    return train_pipe, test_pipe

def oversample(): pass

def apply_noise(): pass

class Normalizer(Model):
    """
    Normalizes Data by Mean and Variance. Analogous to sklearn.preprocessing.Normalizer
    """

    required_components = ['mean', 'variance', 'count', 'rolling_sum', 'rolling_squares']

    def __init__(self, *args, **kwargs):

        Model.__init__(self, *args, **kwargs)
        self.freeze(['mean', 'variance', 'count', 'rolling_sum', 'rolling_squares'])

    def init_default_components(self):

        # self.mean = {}
        # self.variance = {}
        # self.count = 0
        # self.rolling_sum = defaultdict(lambda : 0)
        # self.rolling_squares = defaultdict(lambda : 0)
        self.components['mean'] = {}
        self.components['variance'] = {}
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
                batch[key] = (batch[key] - self.mean[key]) / self.variance[key]

        return batch

    def update(self, batch, method=None):
        """
        Updates internal tracking of mean and variance given a batch.
        """
        if method == 'next' or method is None:
            self.count += len(batch)
            for key in batch.keys():
                self.rolling_sum[key] += sum(batch[key])
                self.rolling_squares[key] += sum(batch[key]**2)

    def compile(self):
        """
        Computes mean and variance given internal rolling sum and squares.
        """
        for key in self.rolling_sum:
            self.mean[key] = self.rolling_sum[key] / self.count
            self.variance[key] = (self.rolling_squares[key] - 2*self.rolling_sum[key]*self.mean[key]) / self.count + self.mean[key]**2

    # def fit(self, dataset=None, continuamos=False):
    #
    #     if dataset is None:
    #         dataset = self.input
    #
    #     if not continuamos:
    #         self.reset()
    #
    #     for batch in dataset:

    def reset(self):

        self.count = 0
        self.rolling_sum = defaultdict(lambda : 0)
        self.rolling_squares = defaultdict(lambda : 0)

        try:
            self.recursive_call('reset')()
        except:
            pass
