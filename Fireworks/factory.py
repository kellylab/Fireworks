import abc
from Fireworks.exceptions import EndHyperparameterOptimization

class Factory:
    """
    Base class for parallel hyperparameter optimization in pytorch using queues.
    """
    def __init__(self, trainer, metrics_dict, generator, eval_dataloader):

        self.trainer = trainer
        self.metrics_dict = metrics_dict
        self.generator = generator
        self.dataloader = eval_dataloader
        self.get_connection()

    @abc.abstractmethod
    def get_connection(self): pass

    def run(self):
        while True:
            past_params, past_metrics = self.read()
            try:
                params = self.generator(past_params, past_metrics)
                evaluator = self.trainer(params)
                # NOTE: This part is pytorch ignite syntax
                for name, metric in self.metrics_dict.items():
                    metric.attach(evaluator, name) # TODO: Make sure this resets the metric
                evaluator.run(self.dataloader)
                computed_metrics = {name: metric.compute() for name, metric in self.metrics_dict.items()}
                self.write(params, computed_metrics)
                evaluator = None
            except EndHyperparameterOptimization: # TODO: Create a specific end of training exception here
                break

    @abc.abstractmethod
    def read(self): pass

    @abc.abstractmethod
    def write(self, params, metrics_dict): pass

class LocalMemoryFactory(Factory):
    """
    Factory that stores parameters in memory.
    """

    def get_connection(self):
        self.params = []
        self.metrics = []

    def read(self):
        return self.params, self.metrics

    def write(self, params, metrics_dict):
        self.params.append(params)
        self.metrics.append(metrics_dict)
