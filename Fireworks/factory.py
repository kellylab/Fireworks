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
                # Generate new set of parameters
                params = self.generator(past_params, past_metrics)
                # Generate an evaluator from the params
                evaluator = self.trainer(params)
                # NOTE: This part is pytorch ignite syntax
                for name, metric in self.metrics_dict.items():
                    metric.attach(evaluator, name) # TODO: Make sure this resets the metric
                # Running the evaluator should perform training on the dataset followed by evlaution and return evaluation metrics
                evaluator.run(self.dataloader)
                # Evaluate the metrics that were attached to the evaluator
                computed_metrics = {name: metric.compute() for name, metric in self.metrics_dict.items()}
                self.write(params, computed_metrics)
                evaluator = None
            except EndHyperparameterOptimization:
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
