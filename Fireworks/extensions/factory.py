import abc
from Fireworks import Message, Junction
from .training import IgniteJunction
from Fireworks.utils.exceptions import EndHyperparameterOptimization
from .database import create_table, TablePipe
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column
from sqlalchemy_utils import JSONType as JSON
from collections import defaultdict
import types

def update(bundle: dict, parameters: dict):
    """
    Args:
        bundle - A dictionary of key: (obj, atr). Obj is the object referred to, and attr is a string with the name of the attribute to be assigned.
        parameters - A dictionary of key: value. Wherever keys match, obj.attr will be set to value.
    """
    for key, param in parameters.items():
        if key in bundle:
            obj, atr = bundle[key]
            setattr(obj, attr, param)

class Factory(Junction):
    """
    Base class for parallel hyperparameter optimization in pytorch using queues.
    """
    # NOTE: This is currently not parallelized yet

    required_components = {'trainer': types.FunctionType, 'eval_set': object, 'parameterizer': types.FunctionType, 'metrics': dict}

    def __init__(self, *args, **kwargs):

        Junction.__init__(self, *args, **kwargs)
        # self.trainer = trainer
        # self.metrics_dict = metrics_dict
        # self.parameterizer = parameterizer
        # self.eval_set = eval_eval_set
        self.get_connection()

    @abc.abstractmethod
    def get_connection(self): pass

    def run(self):
        while True:
            past_params, past_metrics = self.read()
            try:
                # Generate new set of parameters
                params = self.parameterizer(past_params, past_metrics)
                # Generate an evaluator
                evaluator = self.trainer(params)
                # NOTE: This part is pytorch ignite syntax
                for name, metric in self.metrics.items():
                    metric.attach(evaluator, name) # TODO: Make sure this resets the metric
                # Running the evaluator should perform training on the dataset followed by evlaution and return evaluation metrics
                evaluator.run(self.eval_set)
                # Evaluate the metrics that were attached to the evaluator
                computed_metrics = {name: metric.compute() for name, metric in self.metrics.items()}
                self.write(params, computed_metrics)
                evaluator = None
            except EndHyperparameterOptimization:
                self.after()
                break

    @abc.abstractmethod
    def read(self): pass

    @abc.abstractmethod
    def write(self, params, metrics_dict): pass

    def train(self, params):

        self.trainer.model.update_components(params)

    def after(self, *args, **kwargs): pass

class LocalMemoryFactory(Factory):
    """
    Factory that stores parameters in memory.
    """

    def get_connection(self):
        self.params = Message()
        self.computed_metrics = defaultdict(Message)

    def read(self):
        return self.params, self.computed_metrics

    def write(self, params, metrics_dict):
        self.params = self.params.append(params)
        for key in metrics_dict:
            self.computed_metrics[key] = self.computed_metrics[key].append(metrics_dict[key])

# Table for storing hyperparameter data in SQLFactory
# columns = [
#     Column('parameters', JSON),
#     Column('metrics', JSON),
# ]
#
# factory_table = create_table('hyperparmeters', columns)

class SQLFactory(Factory):
    """
    Factory that stores parameters in SQLalchemy database while caching them locally.
    """

    def __init__(self,*args, params_table, metrics_tables, engine, **kwargs):
        self.engine = engine
        # self.database = TablePipe(factory_table, self.engine, columns=['parameters', 'metrics'])
        self.metrics_tables = metrics_tables
        self.params_table = params_table
        self.params_pipe = TablePipe(self.params_table, self.engine)
        self.metrics_pipes = {key: TablePipe(value, self.engine) for key, value in self.metrics_tables.items()}
        self.computed_metrics = defaultdict(Message)

        super().__init__(*args,**kwargs)

    def get_connection(self):

        # TODO: Ensure id consistency accross these tables using foreign key constraints. This should implicitly
        # hold true without such constraints however, because these tables are updated in sync.
        for table in self.metrics_tables.values():
            table.metadata.create_all(self.engine)
        self.params_table.metadata.create_all(self.engine)
        self.id = 0
        self.sync()
        # Session = sessionmaker(bind=self.engine)
        # self.session = Session()

    def write(self, params, metrics):

        # self.database.insert(Fireworks.Message({'params':[params], 'metrics_dict': [metrics_dict]}))
        if len(params) != len(metrics):
            raise ValueError("Parameters and Metrics messages must be equal length.")

        for key, metric in metrics.items():
            self.computed_metrics[key] = self.computed_metrics[key].append(metric)
            self.metrics_pipes[key].insert(metric)
            self.metrics_pipes[key].commit()
        self.params = self.params.append(params)
        self.params_pipe.insert(params)
        self.params_pipe.commit()

    def read(self):

        return self.params, self.computed_metrics

    def read_db(self):

        return self.params_pipe.query().all(), {key: pipe.query().all() for key, pipe in self.metrics_pipes.items()}

    def sync(self):
        """ Syncs local copy of metrics and params with db. """
        self.params, self.computed_metrics = self.read_db()

    def after(self):
        self.sync()
