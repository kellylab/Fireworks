import abc
import Fireworks
from Fireworks.exceptions import EndHyperparameterOptimization
from Fireworks.database import create_table, TableSource
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column
from sqlalchemy_utils import JSONType as JSON

class Factory:
    """
    Base class for parallel hyperparameter optimization in pytorch using queues.
    """
    # NOTE: This is currently not parallelized yet

    def __init__(self, trainer, metrics_dict, generator, eval_dataloader, *args, **kwargs):

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

# Table for storing hyperparameter data in SQLFactory
# columns = [
#     Column('parameters', JSON),
#     Column('metrics', JSON),
# ]
#
# factory_table = create_table('hyperparmeters', columns)

class SQLFactory(Factory):
    """
    Factory that stores parameters in SQLalchemy database.
    """

    def __init__(self,*args, params_table, metrics_tables, engine, **kwargs):
        self.engine = engine
        # self.database = TableSource(factory_table, self.engine, columns=['parameters', 'metrics'])
        self.metrics_tables = metrics_tables
        self.params_table = params_table
        self.params_source = TableSource(self.params_table, self.engine)
        self.metrics_sources = {key: TableSource(value, self.engine) for key, value in self.metrics_tables.items()}

        super().__init__(*args,**kwargs)

    def get_connection(self):

        # TODO: Ensure id consistency accross these tables using foreign key constraints. This should implicitly
        # hold true without such constraints however, because these tables are updated in sync.
        for table in self.metrics_tables.values():
            table.metadata.create_all(self.engine)
        self.params_table.metadata.create_all(self.engine)
        self.id = 0
        # Session = sessionmaker(bind=self.engine)
        # self.session = Session()

    def write(self, params, metrics):

        # self.database.insert(Fireworks.Message({'params':[params], 'metrics_dict': [metrics_dict]}))
        if len(params) != len(metrics):
            raise ValueError("Parameters and Metrics messages must be equal length.")

        for key, metric in metrics.items():
            self.metrics_sources[key].insert(metric)
            self.metrics_sources[key].commit()
        self.params_source.insert(params)
        self.params_source.commit()

    def read(self):

        return self.params_source.query().all(), {key: source.query().all() for key, source in self.metrics_sources.items()}
        # return self.database.query()
