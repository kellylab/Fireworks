from Fireworks import factory, Message
from Fireworks.exceptions import EndHyperparameterOptimization
from Fireworks.database import create_table
from sqlalchemy import create_engine, Column, Integer, String

class DummyEvaluator:

    def __init__(self): pass

    def run(self, dataloader): pass

def dummy_trainer():

    def trainer(parameters: dict):
        return DummyEvaluator()

    return trainer

class DummyMetric:

    def __init__(self): pass

    def attach(self, engine, name): pass

    def compute(self):
        return Message({'metric': ['hiiiii']})

def dummy_generator():

    def generator(params: list, metrics: list):
        if len(params) > 10:
            raise EndHyperparameterOptimization
        else:
            i = len(params)
            return Message({'parameters': [i]})

    return generator

def dummy_dataloader():

    return [1,2,3,4]

def test_LocalMemoryFactory():

    trainer = dummy_trainer()
    metrics_dict = {'metric': DummyMetric()}
    generator = dummy_generator()
    dataloader = dummy_dataloader()

    memfactory = factory.LocalMemoryFactory(trainer, metrics_dict, generator, dataloader)
    memfactory.run()

def test_SQLFactory():

    trainer = dummy_trainer()
    metrics_dict = {'metric': DummyMetric()}
    generator = dummy_generator()
    dataloader = dummy_dataloader()
    params_table = create_table('parameters', columns=[Column('parameters', Integer)])
    metrics_table = {'metric': create_table('metrics', columns=[Column('metric', String)])}

    engine = create_engine('sqlite:///:memory:')
    sequel = factory.SQLFactory(trainer, metrics_dict, generator, generator, dataloader, params_table = params_table, metrics_tables = metrics_table, engine=engine)
    sequel.run()
    assert False #TODO: finish writing tests 
