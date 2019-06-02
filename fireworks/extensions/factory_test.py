from fireworks import factory, Message
from fireworks.utils.exceptions import EndHyperparameterOptimization
from fireworks.extensions.database import create_table
from sqlalchemy import create_engine, Column, Integer, String
from collections import defaultdict

class DummyEvaluator:

    def __init__(self): pass

    def run(self, dataloader, **kwargs): pass

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

def test_update(): pass

def test_LocalMemoryFactory():

    trainer = dummy_trainer()
    metrics_dict = {'metric': DummyMetric()}
    generator = dummy_generator()
    dataloader = dummy_dataloader()

    memfactory = factory.LocalMemoryFactory(components={"trainer": trainer, "metrics": metrics_dict, 'parameterizer': generator, 'eval_set': dataloader})
    memfactory.run()
    params, metrics = memfactory.read()
    assert type(params) is Message
    assert type(metrics) is defaultdict
    assert set(metrics.keys()) == set(['metric'])
    assert len(params) == 11
    assert len(metrics['metric']) == 11
    assert params[5] == Message({'parameters':[5]})
    assert metrics['metric'][5] == Message({'metric': ['hiiiii']})

def test_SQLFactory():

    trainer = dummy_trainer()
    metrics_dict = {'metric': DummyMetric()}
    generator = dummy_generator()
    dataloader = dummy_dataloader()
    params_table = create_table('parameters', columns=[Column('parameters', Integer)])
    metrics_table = {'metric': create_table('metrics', columns=[Column('metric', String)])}

    engine = create_engine('sqlite:///:memory:')
    sequel = factory.SQLFactory(components={
        'trainer':trainer, 'metrics': metrics_dict, 'parameterizer': generator,
        'eval_set': dataloader, 'params_table': params_table, 'metrics_tables': metrics_table,
        'engine': engine}
        )
    sequel.run()
    params, metrics = sequel.read()

    assert type(params) is Message
    assert type(metrics) is dict
    assert set(metrics.keys()) == set(['metric'])
    assert len(params) == 11
    assert len(metrics['metric']) == 11
    assert params[5] == Message({'id':[6], 'parameters':[5]})
    assert metrics['metric'][5] == Message({'id':[6],'metric': ['hiiiii']})
    for mrow, prow in zip(metrics['metric'], params):
        assert mrow['id'][0] == prow['id'][0]
