
""" Modify the nonlinear regression examples to read from a database for input data and write to database for factory metrics """

# Read generated data from database

# SQL factory
factory = SQLFactory(components={
    'trainer': get_trainer(trainer),
    'eval_set': test_set,
    'parameterizer': Parameterizer(),
    'metrics': {'accuracy': AccuracyMetric()},
    'engine': engine,
    'params_table': params_table,
    'metrics_tables': metrics_tables,
    })
