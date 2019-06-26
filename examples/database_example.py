from sqlalchemy import Table, Column, Float, String, create_engine
from fireworks.extensions.database import create_table, TablePipe, DBPipe
from fireworks.toolbox.preprocessing import train_test_split
from fireworks.toolbox.pipes import LoopingPipe, CachingPipe, ShufflerPipe, BatchingPipe, TensorPipe
import os
import json
""" Modify the nonlinear regression examples to read from a database for input data and write to database for factory metrics """

# Read generated data from database
from nonlinear_regression_utils import generate_data

columns = [
    Column('x', Float),
    Column('y', Float),
    Column('errors', Float),
]

table = create_table("nonlinear_regression", columns)

def write_data(filename='example.sqlite', n=1000):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    engine = create_engine('sqlite:///{0}'.format(filename))
    db = TablePipe(table, engine)

    data, params = generate_data(n)
    db.insert(data)

    with open(filename+"_params", 'w') as f:
        f.write(json.dumps(params))

    db.commit()

def load_data(filename='example.sqlite'):
    if not os.path.exists(filename):
        raise FileNotFoundError("File {0} does not exist.".format(filename))
    with open(filename+'_params') as f:
        params = json.load(f)

    engine = create_engine('sqlite:///{0}'.format(filename))
    db = DBPipe(table, engine) # Default query is SELECT * FROM table

    return db, params

def get_data(filename='example.sqlite', n=1000):
    if not os.path.exists(filename) and os.path.exists(filename+'_params'):
        write_data(filename, n)

    data, params = load_data(filename)
    looper = LoopingPipe(data)
    cache = CachingPipe(looper, cache_size=1000)
    train, test = train_test_split(cache, test=.25)

    shuffler = ShufflerPipe(train)
    minibatcher = BatchingPipe(shuffler, batch_size=25)
    train_set = TensorPipe(minibatcher, columns=['x','y'])

    test_set = TensorPipe(test, columns=['x','y'])

    return train_set, test_set, params
