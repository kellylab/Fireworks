from sqlalchemy import Table, Column, Integer, String, create_engine
from fireworks import Message
from fireworks.extensions import database as db
from fireworks import pipe as pl
import os
import numpy as np
import itertools
import copy

class dummy_pipe(pl.Pipe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = 20

    def __getitem__(self, index):

        if type(index) is list:
            index = [i for i in index]
        if type(index) is slice:
            step = index.step or 1
            index = [i for i in range(index.start, index.stop, step)]

        if index == []:
            return None
        elif max(index) < self.length and min(index) >= 0:
            return Message({'name': 'johnny', 'values': np.array(index)})
        else:
            raise IndexError("Out of bounds for dummy pipe with length {0}.".format(self.length))

    def __len__(self):
        return self.length

def dummy_table(table_name):

    columns = [
        Column('name', String),
        Column('values', Integer),
        ]
    tab = db.create_table(table_name, columns)

    return tab

def test_make_row():

    tab = dummy_table('bubsy')
    tom = tab(name='ok', values=33)
    assert tom.name == 'ok'
    assert tom.values == 33

    engine = create_engine('sqlite:///:memory:', echo=True)
    pipe = db.TablePipe(tab, engine)

    message = Message({'name': ['a','b'], 'values': [1,2]})
    row = pipe.make_row(message[0])
    assert row.name == 'a'
    assert row.values == 1

def test_create_table():

    tab = dummy_table('munki')
    assert tab.__table__.name == 'munki'
    for colname in ['name', 'values', 'id']:
        assert hasattr(tab, colname)

def test_parse_columns():

    tab = dummy_table('joseph')
    colnames = db.parse_columns(tab)
    for c in colnames:
        assert type(c) is str
    assert set(['values', 'name']) == set(colnames)
    colnames = db.parse_columns(tab, ignore_id=False)
    for c in colnames:
        assert type(c) is str

    assert set(['values', 'id', 'name']) == set(colnames)

def test_TablePipe_explicit():
    """ Colnames are explicitly labeled here. """

    dummy = dummy_pipe()
    tab = dummy_table('jojo')
    engine = create_engine('sqlite:///:memory:', echo=True)

    tab.metadata.create_all(engine)
    ts = db.TablePipe(tab, engine, ['values', 'name'], input=dummy)
    batch = ts[2:10]
    ts.insert(batch)
    ts.commit()
    # Check if it worked
    for row, i in zip(ts.query(), itertools.count()):
        assert type(row) is Message
        assert row['name'][0] == 'johnny'
        # assert int.from_bytes(row.values, byteorder='little') == i+2 # Have to convert integers back from little endian
        assert row['values'][0] == i+2
    assert i > 1

def test_TablePipe_implicit():
    """ Colnames are implicitly identified here. """

    dummy = dummy_pipe()
    tab = dummy_table('josuke')
    engine = create_engine('sqlite:///:memory:', echo=True)

    ts = db.TablePipe(tab, engine, input=dummy)
    batch = ts[2:10]
    ts.insert(batch)
    ts.commit()
    # Check if it worked
    for row, i in zip(ts.query(), itertools.count()):
        assert type(row) is Message
        assert row['name'][0] == 'johnny'
        # assert int.from_bytes(row.values, byteorder='little') == i+2 # Have to convert integers back from little endian
        assert row['values'][0] == i+2

    # Test deletes
    cop = ts.query().filter('values','between',4,8).all()
    assert len(cop) == 5
    assert (ts.query().all()['values'] == [2,3,4,5,6,7,8,9]).all()
    ts.delete('values', batch['values'][1:4])
    ts.commit()
    assert (ts.query().all()['values'] == [2,6,7,8,9]).all()
    assert 'id' in ts.query().all()

    # Test updates
    batch = ts.query().all()
    # assert False
    new_batch = copy.deepcopy(batch)
    new_batch['values'] = [10,11,12,13,14]
    ts.update('id', new_batch)
    ts.commit()
    newer_batch = ts.query().all()
    assert newer_batch == new_batch
    assert newer_batch != batch

def test_DBPipe():
    dummy = dummy_pipe()
    tab = dummy_table('jotaro')
    engine = create_engine('sqlite:///jotaro.sqlite', echo=True)
    if os.path.exists('jotaro.sqlite'):
        os.remove('jotaro.sqlite')
    tab.metadata.create_all(engine)
    ts = db.TablePipe(tab, engine, ['name', 'values'], input=dummy)
    batch = ts[2:10]
    ts.insert(batch)
    ts.commit()
    deedee = db.DBPipe(tab, engine)
    for row, i in zip(deedee, itertools.count()):
        assert row == Message({'id':[i+1],'name': ['johnny'], 'values':[i+2]})
    deedee.reset_session()
    for row, i in zip(deedee, itertools.count()):
        assert row == Message({'id':[i+1],'name': ['johnny'], 'values':[i+2]})

    # Test using reflections
    ts = db.DBPipe('jotaro', engine)
    for row, i in zip(deedee, itertools.count()):
        assert row == Message({'id':[i+1],'name': ['johnny'], 'values':[i+2]})
    deedee.reset_session()
    for row, i in zip(deedee, itertools.count()):
        assert row == Message({'id':[i+1],'name': ['johnny'], 'values':[i+2]})

    # Reset the session and try again.
    deedee.reset_session()
    for row, i in zip(deedee, itertools.count()):
        assert row == Message({'id':[i+1],'name': ['johnny'], 'values':[i+2]})
    deedee.reset_session()
    for row, i in zip(deedee, itertools.count()):
        assert row == Message({'id':[i+1],'name': ['johnny'], 'values':[i+2]})

    # Test using reflections
    ts = db.DBPipe('jotaro', engine)
    for row, i in zip(deedee, itertools.count()):
        assert row == Message({'id':[i+1],'name': ['johnny'], 'values':[i+2]})
    deedee.reset_session()
    for row, i in zip(deedee, itertools.count()):
        assert row == Message({'id':[i+1],'name': ['johnny'], 'values':[i+2]})

def test_DBPipe_query():

    dummy = dummy_pipe()
    tab = dummy_table('giorno')
    engine = create_engine('sqlite:///:memory:', echo=True)
    tab.metadata.create_all(engine)
    ts = db.TablePipe(tab, engine, ['name', 'values'], input=dummy)
    batch = ts[0:20]
    ts.insert(batch)
    ts.commit()
    result = ts.query().all()
    result = ts.query('values').all()
    assert set(result.columns) == set(['values'])
    result = ts.query('values', 'name').all()
    assert set(result.columns) == set(['name', 'values'])
    result = ts.query('values').filter('values', 'between', 1,5).all()
    assert (result['values'] == [1,2,3,4,5]).all()
    assert set(result.columns) == set(['values'])
    result = ts.query('values', 'name').filter('values', 'between', 1,5).all()
    assert (result['values'] == [1,2,3,4,5]).all()
    assert set(result.columns) == set(['name', 'values'])

    # Test if resetting the session after making a query affects anything.
    query = ts.query('values')
    query.reset_session()
    result = query.all()
    assert set(result.columns) == set(['values'])
    query = ts.query('values', 'name')
    query.reset_session()
    result = query.all()
    assert set(result.columns) == set(['name', 'values'])
    query = ts.query('values').filter('values', 'between', 1,5)
    query.reset_session()
    result = query.all()
    assert (result['values'] == [1,2,3,4,5]).all()
    assert set(result.columns) == set(['values'])
    query = ts.query('values', 'name')
    query.reset_session()
    result = query.filter('values', 'between', 1,5).all()
    assert (result['values'] == [1,2,3,4,5]).all()
    assert set(result.columns) == set(['name', 'values'])

def test_reflect_table():

    tab = dummy_table('jolyne')
    engine = create_engine('sqlite:///:jolyne.sqlite', echo=True)
    tab.metadata.create_all(engine)
    jolyne = db.reflect_table('jolyne', engine)
    assert type(jolyne) is Table
    assert jolyne.columns.keys() == tab.__table__.columns.keys()
