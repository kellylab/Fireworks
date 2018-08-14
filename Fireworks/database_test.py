from sqlalchemy import Table, Column, Integer, String, create_engine
from Fireworks import Message
from Fireworks import database as db
from Fireworks import datasource as ds
import numpy as np
import itertools

class dummy_source(ds.Source):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = 20

    def __getitem__(self, index):

        if type(index) is list:
            index = [i for i in index]
        if type(index) is slice:
            step = index.step or 1
            index = [i for i in range(index.start,index.stop, step)]

        if index == []:
            return None
        elif max(index) < self.length and min(index) >= 0:
            return Message({'name': 'johnny', 'values': np.array(index)})
        else:
            raise IndexError("Out of bounds for dummy source with length {0}.".format(self.length))

    def __len__(self):
        return self.length

def dummy_table(table_name):

    columns = [
        Column('name', String()),
        Column('values', Integer),
        ]
    tab = db.create_table(table_name, columns)

    return tab

def test_TableSource():

    dummy = dummy_source()
    tab = dummy_table('jojo')
    engine = create_engine('sqlite:///:memory:', echo=True)
    tab.metadata.create_all(engine)
    ts = db.TableSource(tab, engine, ['name', 'values'], inputs=dummy)
    batch = ts[2:10]
    ts.insert(batch)
    ts.commit()
    # Check if it worked
    for row, i in zip(ts.query(), itertools.count()):
        assert row.name == 'johnny'
        assert int.from_bytes(row.values, byteorder='little') == i+2 # Have to convert integers back from little endian


def test_make_row():

    tab = dummy_table('bubsy')
    tom = tab(name='ok', values=33)
    assert tom.name == 'ok'
    assert tom.values == 33

def test_create_table():

    tab = dummy_table('munki')
    assert tab.__table__.name == 'munki'
    for colname in ['name', 'values', 'id']:
        assert hasattr(tab, colname)
