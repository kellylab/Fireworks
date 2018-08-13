from sqlalchemy import Table, Column, Integer, String
from Fireworks import database as db
from Fireworks import datasource as ds

class dummy_source(ds.Source):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = 20

    def __getitem__(self, index):

        index = index_to_list(index)

        if index == []:
            return None
        elif max(index) < self.length and min(index) >= 0:
            return {'name': 'johnny', 'values': np.array(index)}
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

def test_TableSource(): pass

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
