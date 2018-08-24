from Fireworks import Message
from Fireworks import experiment as exp
from Fireworks import database as db
from sqlalchemy import Column, Integer, String
from shutil import rmtree
import datetime
import pandas as pd
import os

columns = [
    Column('superpower', String),
    Column('name', String),
    Column('age', Integer),
]
table = db.create_table('superpowers', columns=columns)

def dummy_table(engine):

    return db.TableSource(table, engine, columns=['superpower', 'name', 'age'])

def test_Experiment():

    # Test directory creation
    dirs = os.listdir()
    all_xmen = [x for x in dirs if x.startswith('xmen')]
    for man in all_xmen:
        rmtree(man)
    dirs = os.listdir()
    all_xmen = [x for x in dirs if x.startswith('xmen')]
    assert len(all_xmen) == 0
    xmen = exp.Experiment('xmen', os.getcwd(), description='ok')
    xmen2 = exp.Experiment('xmen', os.getcwd(), description='yahoo')
    dirs = os.listdir()
    all_xmen = [x for x in dirs if x.startswith('xmen')]
    assert set(all_xmen) == set(['xmen_0', 'xmen_1'])

    # Test metadata table
    query = xmen.metadata.query()
    i = 0
    for row in query:
        i += 1
    assert i == 1
    assert row['name'][0] == 'xmen'
    assert row['iteration'][0] == 0
    assert row['description'][0] == 'ok'
    assert type(row['timestamp'][0]) is pd.Timestamp

    # Test engine/session creation
    assert xmen.engines == {}
    nginx = xmen.create_engine('wolverine')
    assert 'wolverine' in xmen.engines
    try:
        apache = xmen.create_engine('wolverine')
        assert False
    except ValueError:
        assert True # It should throw an error if you try to have two engines of the same name
    wolverine = xmen.get_session('wolverine')
    xavier = xmen.get_session('xavier')
    assert 'xavier' in xmen.engines

    # Test that the engine works
    saver = dummy_table(nginx)
    saver.insert(Message({
        'superpower': ['flying', 'walking', 'eating'],
        'name': ['flyman', 'walkergirl', 'bumbo'],
        'age': [2,3,4],
        }))
    saver.commit()
    rows = saver.query()

    for row in rows:
        assert type(row) is Message
    assert row == Message({'superpower': ['eating'], 'name':['bumbo'], 'age':[4], 'id':[3]})

    # Test file opening and saving
    path = xmen.open('ironman', string_only=True)
    assert path == os.path.join(xmen.save_path, 'ironman')
    with xmen.open('ironman', 'w') as ironman:
        ironman.write("Hey guys it's me tony the tiger.")
    # try:
    #     path = xmen.open('ironman', string_only=True)
    #     assert False
    # except IOError: # It should detect that this file already exists
    #     assert True

    with xmen.open('ironman') as ironman:
        assert ironman.read() == "Hey guys it's me tony the tiger."

    # Clean up
    all_xmen = [x for x in dirs if x.startswith('xmen')]
    for man in all_xmen:
        rmtree(man)

def test_load_experiment():
    pass 
