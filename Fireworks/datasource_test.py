import Fireworks
import os
import pandas as pd
from Fireworks import datasource as ds
from Fireworks.message import Message
from Fireworks.utils import index_to_list
import numpy as np

test_dir = Fireworks.test_dir

class one_way_dummy(ds.DataSource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def __next__(self):
        self.count += 1
        if self.count <= 20:
            return {'count': np.array([self.count-1])}
        else:
            raise StopIteration# This will trigger StopIteration

    def reset(self):
        self.count = 0

class reset_dummy(ds.DataSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def reset(self):
        self.count = 0

class next_dummy(ds.DataSource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def __next__(self):
        self.count += 1
        if self.count < 20:
            return {'count': np.array([self.count])}
        else:
            raise StopIteration # This will trigger StopIteration

class getitem_dummy(ds.DataSource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = 20

    def __getitem__(self, index):

        index = index_to_list(index)
        if index == []:
            return None
        elif max(index) < self.length and min(index) >= 0:
            return {'values': np.array(index)}
        else:
            raise ValueError("Out of bounds for dummy source with length {0}.".format(self.length))

    def __len__(self):
        return self.length

def conforms_to_spec(datasource):

    assert hasattr(datasource, '__iter__')
    assert hasattr(datasource, 'to_tensor')
    assert hasattr(datasource, '__next__')

    return True

def test_DataSource(): pass

def test_BioSeqSource():

    test_file = os.path.join(test_dir, 'sample_genes.fa')
    genes = ds.BioSeqSource(test_file)
    assert conforms_to_spec(genes)
    f = lambda batch: [1 for _ in batch]
    embedding_function = {'sequences': f}

    for gene in genes:
        assert type(gene) is pd.DataFrame
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.keys())
        message = genes.to_tensor(gene, embedding_function)
        assert type(message) is Message
        assert set(message.tensor_message.keys()) == set(['sequences'])
        assert set(message.df.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'rawsequences'])
        assert len(message) == 1
    # Reset and do it again to confirm we can repeat the source
    genes.reset()
    for gene in genes:
        assert type(gene) is pd.DataFrame
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.keys())
        message = genes.to_tensor(gene, embedding_function)
        assert type(message) is Message
        assert set(message.tensor_message.keys()) == set(['sequences'])
        assert set(message.df.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'rawsequences'])
        assert len(message) == 1

def test_LoopingSource():

    # Test type checking
    try:
        dumbo = reset_dummy()
        poopy = ds.LoopingSource(inputs = {'mumbo': dumbo})
    except TypeError:
        assert True
    else:
        assert False
    try:
        dumbo = next_dummy()
        scooby = ds.LoopingSource(inputs = {'jumbo': dumbo})
    except TypeError:
        assert True
    else:
        assert False
    dumbo = one_way_dummy()
    loopy = ds.LoopingSource(inputs = {'dumbo': dumbo})
    loopy[10]
    loopy[5]
    numba1 = len(loopy)
    numba2 = len(loopy) # Call __len__ twice to ensure both compute_length and retrieval of length works
    assert numba1 == 20
    assert numba2 == 20
    x = loopy[0]
    assert x == Message({'count': [0]})
    loopy[10]
    loopy = ds.LoopingSource(inputs = {'dumbo': dumbo})
    x = loopy[0]
    assert x == Message({'count': [0]}) # Check that the input sources were reset.
    assert loopy.length is None
    try: # Test if length is implicitly calculated whenever input sources run out.
        loopy[21]
    except StopIteration:
        assert True
    else:
        assert False
    assert loopy.length == 20

def test_CachingSource():

    # Test type checking
    dumbo = next_dummy()
    try:
        cashmoney = ds.CachingSource(inputs={'dumbo': dumbo})
    except TypeError:
        assert True
    else:
        assert False
    dumbo = getitem_dummy()
    cashmoney = ds.CachingSource(inputs={'dumbo': dumbo})

    # Test __getitem__
    assert cashmoney.cache.cache == Message()
    x = cashmoney[2]
    assert x == Message({'values':[2]})
    assert cashmoney.cache.cache == Message({'values':[2]})
    x = cashmoney[2,3,4]
    x = cashmoney[2,3,4]
    assert x == Message({'values':[2,3,4]})
    assert cashmoney.cache.cache == Message({'values':[2,3,4]})
    x = cashmoney[5,2,3]
    assert x == Message({'values':[5,2,3]})
    assert set(cashmoney.cache.pointers.keys()) == set([2,3,4,5])
    x = cashmoney[8,5,1,4,0,9,2]
    assert x == Message({'values':[8,5,1,4,0,9,2]})
    assert set(cashmoney.cache.pointers.keys()) == set([0,1,2,3,4,5,8,9])
    # Test explicit length calculation
    assert cashmoney.length is None
    length = len(cashmoney)
    assert length == 20
    # Test implicit length calculation
    cashmoney = ds.CachingSource(inputs={'dumbo': dumbo})
    assert cashmoney.length is None
    assert cashmoney.lower_bound == 0
    cashmoney[10]
    assert cashmoney.lower_bound == 10
    cashmoney[8]
    assert cashmoney.lower_bound == 10
    cashmoney[12,15]
    assert cashmoney.lower_bound == 15
    length = len(cashmoney)
    assert length == 20
