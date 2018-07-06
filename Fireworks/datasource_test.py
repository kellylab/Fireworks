import Fireworks
import os
import pandas as pd
from Fireworks import datasource as ds
from Fireworks.message import Message
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

    # Test __getitem__

    # Test explicit length calculation

    # Test implicit length calculation
