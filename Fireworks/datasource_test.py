import Fireworks
import os
import pandas as pd
from Fireworks import datasource as ds
from Fireworks.message import Message
from Fireworks.utils import index_to_list
import numpy as np
import math
import itertools

test_dir = Fireworks.test_dir

class one_way_dummy(ds.Source):

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

class reset_dummy(ds.Source):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def reset(self):
        self.count = 0

class next_dummy(ds.Source):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def __next__(self):
        self.count += 1
        if self.count < 20:
            return {'count': np.array([self.count])}
        else:
            raise StopIteration # This will trigger StopIteration

class getitem_dummy(ds.Source):

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
            raise IndexError("Out of bounds for dummy source with length {0}.".format(self.length))

    def __len__(self):
        return self.length

class smart_dummy(ds.Source):
    """
    Implements all of the methods in the above dummies
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = 20
        self.count = 0

    def __getitem__(self, index):

        index = index_to_list(index)

        if index == []:
            return None
        elif max(index) < self.length and min(index) >= 0:
            return Message({'values': np.array(index)})
        else:
            raise IndexError("Out of bounds for dummy source with length {0}.".format(self.length))

    def reset(self):
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        if self.count <= 20:
            return Message({'values': np.array([self.count-1])})
        else:
            raise StopIteration# This will trigger StopIteration

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.reset()

def conforms_to_spec(datasource):

    assert hasattr(datasource, '__iter__')
    assert hasattr(datasource, '__next__')

    return True

def test_DataSource(): pass

def test_Title2LabelSource():

    dumbo = one_way_dummy()
    rumbo = one_way_dummy()
    bumbo = reset_dummy()
    gumbo = next_dummy()
    jumbo = next_dummy()

    labeler = ds.Title2LabelSource(inputs = {'yes': dumbo})
    mislabeler = ds.Title2LabelSource(inputs = {'yes': rumbo}, labels_column='barrels')
    donkeykong = ds.Title2LabelSource(inputs = {'yes':gumbo})
    dixiekong = ds.Title2LabelSource(inputs = {'yes': jumbo}, labels_column='bananas')
    labeler.reset()
    assert labeler.count == 0
    mislabeler.reset()
    assert mislabeler.count == 0
    def test_iteration(source, n, label):
        for i in range(n):
            x = source.__next__()
            assert len(x) == 1
            assert (x[label] == ['yes']).all()

    test_iteration(labeler, 10, 'labels')
    test_iteration(mislabeler, 10, 'barrels')
    test_iteration(donkeykong, 12, 'labels')
    test_iteration(dixiekong, 14, 'bananas')
    assert labeler.count == 10
    assert mislabeler.count == 10
    assert donkeykong.count == 12
    assert dixiekong.count == 14

    labeler = ds.Title2LabelSource(inputs = {'yes': bumbo})
    labeler.reset()
    assert labeler.count == 0



def test_BioSeqSource():

    test_file = os.path.join(test_dir, 'sample_genes.fa')
    genes = ds.BioSeqSource(test_file)
    assert conforms_to_spec(genes)
    f = lambda batch: [1 for _ in batch]
    embedding_function = {'sequences': f}

    for gene in genes:
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.columns)
        assert type(gene) is Message
        assert set(gene.tensor_message.keys()) == set()
        assert set(gene.df.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'sequences'])
        assert len(gene) == 1
    # Reset and do it again to confirm we can repeat the source
    genes.reset()
    for gene in genes:
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.columns)
        assert type(gene) is Message
        assert set(gene.tensor_message.keys()) == set()
        assert set(gene.df.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'sequences'])
        assert len(gene) == 1

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
    except IndexError:
        assert True
    else:
        assert False
    assert loopy.length == 20

def test_AggregatorSource():

    dumbo = one_way_dummy()
    bumbo = one_way_dummy()
    gumbo = one_way_dummy()

    angry = ds.RandomAggregatorSource(inputs={'dumbo': dumbo, 'bumbo': bumbo, 'gumbo': gumbo})
    angry.reset()
    assert angry.available_inputs == set(['dumbo', 'bumbo', 'gumbo'])
    numbaz = Message()

    counter = lambda l,i: sum([1 for x in l if x == i]) # Counts how often i appears in l
    while True:
        try:
            numbaz = numbaz.append(angry.__next__())
        except StopIteration:
            break
    assert dumbo.count == 21
    assert bumbo.count == 21
    assert gumbo.count == 21
    assert len(numbaz) == 60
    counts = {i:counter(numbaz['count'],i) for i in range(20)}
    for count in counts.values():
        assert count == 3 # Make sure each element showed up 3 times, corresponding to the 3 inputs

    mangry = ds.ClockworkAggregatorSource(inputs = {'dumbo': dumbo, 'bumbo': bumbo, 'gumbo': gumbo})
    bumbaz = Message()
    for nextone in mangry:
        bumbaz = bumbaz.append(nextone)
    assert dumbo.count == 21
    assert bumbo.count == 21
    assert gumbo.count == 21
    assert len(bumbaz) == 60
    for x,i in zip(bumbaz, itertools.count()):
        assert x['count'][0] == math.floor(i/3)
    counts = {i:counter(bumbaz['count'],i) for i in range(20)}
    for count in counts.values():
        assert count == 3 # Make sure each element showed up 3 times, corresponding to the 3 inputs

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

def test_IndexMapperSource():

    getty = getitem_dummy()
    input_indices = [i for i in range(12)]
    output_indices = [i for i in reversed(range(12))]

    flipped = ds.IndexMapperSource(input_indices, output_indices, inputs={'getty':getty})
    assert len(flipped) == 12
    for i in range(12):
        assert (flipped[i]['values'] == 11-i).all()

def test_PassThroughSource():

    dumbo = smart_dummy()
    pishpish = ds.PassThroughSource(inputs=dumbo)
    assert pishpish.count == 0
    assert Message(pishpish.__next__()) == Message({'values': [0]})
    assert pishpish.count == 1
    pishpish.reset()
    assert pishpish.count == 0
    assert Message(pishpish[12]) == Message({'values': [12]})
    assert Message(pishpish[10:14]) == Message({'values': [10,11,12,13]})
    for i, j in zip(pishpish.reset(), itertools.count()):
        assert Message(i) == Message({'values': [j]})

    assert Message(i) == Message({'values': [19]})

def test_HookedPassThroughSource():

    dumbo = smart_dummy()
    class Hooker(ds.HookedPassThroughSource):

        def _getitem_hook(self, message):

            message['interception'] = ['aho' for _ in range(len(message))]
            message.df = message.df.reindex_axis(sorted(message.df.columns), axis=1)
            return message

        def _next_hook(self, message):

            message['interception'] = ['yaro' for _ in range(len(message))]
            message.df = message.df.reindex_axis(sorted(message.df.columns), axis=1)
            return message

    pishpish = Hooker(inputs=dumbo)
    assert pishpish.count == 0
    assert Message(pishpish.__next__()) == Message({'values': [0], 'interception': ['yaro']})
    assert pishpish.count == 1
    pishpish.reset()
    assert pishpish.count == 0
    assert Message(pishpish[12]) == Message({'values': [12], 'interception': ['aho']})
    assert Message(pishpish[10:14]) == Message({'values': [10,11,12,13], 'interception': ['aho','aho','aho','aho']})
    pishpish.reset()
    for i, j in zip(pishpish, itertools.count()):
        assert i == Message({'values': [j], 'interception': ['yaro']})

    assert i == Message({'values': [19], 'interception': ['yaro']})
