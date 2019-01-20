import Fireworks
import os
import pandas as pd
from Fireworks import pipeline as pl
from Fireworks.message import Message
from Fireworks.utils import index_to_list
from Fireworks.test_utils import *
import numpy as np
import math
import itertools

test_dir = Fireworks.test_dir

def conforms_to_spec(pipe):

    assert hasattr(pipe, '__iter__')
    assert hasattr(pipe, '__next__')

    return True

def test_pipe(): pass

def test_Title2LabelPipe():

    dumbo = one_way_dummy()
    rumbo = one_way_dummy()
    bumbo = reset_dummy()
    gumbo = next_dummy()
    jumbo = next_dummy()

    labeler = pl.Title2LabelPipe( 'yes', dumbo)
    mislabeler = pl.Title2LabelPipe( 'yes', rumbo, labels_column='barrels')
    donkeykong = pl.Title2LabelPipe('yes', gumbo)
    dixiekong = pl.Title2LabelPipe( 'yes', jumbo, labels_column='bananas')
    labeler.reset()
    assert labeler.count == 0
    mislabeler.reset()
    assert mislabeler.count == 0
    def test_iteration(pipe, n, label):
        for i in range(n):
            x = pipe.__next__()
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

    labeler = pl.Title2LabelPipe('yes', bumbo)
    labeler.reset()
    assert labeler.count == 0



def test_BioSeqPipe():

    test_file = os.path.join(test_dir, 'sample_genes.fa')
    genes = pl.BioSeqPipe(test_file)
    assert conforms_to_spec(genes)
    f = lambda batch: [1 for _ in batch]
    embedding_function = {'sequences': f}

    for gene in genes:
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.columns)
        assert type(gene) is Message
        assert set(gene.tensor_message.keys()) == set()
        assert set(gene.df.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'sequences'])
        assert len(gene) == 1
    # Reset and do it again to confirm we can repeat the Pipe
    genes.reset()
    for gene in genes:
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.columns)
        assert type(gene) is Message
        assert set(gene.tensor_message.keys()) == set()
        assert set(gene.df.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'sequences'])
        assert len(gene) == 1

def test_LoopingPipe():

    dumbo = one_way_dummy()
    loopy = pl.LoopingPipe(dumbo)
    loopy[10]
    loopy[5]
    numba1 = len(loopy)
    numba2 = len(loopy) # Call __len__ twice to ensure both compute_length and retrieval of length works
    assert numba1 == 20
    assert numba2 == 20
    x = loopy[0]
    assert x == Message({'count': [0]})
    loopy[10]
    loopy = pl.LoopingPipe(dumbo)
    x = loopy[0]
    assert x == Message({'count': [0]}) # Check that the input Pipes were reset.
    assert loopy.length is None
    try: # Test if length is implicitly calculated whenever input Pipes run out.
        loopy[21]
    except IndexError:
        assert True
    else:
        assert False
    assert loopy.length == 20

def test_RepeaterPipe():

    dumbo = one_way_iter_dummy()
    robert = pl.RepeaterPipe(dumbo)
    numbaz = Message()
    assert len(numbaz) == 0
    for numba in robert:
        numbaz = numbaz.append(numba)
    assert len(numbaz) == robert.repetitions*20

    dumbo = one_way_dummy()
    robert = pl.RepeaterPipe(dumbo)
    numbaz = Message()
    robert.reset()
    i = 0
    assert len(numbaz) == 0
    while True:
        try:
            numbaz = numbaz.append(robert.__next__())
            i+=1
            if i > 1000: # If something goes horribly wrong, cancel test
                assert False
        except StopIteration:
            break
    assert len(numbaz) == robert.repetitions*20

def test_CachingPipe():

    # Test type checking
    dumbo = getitem_dummy()
    cashmoney = pl.CachingPipe(dumbo)

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
    cashmoney = pl.CachingPipe(dumbo)
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

def test_IndexMapperPipe():

    getty = getitem_dummy()
    input_indices = [i for i in range(12)]
    output_indices = [i for i in reversed(range(12))]

    flipped = pl.IndexMapperPipe(input_indices, output_indices,getty)
    assert len(flipped) == 12
    for i in range(12):
        assert (flipped[i]['values'] == 11-i).all()

def test_PassThroughPipe():

    dumbo = smart_dummy()
    pishpish = pl.Pipe(input=dumbo)

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

def test_ShufflingPipe():

    bobby = getitem_dummy()
    shuffla = pl.ShufflerPipe(input=bobby)
    shuffla[2]
    shuffled = False
    for shu, i in zip(shuffla, itertools.count()):
        if shu['values'][0] != i:
            shuffled = True
    assert shuffled

def test_HookedPassThroughPipe():

    dumbo = smart_dummy()
    class Hooker(pl.HookedPassThroughPipe):

        def _getitem_hook(self, message):

            message['interception'] = ['aho' for _ in range(len(message))]
            message.df = message.df.reindex_axis(sorted(message.df.columns), axis=1)
            return message

        def _next_hook(self, message):

            message['interception'] = ['yaro' for _ in range(len(message))]
            message.df = message.df.reindex_axis(sorted(message.df.columns), axis=1)
            return message

    pishpish = Hooker(input=dumbo)
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
