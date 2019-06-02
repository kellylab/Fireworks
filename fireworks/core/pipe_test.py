from .pipe import Pipe, HookedPassThroughPipe
from .message import Message
from fireworks.utils import index_to_list
from fireworks.utils.test_helpers import *
import numpy as np
import itertools

test_dir = fireworks.test_dir

# def conforms_to_spec(pipe):
#
#     assert hasattr(pipe, '__iter__')
#     assert hasattr(pipe, '__next__')
#
#     return True
#
# def test_pipe(): pass

def test_PassThroughPipe():

    dumbo = smart_dummy()
    pishpish = Pipe(input=dumbo)

    assert pishpish.count == 0
    assert Message(pishpish.__next__()) == Message({'values': [0]})
    assert pishpish.count == 1
    pishpish.reset()
    assert pishpish.count == 0
    assert Message(pishpish[12]) == Message({'values': [12]})
    assert Message(pishpish[10:14]) == Message({'values': [10,11,12,13]})
    for i, j in zip(pishpish, itertools.count()):
        assert Message(i) == Message({'values': [j]})

    assert Message(i) == Message({'values': [19]})

    # Test private attributes
    dumbo._hii = 2
    dumbo.bye = 3
    assert hasattr(dumbo, '_hii')
    assert hasattr(dumbo, 'bye')
    assert not hasattr(pishpish, '_hii')
    assert hasattr(pishpish, 'bye')

def test_HookedPassThroughPipe():

    dumbo = smart_dummy()
    class Hooker(HookedPassThroughPipe):

        def _getitem_hook(self, message):

            message['interception'] = ['aho' for _ in range(len(message))]
            message.df = message.df.reindex(sorted(message.df.columns), axis=1)
            return message

        def _next_hook(self, message):

            message['interception'] = ['yaro' for _ in range(len(message))]
            message.df = message.df.reindex(sorted(message.df.columns), axis=1)
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

def test_recursive_decorator():

    alto = recursion_dummy()
    balto = recursion_dummy(alto)
    calto = recursion_dummy(balto)

    assert alto.height == 0 and alto.depth == 0
    assert balto.height == 0  and balto.depth == 0
    assert calto.height == 0  and calto.depth == 0

    calto.jump()
    assert calto.height == 1
    assert balto.height == 1
    assert alto.height == 1

    balto.jump()
    assert calto.height == 1
    assert balto.height == 2
    assert alto.height == 2

    calto.somersault(1)
    assert alto.depth == 1
    assert balto.depth == 2
    assert calto.depth == 3

    balto.somersault(1)
    assert alto.depth == 2
    assert balto.depth == 4

def test_setstate_getstate():

    dumbo = smart_dummy()
    dumbo.stateful_attributes = ['length', 'count']
    for i in range(10):
        next(dumbo)
    assert dumbo.count == 10
    assert dumbo.length == 20
    state = dumbo.get_state()
    assert state['internal']['count'] == 10
    assert state['internal']['length'] == 20
    new_state = {'internal': {'count': 15, 'length': 30}}
    dumbo.set_state(new_state)
    assert dumbo.count == 15
    assert dumbo.length == 30
    i = 15
    while True:
        try:
            dumbo.__next__()
            i += 1
        except StopIteration:
            break
    assert dumbo.count == 31
    assert i == 30
