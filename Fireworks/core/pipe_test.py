import os
from .pipe import Pipe, HookedPassThroughPipe
from .message import Message
from Fireworks.utils import index_to_list
from Fireworks.utils.test_helpers import *
import numpy as np
import itertools

test_dir = Fireworks.test_dir

def conforms_to_spec(pipe):

    assert hasattr(pipe, '__iter__')
    assert hasattr(pipe, '__next__')

    return True

def test_pipe(): pass

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
    for i, j in zip(pishpish.reset(), itertools.count()):
        assert Message(i) == Message({'values': [j]})

    assert Message(i) == Message({'values': [19]})

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

# def test_recursive_decorator():
#
#     alto = recursion_dummy()
#     balto = recursion_dummy(alto)
#     calto = recursion_dummy(balto)
#
#     assert alto.height == 0 and alto.depth == 0
#     assert balto.height == 0  and balto.depth == 0
#     assert calto.height == 0  and calto.depth == 0
#
#     calto.jump()
#     assert calto.height == 1
#     assert balto.height == 1
#     assert alto.height == 1
#
#     balto.jump()
#     assert calto.height == 1
#     assert balto.height == 2
#     assert alto.height == 2
#
#     calto.somersault(1)
#     assert alto.depth == 1
#     assert balto.depth == 2
#     assert calto.depth == 3
#
#     balto.somersault(1)
#     assert alto.depth == 2
#     assert balto.depth == 4
#
#     assert False
