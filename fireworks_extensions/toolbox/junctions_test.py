import fireworks
import os
from fireworks.toolbox import junctions as jn
from fireworks import Message
from fireworks.utils import index_to_list
from fireworks.utils.test_helpers import *
import numpy as np
import itertools

def test_junction():

    dumbo = one_way_dummy()
    bumbo = one_way_dummy()
    gumbo = one_way_dummy()

    angry = jn.RandomHubJunction(components={'dumbo': dumbo, 'bumbo': bumbo, 'gumbo': gumbo})
    angry.reset()
    assert angry._available_inputs == set(['dumbo', 'bumbo', 'gumbo'])
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

    mangry = jn.ClockworkHubJunction(components = {'dumbo': dumbo, 'bumbo': bumbo, 'gumbo': gumbo})
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

def test_RandomHubJunction():

    a = Message({'x':np.random.rand(100)})
    b = Message({'x':np.random.rand(100)})
    rob = jn.RandomHubJunction(components={'a':a, 'b':b})
    i = 0
    for x in rob:
        i+=1
    assert i == 200
    for x in rob:
        i+=1
    assert i == 400
