import Fireworks
import os
import pandas as pd
from Fireworks import junction as jn
from Fireworks.message import Message
from Fireworks.utils import index_to_list
from Fireworks.test_utils import *
import numpy as np
import math
import itertools

def test_junction():
    
    gumbo = one_way_dummy()

    angry = jn.RandomAggregatorJunction(inputs={'dumbo': dumbo, 'bumbo': bumbo, 'gumbo': gumbo})
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

    mangry = jn.ClockworkAggregatorJunction(inputs = {'dumbo': dumbo, 'bumbo': bumbo, 'gumbo': gumbo})
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
