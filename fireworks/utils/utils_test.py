from . import utils
import numpy as np

def test_one_hot():

    hot = utils.one_hot(2, 10)
    assert (hot == np.array([0,0,1,0,0,0,0,0,0,0])).all()

def test_subset_dict():

    d1 = {'a':1, 'b':2, 'c':3, 'd': 4}
    d2 = utils.subset_dict(d1, ['b', 'c', 'e'])
    assert d2['b'] == d1['b']
    assert d2['c'] == d2['c']
    assert 'e' not in d2

    d3 = utils.subset_dict(d1, ['f'])
    assert d3 == {}
