from fireworks.core.scaffold import Scaffold
from fireworks.utils.test_helpers import *
import os
from shutil import rmtree

test_dir = 'scaffold_test'
epsilon = .001

def test_Scaffold():
    D = RandomJunction(components={})
    C = reset_dummy()
    A = DummyModel({'m': [3.]}, out_column='y1', input=C)
    A.conv1 = torch.nn.Conv2d(4,4,4)
    B = DummyModel({'m': [1.], 'b': [2.]}, input=A, in_column='y1', out_column='y')
    cafe = Scaffold({'a':A, 'b': B, 'c': C, 'd': D})
    state = cafe.serialize()
    for key in ['a','b','c','d']:
        assert 'internal' in state[key]
        assert 'external' in state[key]

    try:
        rmtree(test_dir)
    except:
        pass

    os.mkdir(test_dir)
    cafe.save(path=test_dir, method='json')

    D2 = RandomJunction(components={})
    C2 = reset_dummy()
    A2 = DummyModel({'m': [4.]}, out_column='y1', input=C)
    A2.conv1 = torch.nn.Conv2d(4,4,4)
    B2 = DummyModel({'m': [5.], 'b': [6.]}, input=A, in_column='y3', out_column='y4')
    cafe2 = Scaffold({'a':A2,'b':B2,'c':C2,'d':D2})
    assert (A2.m == 4).all()
    assert (B2.m == 5).all()
    assert (B2.b == 6).all()
    assert not (A2.conv1.state_dict()['bias'] == A.conv1.state_dict()['bias']).all()
    assert not (A2.conv1.state_dict()['weight'] == A.conv1.state_dict()['weight']).all()
    assert B2.in_column == 'y3'
    assert B2.out_column == 'y4'
    cafe2.load(test_dir)
    assert (A2.m == 3).all()
    assert (B2.m == 1).all()
    assert (B2.b == 2).all()
    assert (A2.conv1.state_dict()['bias'] - A.conv1.state_dict()['bias'] < epsilon).all()
    assert (A2.conv1.state_dict()['weight'] - A.conv1.state_dict()['weight'] < epsilon).all()
    assert B2.in_column == 'y1'
    assert B2.out_column == 'y'
    rmtree(test_dir)
