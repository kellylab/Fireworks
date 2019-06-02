from fireworks import cache
from fireworks import Message, TensorMessage
import torch
import numpy as np
import pandas as pd

tensors = {
    'a': torch.Tensor([1,2,3]),
    'b': torch.Tensor([4,5,6]),
    }
vectors = {
    'c': np.array([7,8,9]),
    'd': np.array([10,11,12]),
}
tensors2 = {
    'a': torch.Tensor([13,14,15]),
    'b': torch.Tensor([16,17,18]),
    }
vectors2 = {
    'c': np.array([19,20,21]),
    'd': np.array([22,23,24]),
}

def test_pointer_adjustment():

    r = 5
    f = cache.pointer_adjustment_function(r)
    assert f(4) == 0
    assert f(5) == 1
    assert f(6) == 1
    r = [2,4,6]
    f = cache.pointer_adjustment_function(r)
    assert f(0) == 0
    assert f(2) == 1
    assert f(5) == 2
    assert f(6) == 3
    r = slice(3,6)
    f = cache.pointer_adjustment_function(r)
    assert f(0) == 0
    assert f(3) == 1
    assert f(4) == 2
    assert f(5) == 3

def test_slice_to_list():
    a = slice(0,10)
    l = cache.slice_to_list(a)
    assert l == [0,1,2,3,4,5,6,7,8,9]
    a = slice(0,10,2)
    l = cache.slice_to_list(a)
    assert l == [0,2,4,6,8]

def test_init_set():
    m = cache.UnlimitedCache()
    dummy_message = Message(tensors, vectors)
    assert len(m) == 0
    m[0] = dummy_message[0]
    assert len(m) == 1
    m[3:6] = dummy_message
    assert len(m) == 4
    assert m[0] == dummy_message[0]
    assert m[3:6] == dummy_message
    dummy_2 = Message(tensors2, vectors2)
    m[3:6] = dummy_2
    assert len(m) == 4
    assert m[0] == dummy_message[0]
    assert m[3:6] == dummy_2
    m[7:10] = dummy_message
    assert m[7:10] == dummy_message
    
def test_del():
    m = cache.UnlimitedCache()
    dummy_message = Message(tensors, vectors)
    m[0] = dummy_message[0]
    m[3:6] = dummy_message
    assert 0 in m
    assert 3 in m
    assert 4 in m
    assert 5 in m
    assert m[0] == dummy_message[0]
    # del m[0]
    # assert not 0 in m
    del m[3:5]
    assert m[5] == dummy_message[2]
    assert not 3 in m
    assert not 4 in m
    m[3:6] = dummy_message
    # assert False
    pointers = m.pointers.copy()
    del m[5:8]
    assert not 5 in m
    assert not 6 in m
    assert m[3:5] == dummy_message[0:2]

def test_permute():
    m = cache.UnlimitedCache()
    dummy_message = Message(tensors, vectors)
    m[3:6] = dummy_message
    assert m.cache == dummy_message
    m._permute([2,1,0])
    assert m.cache != dummy_message
    m._permute([2,1,0])
    assert m.cache == dummy_message

def test_LRUCache():
    m = cache.LRUCache(10, buffer_size=2)
    dummy_message = Message(tensors, vectors)
    m[2:5] = dummy_message
    assert m[2:5] == dummy_message
    assert m.rank_dict.keys() == m.pointers.keys()
    m[7:10] = dummy_message
    assert m[2:5] == dummy_message
    assert m[7:10] == dummy_message
    assert m.rank_dict.keys() == m.pointers.keys()
    m[12:15] = dummy_message
    assert m[2:5] == dummy_message
    assert m[7:10] == dummy_message
    assert m[12:15] == dummy_message
    assert m.rank_dict.keys() == m.pointers.keys()
    # At this point, the least recently used elements are in the beginning
    m[15:18] = dummy_message
    assert m[15:18] == dummy_message
    assert len(m) == 10
    assert m.rank_dict.keys() == m.pointers.keys()
    assert not 2 in m
    assert not 3 in m
    assert set([4,7,8,9,12,13,14,15,16,17]) == set(m.rank_dict.keys())
    m[[4,8,9,13,14,15,16]] # Trigger __getitem__; Now 7, 12 and 17 should be queueud for deletion
    m[18:21] = dummy_message
    assert len(m) == 10
    for i in [7,12,17]:
        assert not i in m
    # Trigger a buffer clearance.
    m[29] = dummy_message[0]
    assert len(m) == 9

def test_LFUCache():
    m = cache.LFUCache(10, buffer_size=2)
    dummy_message = Message(tensors, vectors)
    m[2:5] = dummy_message
    assert m[2:5] == dummy_message
    assert m.rank_dict.keys() == m.pointers.keys()
    m[7:10] = dummy_message
    assert m[2:5] == dummy_message
    assert m[7:10] == dummy_message
    assert m.rank_dict.keys() == m.pointers.keys()
    m[12:15] = dummy_message
    assert m[2:5] == dummy_message
    assert m[7:10] == dummy_message
    assert m[12:15] == dummy_message
    assert m.rank_dict.keys() == m.pointers.keys()
    # At this point, 12:15 are the least frequently used elements.
    m[15:18] = dummy_message
    assert m[15:18] == dummy_message
    assert len(m) == 10
    assert m.rank_dict.keys() == m.pointers.keys()
    assert not 12 in m
    assert not 13 in m
    assert set([2,3, 4,7,8,9,14,15,16,17]) == set(m.rank_dict.keys())
    # Trigger __getitem__
    m[29] = dummy_message[0]
    assert len(m) == 9
