from fireworks.core.component_map import Component_Map, PyTorch_Component_Map
import torch

class DummyClass:

    def __init__(self):
        self.x = 1

def test_setitem_delitem():

    dummy = DummyClass()
    mapper = Component_Map({'a': 2, 'b': 3, 'c': (dummy,'x')})
    assert mapper['a'] == 2
    assert mapper['b'] == 3
    assert mapper['c'] == 1
    assert 'a' in mapper._internal_components and 'a' not in mapper._external_modules and 'a' not in mapper._external_attribute_names
    assert 'b' in mapper._internal_components and 'b' not in mapper._external_modules and 'b' not in mapper._external_attribute_names
    assert 'c' not in mapper._internal_components and 'c' in mapper._external_modules and 'c' in mapper._external_attribute_names
    dummy.x = 2
    assert mapper['c'] == 2
    mapper['d'] = (4, 3)
    assert mapper['d'] == (4,3)
    assert 'd' in mapper._internal_components and 'd' not in mapper._external_modules and 'd' not in mapper._external_attribute_names
    new_dummy = DummyClass()
    mapper['e'] = (new_dummy, 'x')
    assert mapper['e'] == 1
    assert 'e' not in mapper._internal_components and 'e' in mapper._external_modules and 'e' in mapper._external_attribute_names
    del mapper['c']
    assert 'c' not in mapper and 'c' not in mapper._external_modules and 'c' not in mapper._external_attribute_names
    del mapper['a']
    assert 'a' not in mapper and 'a' not in mapper._internal_components

    # Test overwrites from external to internal.
    mapper['c'] = 3
    assert mapper['c'] == 3
    assert 'c' in mapper._internal_components and 'c' not in mapper._external_modules and 'c' not in mapper._external_attribute_names
    mapper['a'] = (dummy, 'x')
    assert mapper['a'] == 2
    assert 'a' not in mapper._internal_components and 'a' in mapper._external_modules and 'a' in mapper._external_attribute_names


def test_pytorch_component_map():

    dummy = DummyClass()
    mapper = PyTorch_Component_Map({'a': [1.,2.,3.], 'b': 3, 'c': 'ok', 'd': (dummy, 'x')})
    assert isinstance(mapper['a'], torch.nn.Parameter)
    assert (mapper['a'] == torch.Tensor([1,2,3])).all()
    assert mapper['b'] == 3
    assert mapper['c'] == 'ok'
    assert mapper['d'] == 1
    dummy.x = 2
    assert mapper['d'] == 2
    del mapper['a']
    assert 'a' not in mapper

def test_setstate_getstate():

    dummy = DummyClass()
    mapper = Component_Map({'a': 2, 'b': 3, 'c': (dummy,'x')})
    state = mapper.get_state()
    assert state['internal']['a'] == 2
    assert state['internal']['b'] == 3
    assert state['external']['c'] == (dummy, 'x')
    new_dummy = DummyClass()
    new_state = {'internal': {'a': 4, 'c': 5}, 'external': {'b': (new_dummy, 'x')}}
    mapper.set_state(new_state)
    assert mapper['a'] == 4
    assert mapper['c'] == 5
    assert mapper['b'] == 1
    new_dummy.x = 2
    assert mapper['b'] == 2
