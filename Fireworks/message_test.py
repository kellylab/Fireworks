from Fireworks.message import Message, TensorMessage
import torch
import numpy as np
from itertools import product
import pandas as pd

tensors = {
    'a': torch.Tensor([1,2,3]),
    'b': torch.Tensor([4,5,6]),
    }
vectors = {
    'c': np.array([7,8,9]),
    'd': np.array([10,11,12]),
}

def test_Message():
    """ Test init, getitem, and len methods. """

    def attribute_test(message, length = 3):
        assert len(message) == length
        assert message[0].tensors() == {
            'a': torch.Tensor([1]),
            'b': torch.Tensor([4]),
        }
        assert message[0].dataframe() == pd.DataFrame({
            'c': np.array([7]),
            'd': np.array([10]),
        })
        assert message[0] == Message({'a': torch.Tensor([1]),'b': torch.Tensor([4])}, pd.DataFrame({'c': np.array([7]),'d': np.array([10]),}))

        assert message[1:2].tensors() == {
            'a': torch.Tensor([2,3]),
            'b': torch.Tensor([5,6]),
        }
        assert message[1:2].dataframe() == pd.DataFrame({
            'c': np.array([8,9]),
            'd': np.array([11,12]),
        })
        assert message[1:2] == Message({'a': torch.Tensor([2,3]),'b': torch.Tensor([5,6])}, pd.DataFrame({'c': np.array([8,9]),'d': np.array([11,12])}))

        # Test length
        assert len(message) == length
        # Test __getitem__


    # Init empty message
    m = Message()
    assert len(m) == 0
    # Init message from tensor_dict / TensorMessage and dict of arrays / dataframe using positional arguments.
    tensor_message = TensorMessage(tensors)
    tensor_as_message = Message(tensors = tensors)
    df = pd.DataFrame(vectors)
    df_as_message = Message(df = vectors)

    # Try every combination
    tensor_options = [tensors, tensor_message, tensor_as_message]
    vector_options = [vectors, df, df_as_message]

    for t, v in product(tensor_options, vector_options):
        m = Message(t, v)
        attribute_test(m)
        m = Message(tensors = t, df = v)
        attribute_test(m)

    # Init message from a single dict
    everything = {**tensors, **vectors}
    m = Message(everything)
    attribute_test(m)

def test_cache(): pass

def test_tensors():
    m = Message(tensors, vectors)
    t = m.tensors()
    assert t == TensorMessage(tensors)
    t = m.tensors(keys=['a'])
    assert t == TensorMessage({'a': tensors['a']})

def test_df():
    m = Message(tensors, vectors)
    df = m.df()
    assert df == pd.DataFrame(vectors)
    df = m.df(keys=['c'])
    assert df == pd.DataFrame({'c': vectors['c']})

def test_cpu_gpu():
    m = Message(tensors, vectors)
    m.cpu()
    assert set(m.tensors().keys()) == set(['a','b'])
    for key, tensor in m.tensors():
        assert tensor.device.type == 'cpu'
    if torch.cuda.is_available():
        m.cuda()
        for key, tensor in m.tensors():
            assert tensor.device.type == 'gpu'
        m.cpu()
        for key, tensor in m.tensors():
            assert tensor.device.type == 'cpu'

def test_append(): pass

def test_join(): pass

def test_map(): pass

def test_TensorMessage():

    a = [1,2,3]
    b = [4, 5, 6]

    # Test init
    email = TensorMessage({'a': a, 'b':b})

    #   Test error cases
    # TODO: test error cases

    # Test length
    assert len(email) == 3
    # Test getitem
    x = email[2]
    assert set(x.keys()) == set(['a','b'])
    assert (x['a'] == torch.Tensor([3])).all()
    assert (x['b'] == torch.Tensor([6])).all()
    x = email[0:2]
    assert set(x.keys()) == set(['a','b'])
    assert (x['a'] == torch.Tensor([1,2])).all()
    assert (x['b'] == torch.Tensor([4,5])).all()

    # Test for length 1 init
    gmail = TensorMessage({'a':1, 'b': 80})
    assert len(gmail) == 1
    y = gmail[0]
    assert set(y.keys()) == set(['a','b'])
    assert (y['a'] == torch.Tensor([1])).all()
    assert (y['b'] == torch.Tensor([80])).all()

    # Test extend
    yahoomail = email.append(gmail)
    assert len(yahoomail) == 4
    z = yahoomail[0:4]
    assert set(z.keys()) == set(['a','b'])
    assert (z['a'] == torch.Tensor([1,2,3,1])).all()
    assert (z['b'] == torch.Tensor([4, 5, 6, 80])).all()

def test_TensorMessage_eq():

    a = [1,2,3]
    b = [4, 5, 6]

    # Test init
    email = TensorMessage({'a': a, 'b':b})
    gmail = TensorMessage(email)
