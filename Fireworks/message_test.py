from Fireworks import message as messi
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

def test_compute_length():

    l = messi.compute_length(tensors)
    assert l == 3
    l = messi.compute_length(vectors)
    assert l == 3

def test_extract_tensors():

    target = {**tensors, **vectors}
    t, v = messi.extract_tensors(target)
    assert t == tensors
    assert v == vectors
    t, v = messi.extract_tensors(tensors)
    assert t == tensors
    assert v == {}
    t, v = messi.extract_tensors(vectors)
    assert t == {}
    assert v == vectors

def test_Message():
    """ Test init, getitem, and len methods. """

    def attribute_test(message, length = 3):
        assert len(message) == length
        assert message[0].tensors() == {
            'a': torch.Tensor([1]),
            'b': torch.Tensor([4]),
        }
        assert message[0].dataframe().equals(pd.DataFrame({
            'c': np.array([7]),
            'd': np.array([10]),
        }))

        assert message[0] == Message({'a': torch.Tensor([1]),'b': torch.Tensor([4])}, pd.DataFrame({'c': np.array([7]),'d': np.array([10]),}))

        assert message[1:3].tensors() == {
            'a': torch.Tensor([2,3]),
            'b': torch.Tensor([5,6]),
        }
        assert message[1:3].dataframe().equals(pd.DataFrame({
            'c': np.array([8,9]),
            'd': np.array([11,12]),
        }))

        assert message[1:3] == Message({'a': torch.Tensor([2,3]),'b': torch.Tensor([5,6])}, pd.DataFrame({'c': np.array([8,9]),'d': np.array([11,12])}))

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

    # Test one sided Messages
    for t in tensor_options:
        m = Message(t, None)
        assert len(m) == 3
        assert m == Message(tensors)
    for v in vector_options:
        m = Message(None, v)
        assert len(m) == 3
        assert m == Message(vectors)

    # Init message from a single dict
    everything = {**tensors, **vectors}
    m = Message(everything)
    attribute_test(m)

def test_getitem():

    m = Message(tensors, vectors)
    assert m[0] == Message({'a': torch.Tensor([1]), 'b': torch.Tensor([4])}, {'c': np.array([7]), 'd': np.array([10])})
    assert m[[0,2]] == Message({'a': torch.Tensor([1,3]), 'b': torch.Tensor([4,6])}, {'c': np.array([7,9]), 'd': np.array([10,12])})


def test_cache(): pass

def test_tensors():
    m = Message(tensors, vectors)
    t = m.tensors()
    assert t == TensorMessage(tensors)
    t = m.tensors(keys=['a'])
    assert t == TensorMessage({'a': tensors['a']})
    t = m.tensors(keys=['a','c'])
    assert t == TensorMessage({'a': tensors['a'], 'c': torch.Tensor(vectors['c'])})

def test_df():
    m = Message(tensors, vectors)
    df = m.dataframe()
    assert df.equals(pd.DataFrame(vectors))
    df = m.dataframe(keys=['c'])
    assert df.equals(pd.DataFrame({'c': vectors['c']}))
    df = m.dataframe(keys=['c','a'])
    assert df.equals(pd.DataFrame({'c': vectors['c'], 'a': np.array(tensors['a'])}))

def test_cpu_gpu():
    m = Message(tensors, vectors)
    m.cpu()
    assert set(m.tensors().keys()) == set(['a','b'])
    for key, tensor in m.tensors().items():
        assert tensor.device.type == 'cpu'
    if torch.cuda.is_available():
        m.cuda()
        for key, tensor in m.tensors().items():
            assert tensor.device.type == 'gpu'
        m.cpu()
        for key, tensor in m.tensors().items():
            assert tensor.device.type == 'cpu'

def test_append():

    t = tensors
    v = vectors
    m1 = Message(t, v)
    m2 = Message(t, v)
    m3 = Message(t)
    m4 = TensorMessage(t)
    m5 = Message(pd.DataFrame(v))
    m6 = pd.DataFrame(v)

    m = m1.append(m2)
    assert len(m) == 6
    assert m == Message({'a': torch.Tensor([1,2,3,1,2,3]), 'b': torch.Tensor([4,5,6,4,5,6])}, {'c': np.array([7,8,9,7,8,9]), 'd': np.array([10,11,12,10,11,12])})
    m = m3.append(t)
    assert len(m) == 6
    assert m == Message({'a': torch.Tensor([1,2,3,1,2,3]), 'b': torch.Tensor([4,5,6,4,5,6])})
    m = m3.append(m3)
    assert len(m) == 6
    assert m == Message({'a': torch.Tensor([1,2,3,1,2,3]), 'b': torch.Tensor([4,5,6,4,5,6])})
    m = m3.append(m4)
    assert len(m) == 6
    assert m == Message({'a': torch.Tensor([1,2,3,1,2,3]), 'b': torch.Tensor([4,5,6,4,5,6])})

    m = m4.append(t)
    assert len(m) == 6
    assert m == TensorMessage({'a': torch.Tensor([1,2,3,1,2,3]), 'b': torch.Tensor([4,5,6,4,5,6])})
    m = m4.append(m3)
    assert len(m) == 6
    assert m == TensorMessage({'a': torch.Tensor([1,2,3,1,2,3]), 'b': torch.Tensor([4,5,6,4,5,6])})
    m = m4.append(m4)
    assert len(m) == 6
    assert m == TensorMessage({'a': torch.Tensor([1,2,3,1,2,3]), 'b': torch.Tensor([4,5,6,4,5,6])})

    m = m5.append(v)
    assert len(m) == 6
    assert m == Message({'c': np.array([7,8,9,7,8,9]), 'd': np.array([10,11,12,10,11,12])})
    m = m5.append(m5)
    assert len(m) == 6
    assert m == Message({'c': np.array([7,8,9,7,8,9]), 'd': np.array([10,11,12,10,11,12])})
    m = m5.append(m6)
    assert len(m) == 6
    assert m == Message({'c': np.array([7,8,9,7,8,9]), 'd': np.array([10,11,12,10,11,12])})

    # Test type conversions on appending to TensorMessage
    m = m4.append({'a': np.array([42]), 'b': np.array([24])})
    assert len(m) == 4
    assert m == TensorMessage({'a': torch.Tensor([1,2,3,42]), 'b': torch.Tensor([4,5,6,24])})


def test_join():

    t = tensors
    v = vectors
    t2 = {'d': torch.Tensor([13,14,15])}
    v2 = {'e': np.array([16,17,18])}
    m1 = Message(t,v)
    m2 = Message(t)
    m2_t = TensorMessage(t)
    m3 = Message(v)
    m4 = Message(t2,v2)
    m5 = Message(t2)
    m5_t = TensorMessage(t2)
    m6 = Message(v2)
    m7 = Message(t,v2)
    m8 = Message(t2, v)

    # Test if a tensor message can be merged into a message and vice versa
    assert m2.merge(m3) == m1
    assert m3.merge(m2) == m1
    assert m3.merge(m2_t) == m1
    assert m3.merge(t) == m1

    # Test if the tensors in messages can be merged
    assert m2.merge(t2) == Message({**t, **t2})
    assert m2.merge(m5) == Message({**t, **t2})
    assert m2.merge(m5_t) == Message({**t, **t2})
    assert m2_t.merge(t2) == TensorMessage({**t, **t2})
    assert m2_t.merge(m5) == TensorMessage({**t, **t2})
    assert m2_t.merge(m5_t) == TensorMessage({**t, **t2})

    # Test if the dataframes in messages can be merged
    assert m3.merge(m6) == Message({**v, **v2})
    assert m6.merge(m3) == Message({**v, **v2})
    assert m3.merge(v2) == Message({**v, **v2})

def test_Message_set_get():

    email = Message(tensors, vectors)
    gmail = Message({'a':torch.Tensor([42,2,3]), 'b':torch.Tensor([43,5,6]), 'c': np.array([99,8,9]), 'd': np.array([100,11,12])})
    replacement = {'a': torch.Tensor([42]), 'b': torch.Tensor([43]), 'c': np.array([99]), 'd': np.array([100])}
    assert len(email) == 3
    assert email != gmail
    email[0] = replacement
    assert email == gmail
    email['a'] = torch.Tensor([9,9,9])
    assert torch.equal(email['a'], torch.Tensor([9,9,9]))
    email['c'] = np.array([9,9,9])
    assert email['c'].equals(pd.Series([9,9,9]))

def test_map(): pass

def test_TensorMessage():

    a = [1,2,3]
    b = [4, 5, 6]

    # Test init
    empty = TensorMessage()
    assert len(empty) == 0
    assert empty.keys() == {}.keys()
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


def test_TensorMessage_set_get():

    a = [1,2,3]
    b = [4, 5, 6]
    email = TensorMessage({'a': a, 'b':b})
    replacement = {'a': torch.Tensor([42]), 'b': torch.Tensor([43])}
    gmail = TensorMessage({'a':torch.Tensor([42,2,3]), 'b':  torch.Tensor([43,5,6])})
    yahoomail = TensorMessage({'a':torch.Tensor([2,3]), 'b': torch.Tensor([5,6])})
    assert email != gmail
    email[0] = replacement
    assert email == gmail
    assert len(email) == 3
    email['a'] = torch.Tensor([9,9,9])
    assert torch.equal(email['a'], torch.Tensor([9,9,9]))
    # del email[0]
    # assert len(email) == 2
    # assert email == yahoomail

def test_TensorMessage_eq():

    a = [1,2,3]
    b = [4, 5, 6]

    # Test init
    email = TensorMessage({'a': a, 'b':b})
    gmail = TensorMessage(email)
