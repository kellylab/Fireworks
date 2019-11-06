from fireworks import message as messi
from fireworks import Message, TensorMessage
import torch
import os
import numpy as np
from itertools import product
import pandas as pd
from itertools import count
from io import BytesIO
import pickle

tensors = {
    'a': torch.Tensor([1,2,3]),
    'b': torch.Tensor([4,5,6]),
    }
vectors = {
    'c': np.array([7,8,9]),
    'd': np.array([10,11,12]),
}
dtensors = {
    'a': torch.Tensor([[1,2,3],[4,5,6],[7,8,9]]),
    'b': torch.Tensor([[-1,-2,-3],[-4,-5,-6], [-7,-8,-9]]),
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

def test_complement():

    n = 10
    index = 7
    complement = messi.complement(index, n)
    assert complement == [0,1,2,3,4,5,6,8,9]
    index = slice(2,5)
    complement = messi.complement(index, n)
    assert complement == [0,1,5,6,7,8,9]
    index = [2,4,6]
    complement = messi.complement(index, n)
    assert complement == [0,1,3,5,7,8,9]

def test_Message():
    """ Test init, getitem, and len methopl. """

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

        assert (message['a'] == torch.Tensor([1,2,3])).all()
        assert message[['a','c']] == Message({'a': torch.Tensor([1,2,3]), 'c': np.array([7,8,9])})

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

def test_Message_from_objects():

    v = vectors.copy()
    t = tensors.copy()
    v['c'] = np.array([1.,2.])
    v['r'] = 'howdy'
    t['a'] = torch.randn(5)
    t['q'] = torch.randn([4,3])
    combined = {**t, **v}

    m = Message.from_objects(t, v)
    assert (set(m.keys()) == set(['c','d','r','b','a','q']))
    for key in ['c','d','b','a','q']:
        assert (m[key][0] == combined[key]).all()
    assert m['r'][0] == combined['r']
    assert len(m) == 1

def test_getitem():

    m = Message(tensors, vectors)
    assert m[0] == Message({'a': torch.Tensor([1]), 'b': torch.Tensor([4])}, {'c': np.array([7]), 'd': np.array([10])})
    assert m[[0,2]] == Message({'a': torch.Tensor([1,3]), 'b': torch.Tensor([4,6])}, {'c': np.array([7,9]), 'd': np.array([10,12])})
    # Check that out of bounds index calls raise errors
    try:
        m[3]
        assert False
    except IndexError:
        assert True

    try:
        m[3:5]
        assert False
    except IndexError:
        assert True

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
    assert (df == (pd.DataFrame({'c': vectors['c'], 'a': np.array(tensors['a'])}))).all().all()

def test_to_dataframe():

    mo = Message(tensors,vectors)
    # no = mo.to_dataframe()
    # assert no.tensor_message == {}
    # assert (no['a'] == mo['a']).all()
    # assert (no['b'] == mo['b']).all()
    # for letter in ['a','b','c','d']:
    #     assert letter in no.df
    lo = Message(dtensors, vectors)
    ok = lo.to_dataframe()
    for i in range(3):
        assert (ok['a'][i] == dtensors['a'][i].numpy()).all()
        assert (ok['b'][i] == dtensors['b'][i].numpy()).all()

def test_cpu_gpu():
    m = Message(tensors, vectors)
    m.cpu()
    assert set(m.tensors().keys()) == set(['a','b'])
    for key, tensor in m.tensors().items():
        assert tensor.device.type == 'cpu'
    if torch.cuda.is_available():
        m.cuda()
        for key, tensor in m.tensors().items():
            assert tensor.device.type == 'cuda'
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

    m0 = Message()
    assert(len(m0) == 0)
    m = m0.append(Message(t))
    assert m == Message(t)
    m = m0.append(Message(v))
    assert m == Message(v)
    m = m0.append(Message(t,v))
    assert m == Message(t,v)

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

    # Test point updates
    email = Message(tensors, vectors)
    gmail = Message({'a':torch.Tensor([1,42,3]), 'b':torch.Tensor([4,43,6]), 'c': np.array([7,99,9]), 'd': np.array([10,100,12])})
    replacement = {'a': torch.Tensor([42]), 'b': torch.Tensor([43]), 'c': np.array([99]), 'd': np.array([100])}
    assert len(email) == 3
    assert email != gmail
    email[1] = replacement
    assert email == gmail
    # Test ranged updates
    email = Message(tensors, vectors)
    gmail = Message({'a':torch.Tensor([1,42,33]), 'b':torch.Tensor([4,43,66]), 'c': np.array([7,99,99]), 'd': np.array([10,100,122])})
    replacement = {'a': torch.Tensor([42,33]), 'b': torch.Tensor([43,66]), 'c': np.array([99,99]), 'd': np.array([100,122])}
    assert email != gmail
    email[1:3] = replacement
    assert email == gmail
    # Test updates using lists as indexes
    email = Message(tensors, vectors)
    assert email != gmail
    email[[1,2]] = replacement
    assert email == gmail
    # Test column updates
    email['a'] = torch.Tensor([9,9,9])
    assert torch.equal(email['a'], torch.Tensor([9,9,9]))
    email['c'] = np.array([9,9,9])
    assert email['c'].equals(pd.Series([9,9,9]))
    # Test column updates that switch from df to tensor and vice-versa
    email = Message(tensors, vectors)
    assert set(email.columns) == set(['a','b','c','d'])
    assert set(email.tensor_message.columns) == set(['a','b'])
    assert set(email.df.columns) == set(['c','d'])
    new_a = np.array([1,2,3]) # Switch from tensor to vector
    email['a'] = new_a
    assert set(email.columns) == set(['a','b','c','d'])
    assert set(email.tensor_message.columns) == set(['b'])
    assert set(email.df.columns) == set(['a','c','d'])
    assert (email['a'] == new_a).all()
    new_c = torch.Tensor([7,8,9])
    email['c'] = new_c
    assert set(email.columns) == set(['a','b','c','d'])
    assert set(email.tensor_message.columns) == set(['b','c'])
    assert set(email.df.columns) == set(['a','d'])
    assert (email['c'] == new_c).all()
    # Test column updates that end up clearing either self.df or self.tensor_message
    email = Message(tensors, vectors)
    df = email.dataframe(['a', 'b'])
    assert len(email) == 3
    assert len(email.tensor_message) == 3
    assert len(email.df) == 3
    email[['a','b']] = df
    assert len(email) == 3
    assert len(email.tensor_message) == 0
    assert len(email.df) == 3
    # TODO: Test the other way around

def test_Message_del():

    t = {
        'a': torch.Tensor([1,2,3]),
        'b': torch.Tensor([4,5,6]),
        }
    v = {
        'c': np.array([7,8,9]),
        'd': np.array([10,11,12]),
    }
    t2 = {
        'a': torch.Tensor([1,2]),
        'b': torch.Tensor([4,5]),
        }
    v2 = {
        'c': np.array([7,8]),
        'd': np.array([10,11]),
    }
    t3 = {
        'a': torch.Tensor([1]),
        'b': torch.Tensor([4]),
        }
    v3 = {
        'c': np.array([7]),
        'd': np.array([10]),
    }

    # Test deletions for messages with only tensors, only df, and both
    # Test point deletions
    m = Message(t,v)
    m1 = Message(t)
    m2 = Message(v)
    assert m != Message(t2,v2)
    assert m1 != Message(t2)
    assert m2 != Message(v2)
    assert len(m) == 3
    assert len(m1) == 3
    assert len(m2) == 3
    del m[2]
    del m1[2]
    del m2[2]
    assert len(m) == 2
    assert len(m1) == 2
    assert len(m2) == 2
    assert m == Message(t2,v2)
    assert m1 == Message(t2)
    assert m2 == Message(v2)
    # Test range deletions
    m = Message(t,v)
    m1 = Message(t)
    m2 = Message(v)
    assert m != Message(t3,v3)
    assert m1 != Message(t3)
    assert m2 != Message(v3)
    assert len(m) == 3
    assert len(m1) == 3
    assert len(m2) == 3
    del m[1:3]
    del m1[1:3]
    del m2[1:3]
    assert len(m) == 1
    assert len(m1) == 1
    assert len(m2) == 1
    assert m == Message(t3,v3)
    assert m1 == Message(t3)
    assert m2 == Message(v3)
    # Test list deletions
    m = Message(t,v)
    m1 = Message(t)
    m2 = Message(v)
    assert m != Message(t3,v3)
    assert m1 != Message(t3)
    assert m2 != Message(v3)
    assert len(m) == 3
    assert len(m1) == 3
    assert len(m2) == 3
    del m[[1,2]]
    del m1[[1,2]]
    del m2[[1,2]]
    assert len(m) == 1
    assert len(m1) == 1
    assert len(m2) == 1
    assert m == Message(t3,v3)
    assert m1 == Message(t3)
    assert m2 == Message(v3)

    # Test column deletions
    m = Message(t,v)
    assert set(m.columns) == set(['a','b','c','d'])
    del m['a']
    assert set(m.columns)== set(['b','c','d'])
    del m['c']
    assert set(m.columns) == set(['b','d'])

def test_Message_iter():

    m = Message(tensors, vectors)
    l = len(m)
    for x,i in zip(m, count()):
        assert type(x) is Message
        if i > l:
            assert False
    assert i == l - 1

    t = TensorMessage(tensors)
    l = len(t)
    for x,i in zip(t, count()):
        assert type(x) is TensorMessage
        if i > l:
            assert False
    assert i == l - 1

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

def test_TensorMessage_set_get_del():

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
    assert gmail != yahoomail
    del gmail[0]
    assert len(gmail) == 2
    assert gmail == yahoomail
    # Test column deletions
    email = TensorMessage({'a': a, 'b':b})
    assert set(email.columns) == set(['a','b'])
    del email['a']
    assert set(email.columns) == set(['b'])
    # Test that out of bounds requests raise errors
    try:
        email[3]
        assert False
    except IndexError:
        assert True
    try:
        email[3:5]
        assert False
    except IndexError:
        assert True

    # Test length adjustment if all columns are deleted
    zohomail = TensorMessage({'a':a,'b':b})
    assert len(zohomail) == 3
    del zohomail['a']
    assert len(zohomail) == 3
    del zohomail['b']
    assert len(zohomail) == 0

def test_TensorMessage_eq():

    a = [1,2,3]
    b = [4, 5, 6]

    # Test init
    email = TensorMessage({'a': a, 'b':b})
    gmail = TensorMessage(email)

def test_cat():

    m = Message(tensors, vectors)
    m0 = m[0]
    m1 = m[1]
    m2 = m[2]
    babaghanush = messi.cat([m0,m1,m2])
    assert babaghanush == m

def test_TensorMessage_permute():
    a = [1,2,3]
    b = [4, 5, 6]
    email = TensorMessage({'a': a, 'b':b})
    gmail = email.permute([2,1,0])
    assert gmail == TensorMessage({'a':[3,2,1], 'b':[6,5,4]})
    gmail = email.permute([0,0,0])
    assert gmail == TensorMessage({'a':[1,1,1], 'b':[4,4,4]})

def test_permute():

    tensors = {
        'a': torch.Tensor([1,2,3]),
        'b': torch.Tensor([4,5,6]),
        }
    vectors = {
        'c': np.array([7,8,9]),
        'd': np.array([10,11,12]),
    }
    email = Message(tensors, vectors)
    gmail = email.permute([2,1,0])
    assert gmail == Message({'a':[3,2,1], 'b':[6,5,4]}, {'c': np.array([9,8,7]), 'd': np.array([12,11,10])})
    gmail = email.permute([0,0,0])
    assert gmail == Message({'a':[1,1,1], 'b':[4,4,4]}, {'c': np.array([7,7,7]), 'd': np.array([10,10,10])})
    # Test with only tensors
    email = Message(tensors)
    gmail = email.permute([2,1,0])
    assert gmail == Message({'a':[3,2,1], 'b':[6,5,4]}, {})
    gmail = email.permute([0,0,0])
    assert gmail == Message({'a':[1,1,1], 'b':[4,4,4]}, {})
    # Test with only dataframes
    email = Message(vectors)
    gmail = email.permute([2,1,0])
    assert gmail == Message({'c': np.array([9,8,7]), 'd': np.array([12,11,10])})
    gmail = email.permute([0,0,0])
    assert gmail == Message({'c': np.array([7,7,7]), 'd': np.array([10,10,10])})

def test_to_csv():
    m = Message(tensors, vectors)
    pass #TODO: Implement

def test_to_pickle():
    m = Message(tensors, vectors)
    pass #TODO: Implement

def test_to_sql():
    m = Message(tensors, vectors)
    pass #TODO: Implement

def test_to_dict():
    m = Message(tensors, vectors)
    md = m.to_dict()
    assert type(md) is dict
    assert (md['c'] == md['c'])
    assert (md['d'] == md['d'])
    assert (md['a'] == np.array(md['a'])).all()
    assert (md['b'] == np.array(md['b'])).all()

def test_to_excel():
    m = Message(tensors, vectors)
    pass #TODO: Implement

def test_to_json():
    m = Message(tensors, vectors)
    pass #TODO: Implement

def test_to_string():
    m = Message(tensors, vectors)
    pass #TODO: Implement

def test_save_load():
    m = Message(tensors, vectors)
    test_path = 'test.fireworks'
    m.save(test_path)
    new_m = Message.load(test_path)
    assert new_m == m
    os.remove(test_path)
    buffer = BytesIO()
    m.save(buffer)
    buffed_m = Message.load(buffer)
    assert buffed_m == m 

def test_pickle():
    m = Message(tensors, vectors)
    state = pickle.dumps(m)
    new_m = pickle.loads(state)
    assert new_m == m