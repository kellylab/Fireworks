from MLKit import message
import torch

def test_Message():

    a = [1,2,3]
    b = [4, 5, 6]

    # Test init
    email = message.Message({'a': a, 'b':b})

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
    gmail = message.Message({'a':1, 'b': 80})
    assert len(gmail) == 1
    y = gmail[0]
    assert set(y.keys()) == set(['a','b'])
    assert (y['a'] == torch.Tensor([1])).all()
    assert (y['b'] == torch.Tensor([80])).all()

    # Test extend
    yahoomail = email.extend(gmail)
    assert len(yahoomail) == 4
    z = yahoomail[0:4]
    assert set(z.keys()) == set(['a','b'])
    assert (z['a'] == torch.Tensor([1,2,3,1])).all()
    assert (z['b'] == torch.Tensor([4, 5, 6, 80])).all()
