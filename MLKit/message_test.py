from MLKit import message

def test_Message():

    a = [1,2,3]
    b = ['a','b','c']

    # Test init
    email = message.Message({'a': a, 'b':b})

    #   Test error cases
    # TODO: test error cases

    # Test length
    assert len(email) == 3
    # Test getitem
    x = email[2]
    assert set(x.keys()) == set(['a','b'])
    assert x['a'] == [3]
    assert x['b'] == ['c']
    x = email[0:2]
    assert set(x.keys()) == set(['a','b'])
    assert x['a'] == [1,2]
    assert x['b'] == ['a','b']

    # Test for length 1 init
    gmail = message.Message({'a':1, 'b': 'hiii'})
    assert len(gmail) == 1
    y = gmail[0]
    assert set(y.keys()) == set(['a','b'])
    assert y['a'] == [1]
    assert y['b'] == ['hiii']

    # Test extend
    yahoomail = email.extend(gmail)
    assert len(yahoomail) == 4
    z = yahoomail[0:4]
    assert set(z.keys()) == set(['a','b'])
    assert z['a'] == [1,2,3,1]
    assert z['b'] == ['a','b','c','hiii']
