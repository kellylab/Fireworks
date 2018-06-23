from Fireworks import cache

def test_slice_to_list():
    a = slice(0,10)
    l = cache.slice_to_list(a)
    assert l == [0,1,2,3,4,5,6,7,8,9]
    a = slice(0,10,2)
    l = cache.slice_to_list(a)
    assert l == [0,2,4,6,8]

def test_init():
    m = cache.MessageCache(10)
    assert m.max_size == 10

def test_insert(): pass

def test_free(): pass
