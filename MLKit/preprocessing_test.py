import numpy as np

def test_one_hot():

    hot = du.one_hot(2, 10)
    assert (hot == np.array([0,0,1,0,0,0,0,0,0,0])).all()
