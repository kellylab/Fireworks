import numpy as np
import math
from Fireworks import preprocessing as pr
from Fireworks import Source

def test_one_hot():

    hot = pr.one_hot(2, 10)
    assert (hot == np.array([0,0,1,0,0,0,0,0,0,0])).all()
