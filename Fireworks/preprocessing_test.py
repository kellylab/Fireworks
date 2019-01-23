import numpy as np
import math
from Fireworks import preprocessing as pr
import numpy as np
from Fireworks.pipeline import ShufflerPipe, BatchingPipe

def test_one_hot():

    hot = pr.one_hot(2, 10)
    assert (hot == np.array([0,0,1,0,0,0,0,0,0,0])).all()

def test_Normalizer():

    data = np.random.normal(13,5,200)
    shuffler = ShufflerPipe(data)
    batcher = BatchingPipe(shuffler)
    normie = pr.Normalizer(batcher)
    assert False 
