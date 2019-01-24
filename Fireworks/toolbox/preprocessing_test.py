import numpy as np
from Fireworks.toolbox import preprocessing as pr
import numpy as np
from .pipes import ShufflerPipe, BatchingPipe

def test_Normalizer():

    data = np.random.normal(13,5,200)
    shuffler = ShufflerPipe(data)
    batcher = BatchingPipe(shuffler)
    normie = pr.Normalizer(batcher)
    assert False
