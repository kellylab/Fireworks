import numpy as np
from fireworks import Message
from fireworks.toolbox import preprocessing as pr
from fireworks.toolbox import CachingPipe, TensorPipe
from fireworks.utils.test_helpers import generate_linear_model_data
import numpy as np
from .pipes import ShufflerPipe, BatchingPipe
import math
from itertools import count
import pickle
import torch 

def test_Normalizer():

    a = 10
    b = 3
    c = 20
    d = 4
    n = 1000
    data = Message({'ok': np.random.normal(a,b,n), 'good': np.random.normal(c,d,n)})
    shuffler = ShufflerPipe(data)
    batcher = TensorPipe(BatchingPipe(shuffler))
    normie = pr.Normalizer(input=batcher)
    normie.disable_inference()
    assert normie.count == 0
    if torch.cuda.is_available():
        normie.cuda()
    for batch in normie: # Loop through the dataset
        continue
    assert normie.count == n
    # assert normie.mean == {}
    # assert normie.variance == {}
    normie.compile()
    means = normie.mean
    variances = normie.variance
    assert (abs(means['ok'] - a) < .5).all()
    assert (abs(means['good'] - c) < .5).all()
    assert (abs(variances['ok'] - b**2) < 3).all()
    assert (abs(variances['good'] - d**2) < 6).all()
    normie.disable_updates()
    normie.enable_inference()
    batch = normie[0:100]
    batch = batch.to_dataframe()
    mok = np.mean(batch['ok'])
    vok = np.var(batch['ok'])
    mood = np.mean(batch['good'])
    vood = np.var(batch['good'])
    assert abs(mok )< .3
    assert abs(vok - 1 )< .4
    assert abs(mood )< .3
    assert abs(vok -1 )< .4
    normie2 = pr.Normalizer(input=normie)
    if torch.cuda.is_available():
        normie2.cuda()
    normie2.disable_inference()
    normie.enable_inference()
    normie.disable_updates()
    le = 0
    for batch in normie2:
        le += len(batch)
        continue
    normie2.compile()
    means = normie2.mean
    variances = normie2.variance
    assert (abs(means['ok']) < .5).all()
    assert (abs(means['good']) < .5).all()
    assert (abs(variances['ok'] - 1) < .5).all()
    assert (abs(variances['good'] - 1) < .5).all()

    assert normie.mean['good'] != normie2.mean['good']
    assert normie.mean['ok'] != normie2.mean['ok']
    state = normie.get_state()
    normie2.set_state(state, reset=False)
    assert normie.mean['good'] == normie2.mean['good']
    assert normie.mean['ok'] == normie2.mean['ok']

    # Test serialization
    state = normie.get_state()
    blarmie = pr.Normalizer()
    blarmie.set_state(state)
    for key in ['mean', 'variance', 'rolling_sum', 'rolling_squares']:
        assert blarmie.components['mean'].keys() == normie.components['mean'].keys()
        assert sum(np.array(list(blarmie.components['mean'].values())) - np.array(list(normie.components['mean'].values()))) <= .01

    pickle.dumps(state) # Confirm that this is serializable

def test_train_test_split():

    # Test using a Message as input
    data, metadata = generate_linear_model_data(50)
    train, test = pr.train_test_split(data)
    assert len(train) == 40
    assert len(test) == 10
    train[0:20]
    test[0:4]
    for train_row,i in zip(train,count()): # Check that training set and test set are different
        for test_row,j in zip(test, count()):
             assert (train_row != test_row)

    # Test using a Pipe as input
    cache = CachingPipe(data)
    train2, test2 = pr.train_test_split(cache)
    train2[0:20]
    test2[0:3]

    for train_row in train2: # Check that training set and test set are different
        for test_row in test2:
            assert train_row != test_row
