from Fireworks.extensions import training
from Fireworks.toolbox.pipes import BatchingPipe, ShufflerPipe
from Fireworks.utils.test_helpers import DummyModel, generate_linear_model_data
import torch

base_loss = torch.nn.MSELoss()
loss = lambda batch: base_loss(batch['y'], batch['y_true'])

def test_IgniteJunction():

    # Instantiate model
    model = DummyModel({'m': [1.]})
    # Instantiate dataste
    data, labels = generate_linear_model_data()
    batcher = BatchingPipe(ShufflerPipe(data), batch_size=50)
    # Instantiate engine
    junkie = training.IgniteJunction({'model': model, 'dataset': batcher}, loss=loss, optimizer='Adam')
    # Train and test that changes took place
    assert (model.m == 1.).all()
    assert (model.b == 0.).all()
    junkie.train()
    assert not (model.m == 1.).all()
    assert not (model.b == 0.).all()
