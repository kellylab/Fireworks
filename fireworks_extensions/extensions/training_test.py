from fireworks.extensions import training
from fireworks.toolbox.pipes import BatchingPipe, ShufflerPipe, FunctionPipe
from fireworks.utils.test_helpers import DummyModel, generate_linear_model_data
import torch

base_loss = torch.nn.MSELoss()
loss = lambda batch: base_loss(batch['y'], batch['y_true'])

def test_IgniteJunction():

    # Instantiate model
    model = DummyModel({'m': [1.]})
    # Instantiate dataste
    data, labels = generate_linear_model_data()
    to_tensor = lambda m: m.to_tensors()
    batcher = \
        FunctionPipe(
            BatchingPipe(
                ShufflerPipe(data),
            batch_size=50),
        function=to_tensor)
    # Instantiate engine
    junkie = training.IgniteJunction({'model': model, 'dataset': batcher}, loss=loss, optimizer='Adam')
    # Train and test that changes took place
    assert (model.m == 1.).all()
    assert (model.b == 0.).all()
    junkie.train()
    assert not (model.m == 1.).all()
    assert not (model.b == 0.).all()
