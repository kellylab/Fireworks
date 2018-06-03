import torch
from torch.utils.data import Dataset, DataLoader

class StreamingDataset(Dataset)
    """ Dataset with methods for chunking large datasets that cannot fit in memory. """

    def __init__(self): pass

    def preload(self): pass

    def get_chunk(self): pass

    def to_tensor(self, batch): pass

    def __len__(self): pass

    def __getitem__(self, index): pass

class StreamingLoader(Loader):

    def __init__(self, datasets): pass

    def get_generator(self): pass
