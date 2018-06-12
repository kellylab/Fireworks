import torch
from torch.utils.data import Dataset

class MLKitDataset(Dataset):
    """
    Class for reading data from a datasource. Whereas a data source handles formatting
    and tensorization of data, a dataset decides how to read that data (whether to stream it in chunks
    or load it all at once), how to store it in memory, and how to apply transformations to it as needed
    """

    def __init__(self, sources):

        self.sources = sources

class StreamingDataset(MLKitDataset)
    """ Dataset for streaming data in online manner. """

    def preload(self): pass

    def get_chunk(self): pass

    def to_tensor(self, batch): pass

    def __len__(self): pass

    def __getitem__(self, index): pass

class ChunkingDataset(MLKitDataset): pass
    """ Dataset with methods for chunking large datasets that cannot fit in memory. """

class WholeDataset(MLKitDataset):
    """ Dataset that loads all of its data into memory at once. """

    def __init__(self, source):

        # Establish length
        self.length = self.measure_length(source)
        # Tensorize
        self.data, self.metadata = self.tensorize(source)

    def __getitem__(self, index):
        """ This should return a dictionary of tensors corresponding to the index. """

    def apply_transform(self, transform):
        """ Applies a (list of) transform to dataset. """

class Transform:
    """ Class for applying transforms to a dataset. """

    def __init__(self): pass

    def before_iterate(self, dataset: Dataset): pass

    def during_iteration(self, dataset: Dataset, batch): pass

    def after_iteration(self, dataset: Dataset): pass

class ComputeLengthTransform:
    """ Computes the length of a dataset. """

    def before_iterate(self, dataset: Dataset):
        dataset.length = 0

    def during_iteration(self, dataset: Dataset, batch):
        dataset.length += len(batch)


class CachingSource(DataSource):

    def __init__(self, inputs, cache_size, chunk_size):
        super.__init__(inputs)
        self.cache_size = cache_size
        self.chunk_size = chunk_size
        self.cache = {}
        
    def to_tensor(self, batch: dict, embedding_function: dict): pass

    def check_cache(self, index): pass

    def free_cache(self, n): pass

    def _free_cache(self, n): pass

    def insert_cache(self, message): pass

    def read_cache(self, index): pass
