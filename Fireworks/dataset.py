# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# from Fireworks.datasource import Source
#
# class FireworksDataset(Dataset):
#     """
#     Class for reading data from a datasource. Whereas a data source handles formatting
#     and tensorization of data, a dataset decides how to read that data (whether to stream it in chunks
#     or load it all at once), how to store it in memory, and how to apply transformations to it as needed
#     """
#
#     def __init__(self, sources, cache_size=None, chunk_size=None):
#
#         self.sources = sources
#         self.cache_size = cach_size
#         if chunk_size is None:
#             chunk_size = round(cache_size / 2)
#         self.chunk_size = chunk_size
#         self.length = None
#
#         self.check_sources()
#
#     def get_chunk(self, chunk_size = None):
#         """ Gets the next chunk from the dataset. """
#         if chunk_size is None:
#             chunk_size = self.chunk_size
#
#         return {key: get_next_n(source, chunk_size, global_index) for key, source in self.sources.items()}
#
#     def to_tensor(self, batch):
#         """ Converts a batch from this dataset into a tensor. """
#
#         tensors_and_meta = [source._to_tensor(batch) for source in self.sources.values()]
#         tensors = [x[0] for x in tensors_and_meta]
#         tensors = {key: value for tensor in tensors for key, value in tensor.items()}
#         meta = [x[1] for x in tensors_and_meta]
#         meta = {key: value for met in met for key, value in met.items()}
#
#         return tensors, meta
#
#     def __len__(self):
#         if self.length:
#             return self.length
#         else:
#             raise Error("This dataset does not have a length or is not aware of its length. This can be" + \
#             "because the entire dataset is not stored in memory and is dynamically streamed in at once.")
#
#     def __getitem__(self, index):
#
#         if self.length is None:
#             raise AttributeError("This dataset does not have a length or is not aware of its length." + \
#             "Hence, it's elements cannot be accessed by index.")
#
#         # Check if requested indices are in cache
#         index = set(index_to_list(index))
#         # Get items
#
#     def __iter__(self): return self
#
#     def __next__(self):
#         if self.cache_index < len(self.tensor_cache):
#             return self.tensor_cache[self.cache_index]
#             self.cache_index += 1
#             self.global_index += 1
#             # TODO: Add asynchronous chunk download triggers
#         else:
#             # Update cache with new chunk
#             self.update_cache()
#             # Delete the first s-c elements of cache and get a chunk of c elements
#             # Convert new chunk to_tensor
#             # Combine chunks
#
#     def download_chunk(self):
#         """ Updates cache by downloading a new chunk. """
#
#     def update_cache(self):
#         """ Updates cache by downloading a new chunk. """
#         offset = self.cache_size - self.chunk_size
#         new_cache_index = self.cache_index - offset
#         if new_cache_index < 0: # Deleting the first s-c elements would delete the current position in the dataset
#             offset = self.cache_index
#         if offset > 0:
#             # Delete the first offset elements of cache
#             self.tensor_cache = self.tensor_cache[offset:]
#             self.meta_cache = self.meta_cache.iloc[offset:]
#             # Download new chunk
#             download_size = self.cache_size - offset # Size of chunk to download
#             new_chunk = self.get_chunk(download_size)
#             tensor_chunk, meta_chunk = self.to_tensor(new_chunk)
#             self.tensor_cache = self.tensor_cache.append(tensor_chunk)
#             self.meta_cache = self.meta_cache.append(meta_chunk)
#
#     def check_sources(self):
#         """ Check sources to make sure there are no overlapping keys. """
#         pass
#
# def get_next_n(source, chunk_size, global_index = None):
#     """ Returns the next n elements from source. """
#     # If the source supports random access, get the entire chunk at once
#     if global_index and hasattr(source, '__getitem__'):
#         return source.iloc[global_index:global_index+chunk_size]
#     # Otherwise iterate through the source as needed
#     else:
#         chunk = []
#         for _ in range(chunk_size):
#             chunk.append(next(soure))
#         return pd.DataFrame(chunk)
#
# class StreamingDataset(FireworksDataset):
#     """ Dataset for streaming data in online manner. """
#
#     def preload(self): pass
#
#     def get_chunk(self): pass
#
#     def to_tensor(self, batch): pass
#
#     def __len__(self): pass
#
#     def __getitem__(self, index): pass
#
# class ChunkingDataset(FireworksDataset):
#     """ Dataset with methods for chunking large datasets that cannot fit in memory. """
#     pass
#
# class WholeDataset(FireworksDataset):
#     """ Dataset that loads all of its data into memory at once. """
#
#     def __init__(self, source):
#
#         # Establish length
#         self.length = self.measure_length(source)
#         # Tensorize
#         self.data, self.metadata = self.tensorize(source)
#
#     def __getitem__(self, index):
#         """ This should return a dictionary of tensors corresponding to the index. """
#
#     def apply_transform(self, transform):
#         """ Applies a (list of) transform to dataset. """
#
# class Transform:
#     """ Class for applying transforms to a dataset. """
#
#     def __init__(self): pass
#
#     def before_iterate(self, dataset: Dataset): pass
#
#     def during_iteration(self, dataset: Dataset, batch): pass
#
#     def after_iteration(self, dataset: Dataset): pass
#
# class ComputeLengthTransform:
#     """ Computes the length of a dataset. """
#
#     def before_iterate(self, dataset: Dataset):
#         dataset.length = 0
#
#     def during_iteration(self, dataset: Dataset, batch):
#         dataset.length += len(batch)
#
#
# class CachingSource(DataSource):
#
#     def __init__(self, inputs, cache_size, chunk_size):
#         super.__init__(inputs)
#         self.cache_size = cache_size
#         self.chunk_size = chunk_size
#         self.cache = {}
#
#     def to_tensor(self, batch: dict, embedding_function: dict): pass
#
#     def check_cache(self, index): pass
#
#     def free_cache(self, n): pass
#
#     def _free_cache(self, n): pass
#
#     def insert_cache(self, message): pass
#
#     def read_cache(self, index): pass
#
# def index_to_list(index):
#     """
#     Converts an index to a list.
#     """
#     if type(index) is slice:
#         index = slice_to_list(index)
#     if type(index) is int:
#         index = [index]
#     return index
