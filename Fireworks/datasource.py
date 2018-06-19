from abc import abstractmethod
from Bio import SeqIO
import pandas as pd
from Fireworks.message import Message

class DataSource:
    """ Class for representing a data source. It formats and reads data, and is able to convert batches into tensors. """

    def __init__(self, inputs):
        self.inputs = inputs

    # @abstractmethod
    # def to_tensor(self, batch: dict, embedding_function: dict):
    #     """
    #     Converts a batch (stored as dictionary) to a dictionary of tensors. embedding_function is a dict that specifies optional
    #     functions that construct embeddings and are called on the element of the given key.
    #     """
    #     pass

    def __next__(self):
        return {key: next(_input) for key, _input in self.inputs.values()}

    def __getitem__(self, index):
        return {key: _input.__getitem__(index) for key, _input in self.inputs.values()}

    def __iter__(self):
        return self

class BioSeqSource(DataSource):
    """ Class for representing biosequence data. """

    def __init__(self, path, filetype = 'fasta', **kwargs):
        self.seq = SeqIO.parse(path, filetype, **kwargs)

    def to_tensor(self, batch: dict, embedding_function: dict):

        metadata = {
        'rawsequences': batch['sequences'],
        'names': batch['names'],
        'ids': batch['ids'],
        'descriptions': batch['descriptions'],
        'dbxrefs': batch['dbxrefs'],
            }

        tensor_dict = {
        'sequences': embedding_function['sequences'](batch['sequences']),
        }

        return Message(tensor_dict), pd.DataFrame(metadata)

    def __next__(self):

        gene = self.seq.__next__()

        return pd.DataFrame({
            'sequences': [str(gene.seq)],
            'ids': [gene.id],
            'names': [gene.name],
            'descriptions': [gene.description],
            'dbxrefs': [gene.dbxrefs],
        })
