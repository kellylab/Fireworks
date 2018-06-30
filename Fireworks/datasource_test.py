import Fireworks
import os
import pandas as pd
from Fireworks import datasource as ds
from Fireworks.message import Message

test_dir = Fireworks.test_dir

class one_way_dummy(ds.DataSource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def __next__(self):
        self.count += 1
        if self.count < 20:
            return self.count
        else:
            raise # This will trigger StopIteration

    def reset(self):
        self.count = 0

def conforms_to_spec(datasource):

    assert hasattr(datasource, '__iter__')
    assert hasattr(datasource, 'to_tensor')
    assert hasattr(datasource, '__next__')

    return True

def test_DataSource(): pass

def test_BioSeqSource():

    test_file = os.path.join(test_dir, 'sample_genes.fa')
    genes = ds.BioSeqSource(test_file)
    assert conforms_to_spec(genes)
    f = lambda batch: [1 for _ in batch]
    embedding_function = {'sequences': f}

    for gene in genes:
        assert type(gene) is pd.DataFrame
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.keys())
        message = genes.to_tensor(gene, embedding_function)
        assert type(message) is Message
        assert set(message.tensor_message.keys()) == set(['sequences'])
        assert set(message.df.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'rawsequences'])
        assert len(message) == 1
    # Reset and do it again to confirm we can repeat the source
    genes.reset()
    for gene in genes:
        assert type(gene) is pd.DataFrame
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.keys())
        message = genes.to_tensor(gene, embedding_function)
        assert type(message) is Message
        assert set(message.tensor_message.keys()) == set(['sequences'])
        assert set(message.df.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'rawsequences'])
        assert len(message) == 1

def test_LoopingSource():

    dumbo = one_way_dummy()
    loopy = ds.LoopingSource(inputs = {'dumbo': dumbo})

def test_CachingSource(): pass 
