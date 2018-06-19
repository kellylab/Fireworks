import Fireworks
import os
import pandas as pd
from Fireworks import datasource as ds
from Fireworks.message import Message

test_dir = Fireworks.test_dir

def conforms_to_spec(datasource):

    assert hasattr(datasource, '__iter__')
    assert hasattr(datasource, 'to_tensor')
    assert hasattr(datasource, '__next__')

    return True

def test_DataSource():

    bob = ds.DataSource()
    assert conforms_to_spec(bob)

def test_BioSeqSource():

    test_file = os.path.join(test_dir, 'sample_genes.fa')
    genes = ds.BioSeqSource(test_file)
    assert conforms_to_spec(genes)
    f = lambda batch: [1 for _ in batch]
    embedding_function = {'sequences': f}

    for gene in genes:
        assert type(gene) is pd.DataFrame
        assert set(['sequences', 'ids', 'names', 'descriptions', 'dbxrefs']) == set(gene.keys())
        message, metadata = genes.to_tensor(gene, embedding_function)
        assert type(message) is Message
        assert type(metadata) is pd.DataFrame
        assert set(message.keys()) == set(['sequences'])
        assert set(metadata.keys()) == set(['ids', 'names', 'descriptions', 'dbxrefs', 'rawsequences'])
        assert len(message) == 1
        assert len(metadata) == 1
