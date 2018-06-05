import MLKit
import os
from MLKit import datasource as ds

test_dir = MLKit.test_dir

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
    for gene in genes:
        assert type(gene) is dict
        assert set(['sequences', 'ids', 'names', 'description', 'dbxrefs']) == set(gene.keys())

    f = lambda batch: [1 for _ in batch]
    embedding_function = {'sequences': f}
    batch = 
