import numpy as np

"""
This file contains helper functions that can be useful for NLP tasks.
"""

def character_tokenizer(sequence):
    """ Splits sequence into a list of characters. """

    if sequence:
        return [*sequence] # This unpacks an iterable into a list of elements

def space_tokenizer(sequence):
    """ Splits sequence based on spaces. """
    if sequence:
        return sequence.split(' ')

def pad_sequence(sequence, max_length, embeddings_dict, pad_token = 'EOS'):
    """ Adds EOS tokens until sequence length is max_length. """

    n = len(sequence)
    padded = sequence.copy()

    if n < max_length:
        if type(padded) is np.ndarray:
            padded = np.append(padded, [embeddings_dict[pad_token] for _ in range(max_length-n)], axis=0)
        else:
            padded.extend([embeddings_dict[pad_token] for _ in range(max_length-n)])

    return np.array(padded)

def pad(batch, embeddings_dict, pad_token = 'EOS'): # TODO: Don't rely on embeddings dict for EOS tokens
    """ Pads all embeddings in a batch to be the same length. """

    lengths = [len(x) for x in batch]
    max_length = max(lengths)
    padded_embeddings = np.stack([pad_sequence(seq, max_length, embeddings_dict, pad_token) for seq in batch])
    batch = padded_embeddings

    return batch

def apply_embeddings(sequence, embeddings_dict, tokenizer):
    """ Decomposes sequence into tokens using tokenizer and then converts tokens to embeddings using embeddings_dict. """

    if sequence:
        embeddings = [embeddings_dict['SOS']]
        embeddings.extend([embeddings_dict[token] if token in embeddings_dict else embeddings_dict['UNK']
                        for token in tokenizer(sequence)])
        embeddings.append(embeddings_dict['EOS'])
        # if vectorizor:
        #     embeddings = [vectorizor(embedding) for embedding in embeddings]
        return embeddings

    else: # In case input sequence is empty
        return [embeddings_dict['SOS'], embeddings_dict['EOS']]

def create_pretrained_embeddings(embeddings_file): # TODO: Write test
    """ Loads embeddings vectors from file into a dict. """
    embeddings_dict = {}
    with open(embeddings_file) as vectors:
        for line in vectors:
            splits = line.split(' ')
            embeddings_dict[splits[0]] = np.array([float(x) for x in splits[1:]])

    return embeddings_dict

def load_embeddings(name = 'glove840b'): # TODO: write test
    """ Loads serialized embeddings from pickle. """
    with open(os.path.join(silph.RAW_DIR, name), 'rb') as pretrained:
        embeddings_dict = pickle.load(pretrained)

    return embeddings_dict

def make_vocabulary(text, tokenizer = None, cutoff_rule = None): #TODO: write test
    """ Converts an iterable of phrases into the set of unique tokens that are in the vocabulary. """
    counts = defaultdict(lambda:0)
    if tokenizer is None:
        tokenizer = space_tokenizer

    #vocabulary = set()
    for phrase in text:
        if phrase is not None:
            tokens = tokenizer(phrase)
            #vocabulary.add(tokens)
            for token in tokens:
                counts[token] += 1
        else:
            pass

    if cutoff_rule:
        counts = cutoff_rule(counts)

    return dict(counts) # The keys of this dict are the unique tokens.

def make_indices(vocabulary): # TODO: write test
    """
    Constructs a dictionary of token names to indices from a vocabulary.
    Each index value corresponds to a one-hot vector.
    """
    embeddings_dict = bidict({token: i for token, i in zip(vocabulary, count(start=3))})
    embeddings_dict['SOS'] = special_tokens['SOS']
    embeddings_dict['EOS'] = special_tokens['EOS']
    embeddings_dict['UNK'] = special_tokens['UNK']
    return embeddings_dict

def too_big(dataset, start, end, dim = 300, cutoff = 620000):
    """
    Calculates if a batch consisting of dataset[start:end] is too big based on cutoff. This can be used for constructing dynamic batches.
    """

    sizes = [len(x) for x in dataset[start:end]['embeddings']]
    max_size = max(sizes)
    dim = dim
    batch_size = end-start
    size = dim*max_size*batch_size
    return size > cutoff
