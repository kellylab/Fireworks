from fireworks.toolbox import text
import random

def test_space_tokenizer():
    phrase = "Hello is this dog"
    tokens = text.space_tokenizer(phrase)
    assert (tokens == ['Hello', 'is', 'this', 'dog'])

def test_character_tokenizer():

    sequence = 'hello'
    split = text.character_tokenizer(sequence)
    assert split == ['h', 'e', 'l', 'l', 'o']

test_embedding_dict = {str(i):i for i in range(80)}
test_embedding_dict['EOS'] = 81
test_embedding_dict['SOS'] = 82
test_embedding_dict['UNK'] = 83

def test_pad_sequence():

    width = 100
    test_sequence = lambda : [random.randint(0,90) for _ in range(random.randint(1,width))]
    sequence = test_sequence()
    n = len(sequence)
    pad = 150
    padded = text.pad_sequence(sequence, pad, embeddings_dict = test_embedding_dict)
    assert (padded[0:n] == sequence).all()
    for p in padded[n:pad]:
        assert p == test_embedding_dict['EOS']
    # Test with a different pad_token
    padded = text.pad_sequence(sequence, pad, embeddings_dict = test_embedding_dict, pad_token='2')
    assert (padded[0:n] == sequence).all()
    for p in padded[n:pad]:
        assert p == test_embedding_dict['2']

    # count = 10
    # sequences = [test_sequence() for i in range(count)]
    # padded = [text.pad_sequence(sequence, n, embeddings_dict = test_embedding_dict) for sequence in sequences]
    # assert len(padded) == count
    # assert padded.shape == (count, width)

def test_pad():

    lengths = [random.randint(10, 15) for _ in range(10)]
    test_sequence = lambda n: [1 for _ in range(n)]
    sequences = [test_sequence(i) for i in lengths]
    padded = text.pad(sequences, test_embedding_dict)
    max_length = max(lengths)
    for original, new in zip(sequences, padded):
        assert len(new) == max_length
        a = len(original)
        b = max_length
        for token in new[a:b]:
            assert token == test_embedding_dict['EOS']

def test_apply_embeddings():

    phrase = '1234x'
    embeddings = text.apply_embeddings(phrase, embeddings_dict=test_embedding_dict, tokenizer = text.character_tokenizer)
    assert len(embeddings) == 7
    assert embeddings[0] == test_embedding_dict['SOS']
    assert embeddings[1] == test_embedding_dict['1']
    assert embeddings[2] == test_embedding_dict['2']
    assert embeddings[3] == test_embedding_dict['3']
    assert embeddings[4] == test_embedding_dict['4']
    assert embeddings[5] == test_embedding_dict['UNK']
    assert embeddings[6] == test_embedding_dict['EOS']


# def test_create_pretrained_embeddings():
#     embeddings = nlp.create_pretrained_embeddings(os.path.join(fireworks.EXTERNAL_DIR, 'glove.test.txt'))
#     assert len(embeddings.keys()) == 18
#     for key, item in embeddings.items():
#         assert len(item) == 300

def test_too_big() : pass
