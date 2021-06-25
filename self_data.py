import numpy as np
import os

import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform



#read id

def readid(path):
    path = os.path.join(path)
    with open(path, encoding='utf-8') as file:
        each = file.readlines()
        word2id = {}
        tuip = []
        for i, j in enumerate(each):
            j = j.strip()
            t = (j, i)
            tuip.append(t)

        word2id = dict(list(tuip))
    return word2id




def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat



def load_data(vocabs,path1,path2):
    train = _parse_data(open(path1, 'rb'))
    test = _parse_data(open(path2, 'rb'))

    # word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(vocabs.items())]
    chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    # if platform.system() == 'Windows':
    #     split_text = '\r\n'
    # else:
    #     split_text = '\n'
    #
    # string = fh.read().decode('utf-8')
    # data = [[row.split() for row in sample.split(split_text)] for
    #         sample in
    #         string.strip().split(split_text + split_text)]
    # data1 = [[[row for row in sample.split('\t')] for sample in string.split('\n')]]
    strings = fh.readlines()
    x = []
    data = []
    for i in range(len(strings)):
        if strings[i].decode() == '\n':
            data.append(x)
            x = []
            continue
        x.append(strings[i].decode().replace('\n', '').split('\t'))
    fh.close()



    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        # maxlen = 300
        maxlen = max(len(s) for s in data)

    word2idx = dict((w, i) for i, w in enumerate(vocab))

    #建立索引
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    # for i in data[0][0]:
    #     x = word2idx.get(i[0].lower(), 1)

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk