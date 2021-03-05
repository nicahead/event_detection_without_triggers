# -*-coding:utf8-*-
from numpy import array


def load_dict(path):
    '''
    word_dict
    label_dict
    entype_dict
    ret: {word1:id1, word2:id2,...}
    '''
    fin = open(path)
    ret = {}
    for idx, line in enumerate(fin):
        if not line.strip(): continue
        ret[line.strip()] = (idx + 1)  # id from 1, 0 is for empty word/label/entype
    return ret


def load_embedding(path):
    '''
    load word embedding or
    position embedsing or
    label embedding
    '''
    fin = open(path)
    data = []
    for line in fin:
        if not line.strip(): continue
        data.append(list(map(float, line.strip().split(' '))))
    return array(data, dtype='float32')
    # return array(data)


def get_max_len(data_path):
    max_len = -1
    for line in open(data_path):
        line = line.strip()
        if not line: continue
        if len(line.split('\t')) < 3: continue
        wds = line.split('\t')[:-1]
        wds, ents = zip(*[x.split(' ') for x in wds])
        if len(wds) > max_len:
            max_len = len(wds)
    return max_len
