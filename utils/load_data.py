# -*-coding:utf8-*-
import sys
import os
import math
import collections

import utils.tools

from numpy import array


def _sent2array_ent(sent, wdict, edict, ydict, max_len):
    """
    将sent(wds, ents, ls)根据字典进行编码
    """
    words = list(map(lambda x: wdict.get(x, wdict['<UNK>']), sent[0]))
    ents = list(map(lambda x: edict.get(x, edict['NEGATIVE']), sent[1]))
    MAX_SEN_LEN = max_len
    if len(words) < MAX_SEN_LEN:
        words += ([wdict['<PAD>']] * (MAX_SEN_LEN - len(words)))  # 长度小于MAX_SEN_LEN的句子用-1填充字符
        ents += ([edict['NEGATIVE']] * (MAX_SEN_LEN - len(ents)))  # 长度小于MAX_SEN_LEN的句子用'NEGATIVE'填充实体标记
    elif len(words) > MAX_SEN_LEN:
        words = words[:MAX_SEN_LEN]  # 长度大于MAX_SEN_LEN的句子直接截取
        ents = ents[:MAX_SEN_LEN]

    labels = [ydict.get(x.lower(), 'negative') for x in sent[2]]

    return words, ents, labels


def load_data_ent(data_path, wdict, edict, ydict, max_len):
    """
    读取数据文件，返回三维列表 [[单词序列列表],[实体序列列表],[事件类型列表]]
    """
    sen, ent, y = [], [], []  # 都是二维数组

    for line in open(data_path, encoding='utf8'):
        # line: Rodong ORG_Media\tcalled NEGATIVE
        line = line.strip()
        if not line: continue
        if len(line.split('\t')) < 3: continue

        wds = line.split('\t')[:-1]  # ['Rodong ORG_Media','...'] 最后一个是label
        wds, ents = zip(*[x.split(' ') for x in wds])  # wds:['Rodong','..'] ents:['ORG_Media','...']
        ls = line.split('\t')[-1].strip().lower().split(' ')  # label: ['meet']
        words, ents, labels = _sent2array_ent((wds, ents, ls), wdict, edict, ydict, max_len)  # 根据字典进行编码
        sen.append(words)
        ent.append(ents)
        y.append(labels)

    return [array(sen, dtype='int32'), array(ent, dtype='int32'), y]
