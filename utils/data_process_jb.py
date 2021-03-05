# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
import jieba.posseg as pseg


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_corpus(name):
    """
    将语料置为corpus_test_10.txt中的格式
    词 实体类型\t词 实体类型\t...事件类型
    """

    def merge_word_entity(words):
        """
        [百度, 是, 一家, 高科技, 公司], [ORG, v, m, n, n] -> '百度 ORG\t是 v\t ...'
        """
        line = ''
        word_list = []
        for word, flag in words:
            if is_number(word):
                word = '<NUM>'
            if word not in ['，', '。', '?', '!', '“', '：', '‘', '-', '（', '）', '《', '》', '；', '.',
                            '、',
                            '...',
                            ',', '？', '！']:
                line += word + ' ' + flag + '\t'
                word_list.append(word)
        return line, word_list

    words = []
    out = ''
    df = pd.read_csv('../data/DuEE/corpus_' + name + '.csv')
    for row in zip(df['text'], df['label']):
        # 句子分词、命名实体识别
        seg_result = pseg.cut(row[0].replace('\n', '').replace(' ', ','))
        line, word_list = merge_word_entity(seg_result)
        words.extend(word_list)
        out += line + row[1] + '\n'

    f = open('../data/corpus_' + name + '.txt', 'w', encoding='utf8')
    f.write(out)
    f.close()


def get_dict():
    """
    读取语料，得到原模型需要的dicts
    ent_dict.txt ： 实体类型
    word_dict.txt ： 词典
    label_dict.txt ： 事件类型
    一行是一条记录
    """
    data_paths = ['train', 'dev', 'test']
    words = []
    entities = []
    events = []
    for item in data_paths:
        for line in open(os.path.join('../data/', 'corpus_' + item + '.txt'), encoding='utf8'):
            # line: Rodong ORG_Media\tcalled NEGATIVE
            line = line.strip()
            if not line: continue
            wds = line.split('\t')[:-1]  # ['Rodong ORG_Media','...'] 最后一个是label
            try:
                wds, ents = zip(*[x.split(' ') for x in wds])  # wds:['Rodong','..'] ents:['ORG_Media','...']
            except:
                print(wds)
            ls = line.split('\t')[-1].strip().lower().split(' ')  # label: ['meet']
            words.extend(wds)
            entities.extend(ents)
            events.extend(ls)

    words = '\n'.join(list(set(words)))
    words += '\n<UNK>'
    words += '\n<PAD>'
    words += '\n<NUM>'
    entities = '\n'.join(list(set(entities)))
    entities += '\nNEGATIVE'
    events = '\n'.join(list(set(events)))
    events += '\nNEGATIVE'
    f1 = open('../data/dicts/word_dict.txt', 'w', encoding='utf8')
    f1.write(words)
    f1.close()
    f2 = open('../data/dicts/ent_dict.txt', 'w', encoding='utf8')
    f2.write(entities)
    f2.close()
    f3 = open('../data/dicts/label_dict.txt', 'w', encoding='utf8')
    f3.write(events)
    f3.close()


def get_word_embed():
    """
    读取大的预训练词典，根据本语料中的词和字生成缩小版词典
    :return:
    """

    def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
        """
        读取txt文件，得到词和向量
        :param path:
        :param topn:
        :return:
        """
        lines_num, dim = 0, 0
        vectors = {}
        iw = []
        wi = {}
        with open(path, encoding='utf-8', errors='ignore') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    dim = int(line.rstrip().split()[1])
                    continue
                lines_num += 1
                tokens = line.rstrip().split(' ')
                vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                iw.append(tokens[0])
                if topn != 0 and lines_num >= topn:
                    break
        for i, w in enumerate(iw):
            wi[w] = i
        return vectors, iw, wi, dim

    vectors, iw, wi, dim = read_vectors('/home/nic/projects/event_detection/data/merge_sgns_bigram_char300.txt',
                                        0)  # 所有字的词典
    wi['<PAD>'] = len(iw)
    wi['<UNK>'] = len(iw) + 1
    wi['<NUM>'] = len(iw) + 2
    vectors['<PAD>'] = np.zeros((300,))
    vectors['<UNK>'] = np.zeros((300,))
    vectors['<NUM>'] = np.zeros((300,))

    words = []
    for line in open(os.path.join('../data/dicts/word_dict.txt'), encoding='utf8'):
        words.append(line)
    # 取出token对应的vector matrix
    vec_matrix = []
    for word in words:
        word = word.replace('\n', '')
        vec = vectors.get(word, vectors['<UNK>'])
        vec_str = ' '.join(str(i) for i in vec)
        vec_matrix.append(vec_str)
    vec_matrix = '\n'.join(vec_matrix)
    f = open('../data/embeddings/300.txt', 'w', encoding='utf8')
    f.write(vec_matrix)
    f.close()


if __name__ == '__main__':
    get_corpus('train')
    get_corpus('dev')
    get_corpus('test')

    get_dict()

    get_word_embed()
