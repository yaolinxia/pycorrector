# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
import os
import random

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.corpus_reader import load_word_dict
from pycorrector.seq2seq_attention.evaluate import gen_target
from pycorrector.seq2seq_attention.seq2seq_attn_multiembedding import Seq2seqAttn_multiembedding


class Inference(object):
    def __init__(self, save_vocab_path='', attn_model_path='', maxlen=400):
        if os.path.exists(save_vocab_path):
            self.char2id = load_word_dict(save_vocab_path)
            self.id2char = {int(j): i for i, j in self.char2id.items()}
            self.chars = set([i for i in self.char2id.keys()])
        else:
            print('not exist vocab path')
        seq2seq_attn_model = Seq2seqAttn_multiembedding(self.chars, attn_model_path=attn_model_path)
        self.model = seq2seq_attn_model.build_model()
        self.maxlen = maxlen

    def infer(self, sentence):
        return gen_target(sentence, self.model, self.char2id, self.id2char, self.maxlen, topk=3)

def input2list(test_path=config.test_sighan_path):
    test_list = []
    with open(test_path, 'r', encoding='utf-8') as f:
        sen_list= f.readlines()
        for i in range(0, len(sen_list), 2):
            # print(sen_list[i][4:-1])
            test_list.append(sen_list[i][4:-1])
        print(test_list)
        return test_list

def input2dict(test_path=config.test_sighan_path):
    test_dict = {}
    with open(test_path, 'r', encoding='utf-8') as f:
        sen_list= f.readlines()
        for i in range(0, len(sen_list), 2):
            test_dict[sen_list[i][4:-1]] = sen_list[i+1][4:-1]
        print(test_dict)
        return test_dict

def infer(test_list, p, out_path):
    test_len = len(test_list)
    test_rand = random.sample(test_list, test_len//p)
    inference = Inference(save_vocab_path=config.save_pinyin_path,
                          attn_model_path=config.attn_pinyin_path,
                          maxlen=400)
    total_nums = 0
    correct_nums = 0
    src2target = input2dict()
    with open(out_path, 'a', encoding='utf-8') as f:
        for i in test_rand:
            infer_target = inference.infer(i)
            print("input: " + i)
            print("infer: " + infer_target)
            print("traget:" + src2target[i])
            total_nums += len(i)
            correct_nums += minDistance(infer_target, src2target[i])
            f.write("input: " + i + '\n')
            f.write("infer: " + infer_target + '\n')
            f.write("target:" + src2target[i] + '\n')
            # if infer_target == src2target[i]:
            #     correct_nums += 1

    pred = correct_nums / total_nums
    print("total_nums:", total_nums)
    print("correct_nums:", correct_nums)
    print("prediction:", pred)

# 最小编辑距离
def minDistance(word1, word2):
    m = len(word1)
    n = len(word2)
    D = [[0] * (n+1) for _ in range(m+1)]
    D[0] = [i for i in range(n+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            tmp1 = min(D[i-1][j], D[i][j-1]) + 1
            tmp2 = D[i-1][j-1] + (0 if word1[i-1] == word2[j-1] else 1)
            D[i][j] = min(tmp1, tmp2)
    return D[m][n]


def main():
    inputs = [
        '由我起开始做。',
        '没有解决这个问题，',
        '由我起开始做。',
        '由我起开始做',
        '不能人类实现更美好的将来。',
        '这几年前时间，',
        '歌曲使人的感到快乐，',
        '会能够大幅减少互相抱怨的情况。'
    ]
    inference = Inference(save_vocab_path=config.save_pinyin_path,
                          attn_model_path=config.attn_pinyin_path,
                          maxlen=400)
    for i in inputs:
        target = inference.infer(i)
        print('input:' + i)
        print('output:' + target)
    while True:
        sent = input('input:')
        print("output:" + inference.infer(sent))

# result:
# input:由我起开始做。
# output:我开始做。
# input:没有解决这个问题，
# output:没有解决的问题，
# input:由我起开始做。

if __name__ == "__main__":
    test_list = input2list()
    out_path = config.result_pinyin_path
    infer(test_list, 2, out_path)
    # input2dict()
