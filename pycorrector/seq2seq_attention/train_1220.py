# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.corpus_reader import CGEDReader, str2id, padding, load_word_dict, save_word_dict
from pycorrector.seq2seq_attention.evaluate import Evaluate
from pycorrector.seq2seq_attention.seq2seq_attn_multiembedding import Seq2seqAttn_multiembedding
from pypinyin import pinyin, lazy_pinyin, Style

def data_generator(input_texts, target_texts, char2id, input_pinyins, output_pinyins, pingyin2id, batch_size, maxlen=400):
    # 数据生成器
    while True:
        X, Y = [], []
        X_p, Y_p = [], []
        for i in range(len(input_texts)):
            X.append(str2id(input_texts[i], char2id, maxlen))
            Y.append(str2id(target_texts[i], char2id, maxlen))
            X_p.append(str2id(input_pinyins[i], pingyin2id, maxlen))
            Y_p.append(str2id(output_pinyins[i], pingyin2id, maxlen))
            if len(X) == batch_size:
                X = np.array(padding(X, char2id))
                Y = np.array(padding(Y, char2id))
                X_p = np.array(padding(X_p, pingyin2id))
                Y_p = np.array(padding(Y_p, pingyin2id))
                yield [X, Y, X_p, Y_p], None
                X, Y, X_p, Y_p = [], [], [], []
        # for i in range(len(input_pinyins)):
        #     X_p.append(str2id(input_pinyins[i], pingyin2id, maxlen))
        #     Y_p.append(str2id(output_pinyins[i], pingyin2id, maxlen))
        #     if len(X) == batch_size:
        #         X_p = np.array(padding(X_p, pingyin2id))
        #         Y_p = np.array(padding(Y_p, pingyin2id))
        #         yield [X_p, Y_p], None
        #         X_p, Y_p = [], []

def get_validation_data(input_texts, target_texts, char2id, input_pinyins, output_pinyins, pingyin2id, maxlen=400):
    # 数据生成器
    X, Y = [], []
    X_p, Y_p = [], []
    for i in range(len(input_texts)):
        X.append(str2id(input_texts[i], char2id, maxlen))
        Y.append(str2id(target_texts[i], char2id, maxlen))
        X_p.append(str2id(input_pinyins[i], pingyin2id, maxlen))
        Y_p.append(str2id(output_pinyins[i], pingyin2id, maxlen))
        X = np.array(padding(X, char2id))
        Y = np.array(padding(Y, char2id))
        X_p = np.array(padding(X_p, pingyin2id))
        Y_p = np.array(padding(Y_p, pingyin2id))
        return [X, Y, X_p, Y_p], None

def to_pinyin(str_list):
    pinyin_list = []
    for c_list in str_list:
        sub_pinyin = []
        for c in c_list:
            # print(lazy_pinyin(c, style=Style.TONE8))
            str_pinyin = str(lazy_pinyin(c, style=Style.TONE2, errors='replace'))[2:-2]
            sub_pinyin.append(str_pinyin)
        pinyin_list.append(sub_pinyin)
    # print(pinyin_list)
    return pinyin_list

def train(train_path='', test_path='', save_vocab_path='', save_pinyin_path='', attn_model_path='',
          batch_size=64, epochs=100, maxlen=400, hidden_dim=128, dropout=0.2, use_gpu=False):
    data_reader = CGEDReader(train_path)
    input_texts, target_texts = data_reader.build_dataset(train_path) # 原始文本，修改后文本
    # todo 获取相应拼音文本
    input_pinyins = to_pinyin(input_texts)
    output_pinyins = to_pinyin(target_texts)

    test_input_texts, test_target_texts = data_reader.build_dataset(test_path)
    test_input_pinyins = to_pinyin(test_input_texts)
    test_output_pinyins = to_pinyin(test_target_texts)

    # load or save word dict
    if os.path.exists(save_vocab_path):
        char2id = load_word_dict(save_vocab_path)
        id2char = {int(j): i for i, j in char2id.items()} # {0: '"', 1: '%', 2: '(', 3: ')', 4: '+'}
        chars = set([i for i in char2id.keys()]) # {'+', '%', ')', '(', '"'}
    else:
        print('Training data...')
        print('input_texts:', input_texts[0])
        print('target_texts:', target_texts[0])
        max_input_texts_len = max([len(text) for text in input_texts])

        print('num of samples:', len(input_texts))
        print('max sequence length for inputs:', max_input_texts_len)
        # chars:保存所有字符，存放到列表中
        chars = data_reader.read_vocab(input_texts + target_texts)
        id2char = {i: j for i, j in enumerate(chars)}
        char2id = {j: i for i, j in id2char.items()}
        save_word_dict(char2id, save_vocab_path)
    print("chars:", chars)

    # load pinyin vocab dict
    if os.path.exists(save_pinyin_path):
        pingyin2id = load_word_dict(save_pinyin_path)
        id2pinyin = {int(j): i for i, j in pingyin2id.items()}
        pinyins = set([i for i in pingyin2id.keys()])
    else:
        pinyins = data_reader.read_vocab(input_pinyins + output_pinyins)
        id2pinyin = {i: j for i, j in enumerate(pinyins)}
        pingyin2id = {j: i for i, j in id2pinyin.items()}
        save_word_dict(pingyin2id, save_pinyin_path)
    print("pinyins:", pinyins)

    model = Seq2seqAttn_multiembedding(chars, pinyins,
                             attn_model_path=attn_model_path,
                             hidden_dim=hidden_dim,
                             use_gpu=use_gpu,
                             dropout=dropout).build_model()
    evaluator = Evaluate(model, attn_model_path, char2id, id2char, maxlen)
    model.fit_generator(data_generator(input_texts, target_texts, char2id, input_pinyins, output_pinyins, pingyin2id,batch_size, maxlen),
                        steps_per_epoch=(len(input_texts) + batch_size - 1) // batch_size,
                        epochs=epochs,
                        validation_data=get_validation_data(test_input_texts, test_target_texts, char2id,test_input_pinyins, test_output_pinyins, pingyin2id, maxlen),
                        callbacks=[evaluator])

if __name__ == "__main__":
    # train(train_path=config.train_sighan_path,
    #       test_path=config.test_sighan_path,
    #       save_vocab_path=config.save_taiwan_vocab_path,
    #       save_pinyin_path=config.save_pinyin_path,
    #       attn_model_path=config.attn_model_path,
    #       batch_size=config.batch_size,
    #       epochs=config.epochs,
    #       maxlen=config.maxlen,
    #       hidden_dim=config.rnn_hidden_dim,
    #       dropout=config.dropout,
    #       use_gpu=config.use_gpu)

    # 本地调试，文件换成较小文件
    train(train_path=config.test_sighan_path,
          test_path=config.test_sighan_path,
          save_vocab_path=config.save_taiwan_vocab_path,
          save_pinyin_path=config.save_pinyin_path,
          attn_model_path=config.attn_pinyin_path,
          batch_size=config.batch_size,
          epochs=config.epochs,
          maxlen=config.maxlen,
          hidden_dim=config.rnn_hidden_dim,
          dropout=config.dropout,
          use_gpu=config.use_gpu)

