import numpy as np
from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.corpus_reader import CGEDReader, str2id, padding, load_word_dict, save_word_dict

def data_generator(input_texts, target_texts, char2id, batch_size, maxlen=400):
    # 数据生成器
    while True:
        X, Y = [], []
        for i in range(len(input_texts)):
            X.append(str2id(input_texts[i], char2id, maxlen))
            # print("X", X)
            Y.append(str2id(target_texts[i], char2id, maxlen))
            # print("Y", Y)
            if len(X) == batch_size:
                X = np.array(padding(X, char2id))
                Y = np.array(padding(Y, char2id))
                # yield [X, Y], None
                X, Y = [], []

if __name__ == '__main__':
    train_path = config.test_path
    data_reader = CGEDReader(train_path)
    input_texts, target_texts = data_reader.build_dataset(train_path) # 原始文本，修改后文本
    print("input_text:", input_texts)
    save_vocab_path = config.save_vocab_path
    char2id = load_word_dict(save_vocab_path)
    batch_size = config.batch_size
    data_generator(input_texts, target_texts, char2id, batch_size)