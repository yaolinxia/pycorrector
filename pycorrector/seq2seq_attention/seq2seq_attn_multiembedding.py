# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import os

from keras import backend as K
from keras.layers import Input, Lambda, Layer, Embedding, Bidirectional, Dense, Activation, GRU, CuDNNGRU, Concatenate
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """

    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class Interact(Layer):
    """交互层，负责融合encoder和decoder的信息
    """

    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(in_dim, out_dim),
                                      initializer='glorot_normal')

    def call(self, inputs):
        q, v, v_mask = inputs
        k = v
        mv = K.max(v - (1. - v_mask) * 1e10, axis=1, keepdims=True)  # maxpooling1d
        mv = mv + K.zeros_like(q[:, :, :1])  # 将mv重复至“q的timesteps”份
        # 下面几步只是实现了一个乘性attention
        qw = K.dot(q, self.kernel)
        a = K.batch_dot(qw, k, [2, 2]) / 10.
        a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
        a = K.softmax(a)
        o = K.batch_dot(a, v, [2, 1])
        # 将各步结果拼接
        return K.concatenate([o, q, mv], 2)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1],
                input_shape[0][2] + input_shape[1][2] * 2)


class Seq2seqAttn_multiembedding(object):
    def __init__(self, chars, pinyins, hidden_dim=128, attn_model_path=None, use_gpu=False, dropout=0.2):
        self.chars = chars
        self.pinyins = pinyins
        self.hidden_dim = hidden_dim
        self.model_path = attn_model_path
        self.use_gpu = use_gpu
        self.dropout = dropout

    def build_model(self):
        # 搭建seq2seq模型
        x_in = Input(shape=(None,))
        y_in = Input(shape=(None,))
        x_p_in = Input(shape=(None,))
        y_p_in = Input(shape=(None,))
        x = x_in
        y = y_in
        x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
        y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)
        print("x_mask:", x_mask)
        print("y_mask:", y_mask)

        x_one_hot = Lambda(self._one_hot)([x, x_mask])
        x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（target的字词很可能在input出现过）

        # char_embedding
        embedding = Embedding(len(self.chars), self.hidden_dim // 2 , mask_zero=True) # 其中一个是指input_dim, 另一个参数是指output_dim:词向量的纬度
        # print("embedding.shape", embedding.shape)
        x = embedding(x)
        y = embedding(y)
        print("x.shape", x.shape)
        # todo: self.pinyins表示；pinyin_vocab, vocabembedding_p()
        # pinyin_embedding
        embedding_p = Embedding(len(self.pinyins), self.hidden_dim // 2, mask_zero=True)
        x_p = embedding_p(x_p_in)
        y_p = embedding_p(y_p_in)

        # cat = Concatenate(axis=1)([embedding, embedding_p])

        # print("x_p.shape", x_p.shape)
        # x = K.concatenate([x, x_p],axis=-1)
        x = Concatenate(axis=-1)([x, x_p])
        #x = tf.concat([x, x_p], axis=-1)
        print("x_shape_concat", x.shape)

        # y = K.concatenate([y, y_p],axis=-1)
        y = Concatenate(axis=-1)([y, y_p])
        #$ y = tf.concat([y, y_p], axis=-1)

        # encoder，双层双向GRU; decoder，双层单向GRU
        if self.use_gpu:
            # encoder
            x = Bidirectional(CuDNNGRU(int(self.hidden_dim / 2), return_sequences=True))(x)
            x = Bidirectional(CuDNNGRU(int(self.hidden_dim / 2), return_sequences=True))(x)
            # decoder
            # todo: decoder层的y根据领域知识，字形相似性词典，字音相似性词典
            y = CuDNNGRU(self.hidden_dim, return_sequences=True)(y)
            y = CuDNNGRU(self.hidden_dim, return_sequences=True)(y)
        else:
            # encoder
            x = Bidirectional(GRU(int(self.hidden_dim / 2), return_sequences=True, dropout=self.dropout))(x)
            x = Bidirectional(GRU(int(self.hidden_dim / 2), return_sequences=True, dropout=self.dropout))(x)
            # decoder
            y = GRU(self.hidden_dim, return_sequences=True, dropout=self.dropout)(y)
            y = GRU(self.hidden_dim, return_sequences=True, dropout=self.dropout)(y)

        x = Lambda(lambda x: x, output_shape=lambda s: s)(x)
        y = Lambda(lambda x: x, output_shape=lambda s: s)(y)

        xy = Interact()([y, x, x_mask])
        xy = Dense(512, activation='relu')(xy)
        xy = Dense(len(self.chars))(xy)
        xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
        xy = Activation('softmax')(xy)

        # 交叉熵作为loss，但mask掉padding部分
        cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
        loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

        model = Model([x_in, y_in, x_p_in, y_p_in], xy)
        model.add_loss(loss)
        model.compile(optimizer=Adam(1e-3))
        if os.path.exists(self.model_path):
            model.load_weights(self.model_path)
        return model

    def _one_hot(self, x):
        """
        输出 one hot 向量
        :param x:
        :return:
        """
        x, x_mask = x
        x = K.cast(x, 'int32')
        x = K.one_hot(x, len(self.chars))
        x = K.sum(x_mask * x, 1, keepdims=True)
        x = K.cast(K.greater(x, 0.5), 'float32')
        return x
