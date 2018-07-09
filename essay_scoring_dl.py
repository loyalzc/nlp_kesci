# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/2/8 15:48
@Function:
"""
import pandas
from keras import Sequential
from keras.layers import Embedding
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import nltk
from nltk import WordNetLemmatizer, RegexpTokenizer
import numpy as np
import pandas as pd
from tensorflow.contrib import learn
import tensorflow as tf
FLAGS = None


MAX_DOCUMENT_LENGTH = 1000  # 文档最长长度
MIN_WORD_FREQUENCE = 1  # 最小词频数
EMBEDDING_SIZE = 20     # 词向量的维度
N_FILTERS = 10          # filter个数
WINDOW_SIZE = 20        # 感知野大小
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]    # filter的形状
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0


path_train_lab = "data/essay_train.txt"
data = pd.read_csv(path_train_lab, names=["id", "essay", "s1", "s2", "score"], sep="\t")


def data_cleaning(data):

    data["essay"] = data["essay"].str.lower()

    # 分词
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    data["essay_token"] = data["essay"].apply(tokenizer.tokenize)

    # 去停用词
    # stop_words = stopwords.words('english')
    # data["essay_token"] = data["essay_token"].apply(lambda x: [word for word in x if word not in stop_words])

    # 词形还原
    lemmatizer = WordNetLemmatizer()
    data["essay_token"] = data["essay_token"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # 词干化
    stemmer = nltk.stem.SnowballStemmer('english')
    data["essay_token"] = data["essay_token"].apply(lambda x: [stemmer.stem(word) for word in x])
    data = data.fillna(6)
    essays = []
    for essay, score in zip(data['essay_token'], data['score']):
        essays.append((' '.join(essay), int(score)))

    return essays


def CNN_model(features, target):
    """
    基于卷积神经网络的中文文本分类 2层的卷积神经网络，用于短文本分类
    """
    # 先把词转成词向量
    # 我们得到一个形状为[n_words, EMBEDDING_SIZE]的词表映射矩阵
    # 接着我们可以把一批文本映射成[batch_size, sequence_length, EMBEDDING_SIZE]的矩阵形式
    target = tf.one_hot(target, 12, 1, 0)  # score
    word_vectors = tf.contrib.layers.embed_sequence(
        features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')
    word_vectors = tf.expand_dims(word_vectors, 3)

    with tf.variable_scope('CNN_Layer1'):

        conv1 = tf.contrib.layers.convolution2d(word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1], strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        conv2 = tf.contrib.layers.convolution2d(pool1, N_FILTERS, FILTER_SHAPE2, padding='VALID')
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    # 全连接层
    logits = tf.contrib.layers.fully_connected(pool2, 12, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                               optimizer='Adam', learning_rate=0.02)

    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


if __name__ == '__main__':
    essays = data_cleaning(data)
    x, y = zip(*essays)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2018)
    # essay 展开 长度为1000的向量
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
    x_train = np.array(list(vocab_processor.fit_transform(X_train)))

    x_test = np.array(list(vocab_processor.transform(X_test)))
    n_words = len(vocab_processor.vocabulary_)
    y_train = pandas.Series(y_train)
    y_test = pandas.Series(y_test)
    print('Total words: %d' % n_words)
    print(x_train.shape)

    # 构建模型
    classifier = learn.SKCompat(learn.Estimator(model_fn=CNN_model))
    # 训练和预测
    classifier.fit(x_train, y_train, steps=1000)
    y_pred = classifier.predict(x_test)['class']
    r, p = pearsonr(y_pred, y_test)
    print('Result pearsonr :', r)

    print('end...')
