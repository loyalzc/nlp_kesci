# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/1/29 23:03
@Function:
"""

import nltk
from gensim.models.word2vec import Word2Vec
from nltk import WordNetLemmatizer, RegexpTokenizer
import numpy as np
from nltk.corpus import stopwords
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import collections
import pandas as pd


path_train_lab = "data/sentence_train.txt"
data = pd.read_csv(path_train_lab, names=["id", "s1", "s2", "score"], sep="\t")


def data_cleaning(data):
    data["s1"] = data["s1"].str.lower()
    data["s2"] = data["s2"].str.lower()

    # 分词
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    data["s1_token"] = data["s1"].apply(tokenizer.tokenize)
    data["s2_token"] = data["s2"].apply(tokenizer.tokenize)

    # 去停用词
    stop_words = stopwords.words('english')
    def word_clean_stopword(word_list):
        words = [word for word in word_list if word not in stop_words]
        return words

    data["s1_token"] = data["s1_token"].apply(word_clean_stopword)
    data["s2_token"] = data["s2_token"].apply(word_clean_stopword)

    # 词形还原
    lemmatizer = WordNetLemmatizer()
    def word_reduction(word_list):
        words = [lemmatizer.lemmatize(word) for word in word_list]
        return words

    data["s1_token"] = data["s1_token"].apply(word_reduction)
    data["s2_token"] = data["s2_token"].apply(word_reduction)

    # 词干化
    stemmer = nltk.stem.SnowballStemmer('english')
    def word_stemming(word_list):
        words = [stemmer.stem(word) for word in word_list]
        return words

    data["s1_token"] = data["s1_token"].apply(word_stemming)
    data["s2_token"] = data["s2_token"].apply(word_stemming)

    return data


def count_vector(data):
    """bag of words"""
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(data["sents_bow"].tolist())

    sent1 = count_vectorizer.transform(data["s1_sent"])
    sent2 = count_vectorizer.transform(data["s2_sent"])
    return sent1, sent2


def tfidf_vector(data):
    """TF-IDF"""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(data["sents_bow"].tolist())
    sent1 = tfidf_vectorizer.transform(data["s1_sent"])
    sent2 = tfidf_vectorizer.transform(data["s2_sent"])
    return sent1, sent2


def word2vec_vector(data):
    """word2vec"""
    model = Word2Vec(data['words_bow'], size=200, min_count=1, iter=200, window=10)

    def get_matr(sents):
        sents_vec = []
        for sent in sents:
            s_len = len(sent)
            vec = np.zeros(200)
            for word in sent:
                vec += model.wv[word]
            if s_len > 1:
                vec = vec / s_len
            sents_vec.append(vec.reshape(1, -1))
        return sents_vec
    sent1 = get_matr(data['s1_token'])
    sent2 = get_matr(data['s2_token'])
    return sent1, sent2


def ICLR_vector(data):
    """ICLR : A simple but tough-to-beat baseline for sentence embedding"""
    model = Word2Vec(data['words_bow'], size=200, min_count=1, iter=200, window=10)
    words_list = []
    for sent in data['words_bow']:
        words_list += sent
    word_dict = collections.Counter(words_list)
    for key in word_dict:
        word_dict[key] = word_dict[key] / len(word_dict)

    a = 1e-3 / 4

    def get_matr(sents):
        sents_vec = []
        for sent in sents:
            s_len = len(sent)
            vec = np.zeros(200)
            for word in sent:
                vec += a / (a + word_dict[word]) * model.wv[word]
            if s_len > 1:
                vec = vec / s_len
            sents_vec.append(vec)
        return sents_vec

    sent1 = get_matr(data['s1_token'])
    sent2 = get_matr(data['s2_token'])

    data = sent1 + sent2

    print('np.shape(data): ', np.shape(data))

    def PCA_prep(data):
        """
        PCA 算法进行数据转换
        :param all_train:
        :return:
        """
        pca = PCA(n_components=200)
        pca.fit(data)
        u = pca.components_[0]
        u = np.multiply(u, np.transpose(u))
        mat = []
        for vs in data:
            mat.append(vs - np.multiply(u, vs))
        return mat
    data = PCA_prep(data)
    sent1 = data[0: len(sent1)]
    sent2 = data[len(sent1):]
    return sent1, sent2


def get_pearsonr(sent1, sent2):
    print(np.shape(sent1), np.shape(sent2))
    y_pred = []
    for s1, s2 in zip(sent1, sent2):
        # y_pred.append(cosine_similarity([s1], [s2])[0][0])
        y_pred.append(cosine_similarity(s1, s2)[0][0])
    r, p = pearsonr(y_pred, data['score'].tolist())
    print('Result pearsonr :', r)


if __name__ == '__main__':
    data = data_cleaning(data)
    data = data[["id", "s1_token", "s2_token", "score"]]
    data['s1_sent'] = data['s1_token'].apply(lambda x: ' '.join(x))
    data['s2_sent'] = data['s2_token'].apply(lambda x: ' '.join(x))

    data["sents_bow"] = data["s1_sent"] + data["s2_sent"]
    data["words_bow"] = data["s1_token"] + data["s2_token"]

    # sent1, sent2 = count_vector(data)
    # sent1, sent2 = tfidf_vector(data)
    sent1, sent2 = word2vec_vector(data)
    # sent1, sent2 = ICLR_vector(data)
    get_pearsonr(sent1, sent2)


    print('end...')