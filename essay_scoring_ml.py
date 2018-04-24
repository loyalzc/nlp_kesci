# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/2/8 15:48
@Function:
"""
import nltk
import numpy as np
import pandas as pd
from textblob import Word
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor


path_train_lab = "data/essay_train.txt"
data = pd.read_csv(path_train_lab, names=["id", "essay", "s1", "s2", "score"], sep="\t")


def data_cleaning(data):
    print('---data_cleaning start...')
    # 分词
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    data["words"] = data["essay"].apply(tokenizer.tokenize)
    # 分句
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    data['sents'] = data["essay"].apply(sent_detector.tokenize)
    # 分字母,求得长度
    data['character_count'] = data['words'].apply(lambda x: len(''.join(x)))
    # 分词的tag(tag)
    data['tags'] = data['words'].apply(pos_tag)
    print('---data_cleaning end...')
    return data


def get_feature(data):
    print('---train_features start---')
    train_features = pd.DataFrame()
    # 单词数
    train_features['word_count'] = data['words'].apply(len)
    # 句子数
    train_features['sentence_count'] = data['sents'].apply(len)
    # 每个句子的平均单词数
    train_features['avg_sentence_len'] = train_features['word_count'].values / train_features['sentence_count'].values
    # 每个单词的平均长度
    train_features['avg_word_len'] = data['character_count'].values / train_features['word_count'].values
    # 长单词数
    train_features['long_word'] = data['words'].apply(lambda x: sum([len(word) >= 7 for word in x]))
    # 停用词个数
    stop_words = stopwords.words('english')
    train_features['stopwords_count'] = data['words'].apply(lambda x: len([word for word in x if word in stop_words]))
    # 感叹号出现次数
    train_features['exc_count'] = data['essay'].apply(lambda x: x.count('!'))
    # 问号出现次数
    train_features['que_count'] = data['essay'].apply(lambda x: x.count('?'))
    # 逗号出现次数
    train_features['comma_count'] = data['essay'].apply(lambda x: x.count(','))

    # # 拼写错误的单词数
    train_features['spelling_errors'] = data['words'].apply(
        lambda x: sum([Word(word).spellcheck()[0][0] != word for word in x]))

    # 求不重复的ngram数量：这可以表明作者用词、短语的丰富程度
    train_features['unigrams_count'] = data['words'].apply(lambda x: len(set([grams for grams in ngrams(x, 1)])))
    train_features['bigrams_count'] = data['words'].apply(lambda x: len(set([grams for grams in ngrams(x, 2)])))
    train_features['trigrams_count'] = data['words'].apply(lambda x: len(set([grams for grams in ngrams(x, 3)])))

    # 词性的 特征  名词 形容词 副词  动词 外来词
    train_features['noun_count'] = data['tags'].apply(lambda x: len([tag for tag in x if tag[1].startswith("NN")]))
    train_features['adj_count'] = data['tags'].apply(lambda x: len([tag for tag in x if tag[1].startswith("JJ")]))
    train_features['adv_count'] = data['tags'].apply(lambda x: len([tag for tag in x if tag[1].startswith("RB")]))
    train_features['verb_count'] = data['tags'].apply(lambda x: len([tag for tag in x if tag[1].startswith("VB")]))
    train_features['fw_count'] = data['tags'].apply(lambda x: len([tag for tag in x if tag[1].startswith("FW")]))

    # 分析文中每句话的语气: positive，negative or neutrual
    sid = SentimentIntensityAnalyzer()
    data['sents_polar'] = data['sents'].apply(lambda x: [sid.polarity_scores(sent) for sent in x])

    train_features['neg_sentiment'] = data['sents_polar'].apply(
        lambda x: sum([p for item in x for polar, p in item.items() if polar == "neg"]))
    train_features['neu_sentiment'] = data['sents_polar'].apply(
        lambda x: sum([p for item in x for polar, p in item.items() if polar == "neu"]))
    train_features['pos_sentiment'] = data['sents_polar'].apply(
        lambda x: sum([p for item in x for polar, p in item.items() if polar == "pos"]))
    print('---train_features end...')
    return train_features


def model(X_train, X_test, y_train, y_test):
    model_rf = RandomForestRegressor(n_estimators=30, min_samples_split=90, min_samples_leaf=10, max_depth=8,
                                       random_state=10)

    # 拟合score1
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)

    r, p = pearsonr(y_pred, y_test)
    print('Result pearsonr :', r)


if __name__ == '__main__':
    data = data_cleaning(data)
    train_feature = get_feature(data)
    print(np.shape(train_feature))
    X_train, X_test, y_train, y_test = train_test_split(train_feature, data['score'], test_size=0.2, random_state=2018)
    model(X_train, X_test, y_train, y_test)
    print('end....')