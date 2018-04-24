## 英文语句相似度 作文自动打分

### 句子相似度
#### 博客：http://www.cnblogs.com/infaraway/p/8666269.html
- CountVectorizer 向量
- TF-IDF 向量
- word2vec 向量
- ICLR2017 论文方案：A simple but tough-to-beat baseline for sentence embedding


### 英文作文自动打分

- 传统机器学习方案：提取文本特征； Machine learning based

 1. 长度相关：Length Features
    - 单词数 word_count
    - 句子数 sentence_count
    - 每个句子的平均单词数 avg_sentence_len
    - 每个单词的平均长度 avg_word_len
    - 长单词数long_word (这里选定长度≥7的为长单词)
    - 停用词个数stopwords_count

 2. 标点相关：Occurrence Features
    - 感叹号出现的数目exc_count
    - 问号出现的数目que_count
    - 逗号出现的数目comma_count

 3. Error Features
    - 拼写错误的单词数spelling_errors

 4. n-gram相关：ngrams_counts Features， 此特征可以说明作者的词汇丰富程度
    - unigrams_count：将文章分词后-->采用1-gram-->统计非重复gram的个数
    - bigrams_count：将文章分词后-->采用2-grams-->统计非重复grams的个数
    - trigrams_count：将文章分词后-->采用3-grams-->统计非重复grams的个数

 5. 词性相关：POS counts Features，此特征用于统计文章中不同词性的个数
    - 名词noun_count
    - 形容词adj_count
    - 副词adv_count
    - 动词verb_count
    - 外来词fw_count

 6. 语气相关：Personality Features 分析文中每句话的语气:positive，negative or neutrual
    - 消极语气neg_sentiment
    - 中立语气neu_sentiment
    - 积极语气pos_sentiment

- 深度学习方式： Deeplearning based
 ToDo



