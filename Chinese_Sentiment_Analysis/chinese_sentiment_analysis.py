import os

import numpy as np
import re
import jieba
from gensim.models import KeyedVectors
from keras.optimizers import Adam

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 词嵌入：找到一个合适的数组来表示一个单词

# 导入词向量矩阵
word2vec_path = os.path.join(os.getcwd(), "chinese_word_vectors", "sgns.zhihu.bigram")
cn_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)

# 导入数据
pos_data_path = os.path.join(os.getcwd(), 'pos')
neg_data_path = os.path.join(os.getcwd(), 'neg')
pos_txts = os.listdir(pos_data_path)
neg_txts = os.listdir(neg_data_path)

# 读出每一个文档的数据并存入列表
train_txt_original = []
for i in range(len(pos_txts)):
    txt_path = os.path.join(pos_data_path, pos_txts[i])
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read().strip()
        train_txt_original.append(txt)

for i in range(len(neg_txts)):
    txt_path = os.path.join(neg_data_path, neg_txts[i])
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read().strip()
        train_txt_original.append(txt)

# 分词 和 tokenize
train_tokens = []
pattern = '[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+'
for txt in train_txt_original:
    # 1. 去掉标点
    txt = re.sub(pattern, '', txt)
    # 2. jieba分词，结巴分词的输出为一个生成器，需要将生成器转换为list
    cut = jieba.cut(txt)
    cut_lst = [ele for ele in cut]
    # 3. 将语句转换为数字词典的形式
    for i, word in enumerate(cut_lst):
        try:
            cut_lst[i] = cn_model.key_to_index[word]
        except:
            cut_lst[i] = 0
    train_tokens.append(cut_lst)

# 确定 tokens 的长度
tokens_length = np.array([len(token) for token in train_tokens])
max_tokens = int(np.mean(tokens_length) + 2 * np.std(tokens_length))


# 反向 token
def reverse_tokens(token):
    txt = ''
    for i in token:
        if i != 0:
            txt = txt + cn_model.index_to_key[i]
        else:
            txt = txt + ''
    return txt


embedding_dim = 300
num_words = 50000
embedding_matrix = np.zeros(shape=[num_words, embedding_dim])  # 设计一个小的嵌入层
for i in range(num_words):
    embedding_matrix[i] = cn_model[cn_model.index_to_key[i]]
embedding_matrix = embedding_matrix.astype('float32')

train_pad = pad_sequences(train_tokens, maxlen=max_tokens, value=0, padding='pre', truncating='pre')
train_pad[train_pad >= num_words] = 0

# 设置标签
train_targets = np.concatenate((np.ones(3000), np.zeros(3000)))

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(train_pad, train_targets, test_size=0.1, random_state=420)

# 搭建网络
# 输入矩阵维度 -- (batch_size, max_tokens)
# 经过嵌入层后 -- (batch_size, max_tokens, embedding_dim)

model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_tokens, trainable=False))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

# 定义 callbacks
early_stopping = EarlyStopping(patience=3, verbose=1)
reduce_lr = ReduceLROnPlateau(min_lr=1e-5, patience=0, verbose=1)
callbacks = [early_stopping, reduce_lr]

model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.1, callbacks=callbacks)


# test
def predict_sentiment(txt):
    # 去掉标点
    txt = re.sub('[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+', '', txt)
    # 分词
    cut = jieba.cut(txt)
    cut_list = [i for i in cut]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.key_to_index[word]
        except:
            cut_list[i] = 0
    # padding
    tokes_pad = pad_sequences([cut_list], maxlen=max_tokens,
                              padding='pre',
                              truncating='pre')
    # 预测
    result = model.predict(tokes_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价', 'output=%.2f' % coef)
    else:
        print('是一例负面评价', 'output=%.2f' % coef)


test_list = [
    '酒店设施不是新的，服务态度很不好',
    '酒店卫生条件非常不好',
    '床铺非常舒适',
    '我觉得还好吧'
]
for txt in test_list:
    predict_sentiment(txt)
