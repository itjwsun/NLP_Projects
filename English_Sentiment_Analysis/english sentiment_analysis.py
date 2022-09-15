import os

import numpy as np
import tensorflow as tf

# 打开文件
reviews_path = os.path.join(os.getcwd(), "data/reviews.txt")
labels_path = os.path.join(os.getcwd(), "data/labels.txt")
with open(reviews_path, "r") as f:
    reviews = f.read()
with open(labels_path, "r") as f:
    labels = f.readlines()

# 所有的NLP处理的任务第一步，都是将文本变成数字
# 为了把文档变成数字，需要一个vocabulary

# 1. 预处理第一步，将标点符号去掉
from string import punctuation

reviews = "".join([c for c in reviews if c not in punctuation])
reviews = reviews.split("\n")
all_txt = "".join(reviews)  # 去掉回车换行符
all_word = all_txt.split()  # 将单词拆分出来

# 统计单词出现频次
from collections import Counter

count = Counter(all_word)  # 统计字符和其出现的频次
vocab = sorted(count, key=count.get, reverse=True)  # 按单词出现频次构建单词表

# 2. 将每一个单词转换为数字的字典
vocab_to_int = {word: i for i, word in enumerate(vocab)}

# 按照构建好的字典将原始数据集转换为数字列表
reviews_int = []
for review in reviews:
    words = review.split()
    reviews_int.append([vocab_to_int[word] for word in words])

# 到此步为止完成了初步的文本到数字的转换

# 计算每一条评论的长度
review_length = []
for review in reviews_int:
    review_length.append(len(review))

# 3.   虽然循环神经网络可以处理不同长度的序列，但是由于我们是一个批次一个批次，
#      所以还是需要把所有的样本归一化到同样的长度
seq_length = 200

features = np.zeros(shape=(len(reviews), seq_length))
# 填充特征数组
for i, review in enumerate(reviews_int):
    features[i, -len(review):] = review[:seq_length]

# 处理标签数据
labels = [1 if label == "positive\n" else 0 for label in labels]
labels = np.array(labels)[:, None]

# 4. 打乱数据集
random_lst = list(range(len(reviews)))
np.random.shuffle(random_lst)
features = features[random_lst]
labels = labels[random_lst]

# 划分数据集和测试集
train_features, val_features, test_features = features[:20000], features[20000:25000], features[22500:]
train_labels, val_labels, test_labels = labels[:20000], labels[20000:25000], labels[22500:]


# 6. 在搭建网络之前，写一个get_batch方法
def get_batch(features, labels, batch_size):
    num_batches = features.shape[0] // batch_size
    features = features[:num_batches * batch_size]  # 去掉不完整batch的数据
    for i in range(0, features.shape[0], batch_size):
        yield features[i:i + batch_size], labels[i:i + batch_size]


# 搭建网络
lstm_size = 64
lstm_layers = 2
batch_size = 128
epochs = 20
embedding_size = 300

tf.reset_default_graph()

inputs = tf.placeholder(tf.int32, shape=[None, seq_length])
targets = tf.placeholder(tf.float32, shape=[None, 1])

# 接入嵌入层
embedding_weights = tf.Variable(tf.random_normal(shape=[len(vocab), embedding_size], stddev=.1))
embedded = tf.nn.embedding_lookup(embedding_weights, inputs)


def lstm_cell():
    return tf.nn.rnn_cell.BasicLSTMCell(lstm_size)


multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
initial_state = multi_cell.zero_state(batch_size, tf.float32)
output, final_state = tf.nn.dynamic_rnn(multi_cell, embedded, initial_state=initial_state)

# 重点：对于当前的分类问题，我们只需要最后一个时刻的输出，所以我们对Outputs第二维度，也就是时间维度的前面部分，都不感兴趣
# 只对，最后一个值感兴趣，所以说我们拿到了outputs[:,-1,:]

# 输出层
weights_out = tf.Variable(tf.random_normal(shape=[lstm_size, 1], stddev=.1))
bias_out = tf.Variable(tf.zeros(1))

logits = tf.matmul(output[:, -1, :], weights_out) + bias_out
predictions = tf.nn.sigmoid(logits)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets))
optimizer = tf.train.AdamOptimizer().minimize(cost)
corrected_predictions = tf.equal(tf.cast(tf.round(predictions), tf.int32), tf.cast(targets, tf.int32))
accuracy = tf.reduce_mean(tf.cast(corrected_predictions, tf.float32))

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    sess.run(init)
    for epoch in range(epochs):
        for batch_x, batch_y in get_batch(train_features, train_labels, batch_size):
            sess.run(optimizer, feed_dict={
                inputs: batch_x,
                targets: batch_y
            })
        val_acc_lst = []
        for batch_x, batch_y in get_batch(val_features, val_labels, batch_size):
            val_acc = sess.run(accuracy, feed_dict={
                inputs: batch_x,
                targets: batch_y
            })
            val_acc_lst.append(val_acc)
        print(f"Epoch: {epoch}\tval_acc: {np.mean(val_acc_lst)}")
        saver.save(sess, f"./checkpoint3/{epoch}_epoch")
