import os

import numpy as np
import tensorflow as tf

# 1. 导入数据集
data_path = os.path.join(os.getcwd(), "Anna1.txt")

with open(data_path, mode="r") as f:
    txt = f.read()

# 2. 处理数据集
# 2.1. 提取字符集
vocab = sorted(set(txt))
# 2.2. 将字符和数字编码一一对应
vocab_to_int = {v: k for k, v in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
# 2.3. 将原始字符集表示成数字编码格式
encoded = np.array([vocab_to_int[v] for v in txt], dtype=np.int)


# 3. 定义批次数据生成器
def get_batch(encoded, batch_size, n_steps):
    # 得到每一个批次的数据长度
    len_batch = batch_size * n_steps
    n_batches = len(encoded) // len_batch
    encoded = encoded[:len_batch * n_batches]  # 将多余的数据去掉
    encoded = encoded.reshape((batch_size, -1))

    for i in range(0, encoded.shape[1], n_steps):
        x = encoded[:, i:i + n_steps]  # 特征
        y_temp = encoded[:, i + 1: i + 1 + n_steps]  # 标签
        # 解决标签数据的列维度少1的情况
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:, :y_temp.shape[1]] = y_temp

        yield x, y


num_classes = len(vocab)


class CharRNN:
    def __init__(self, batch_size, n_steps):
        tf.reset_default_graph()

        self.lstm_size = 64
        self.lstm_layers = 2

        self.inputs = tf.placeholder(shape=[None, n_steps], dtype=tf.int32, name="inputs")
        inputs_hot = tf.one_hot(self.inputs, num_classes)  # 独热编码 -- 升维

        self.targets = tf.placeholder(shape=[None, n_steps], dtype=tf.int32, name="targets")
        targets_hot = tf.one_hot(self.targets, num_classes)  # 独热编码 -- 升维

        self.keep_prob = tf.placeholder(tf.float32)

        def lstm_cell():
            # 搭建一个基本的 lstm 单元, 带有时间维度
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 按层搭建 LSTM 网络
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.lstm_layers)])
        # 初始化隐含层状态
        self.initial_state = multi_cell.zero_state(batch_size, tf.float32)
        output, self.final_state = tf.nn.dynamic_rnn(multi_cell, inputs_hot, initial_state=self.initial_state)

        output = tf.reshape(output, shape=(-1, self.lstm_size))  # 接入全连接网络前需要将三维数据转化为二维数据

        weights_out = tf.Variable(tf.random_normal([self.lstm_size, num_classes], stddev=.1))
        bias_out = tf.Variable(tf.zeros([num_classes]))

        logits = tf.matmul(output, weights_out) + bias_out
        self.predictions = tf.nn.softmax(logits)  # 得到概率输出

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets_hot, logits=logits))

        t_var = tf.trainable_variables()
        grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_var), 5)
        train_op = tf.train.AdamOptimizer()
        self.optimizer = train_op.apply_gradients(zip(grad, t_var))


batch_size = 64
n_steps = 50
epochs = 30
keep_prob_ = 0.5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_model = CharRNN(batch_size=batch_size, n_steps=n_steps)
saver = tf.train.Saver()

# 训练阶段
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        new_state = sess.run(train_model.initial_state)
        for batch_x, batch_y in get_batch(encoded, batch_size=batch_size, n_steps=n_steps):
            feed_dict = {
                train_model.inputs: batch_x,
                train_model.targets: batch_y,
                train_model.keep_prob: keep_prob_,
                train_model.initial_state: new_state
            }
            _, train_loss_, new_state = sess.run([train_model.optimizer, train_model.loss, train_model.final_state],
                                                 feed_dict=feed_dict)
        print('Epoch: {}\tloss: {}'.format(epoch, train_loss_))
        saver.save(sess, './checkpoint2/i_epoch{}'.format(epoch))


def pick_top_n(preds, vocab_size, top_n=5):
    preds = np.squeeze(preds)
    preds[np.argsort(preds)[:-top_n]] = 0  # 将概率最大的5个数之外的数均置为0
    preds = preds / sum(preds)
    return np.random.choice(vocab_size, p=preds)


# 测试阶段
# 两个网络的参数是一样的，但是数据接口是不一样的
test_model = CharRNN(batch_size=1, n_steps=1)
checkpoint = tf.train.latest_checkpoint("./checkpoint2")
start = "The"
n_samples = 100
samples = [c for c in start]

saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    saver.restore(sess, checkpoint)
    new_state = sess.run(test_model.initial_state)
    for c in start:
        x = np.zeros([1, 1])
        x[0][0] = vocab_to_int[c]
        predictions, new_state = sess.run([test_model.predictions, test_model.final_state], feed_dict={
            test_model.inputs: x,
            test_model.keep_prob: 1.,
            test_model.initial_state: new_state
        })

    c = pick_top_n(predictions, len(vocab))
    samples.append(int_to_vocab[c])

    res = ""
    for i in range(n_samples):
        x[0][0] = c
        predictions, new_state = sess.run([test_model.predictions, test_model.final_state], feed_dict={
            test_model.inputs: x,
            test_model.keep_prob: 1.,
            test_model.initial_state: new_state
        })

        c = pick_top_n(predictions, len(vocab))
        res += int_to_vocab[c]
        samples.append(int_to_vocab[c])



