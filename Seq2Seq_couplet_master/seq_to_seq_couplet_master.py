import os

import tensorflow as tf
from seq_to_seq_couplet_master_utils import get_batches, train_en_arr, train_de_arr, get_test_arr, id_to_word
# from NLP.seq_to_seq_couplet_master_utils import get_batches, train_en_arr, train_de_arr, get_test_arr, id_to_word

seq_length = 30
vocab_size = 5000
embedding_dim = 300
lstm_size = 64
lstm_layers = 2
epochs = 20
batch_size = 512

# 读取文件
vocab_path = os.path.join(os.getcwd(), 'data', 'vocabs')
# vocab_path = os.path.join(os.getcwd(), 'NLP', 'data', 'vocabs')
train_in_txt_path = os.path.join(os.getcwd(), 'data', 'train', 'in.txt')
# train_in_txt_path = os.path.join(os.getcwd(), 'NLP', 'data', 'train', 'in.txt')
train_out_txt_path = os.path.join(os.getcwd(), 'data', 'train', 'out.txt')
# train_out_txt_path = os.path.join(os.getcwd(), 'NLP', 'data', 'train', 'out.txt')

train_en_arr, train_en_length = train_en_arr(train_in_txt_path, seq_length)
train_de_arr, train_de_length = train_de_arr(train_out_txt_path, seq_length)

tf.reset_default_graph()

# 数据入口
en_seq = tf.placeholder(tf.int32, [None, seq_length], name='en_seq')  # 编码器原始特征数据 -- 填充之后
en_length = tf.placeholder(tf.int32, [None], name='en_length')  # 编码器原始特征数据长度 -- 未填充
de_seq = tf.placeholder(tf.int32, [None, seq_length], name='de_seq')  # 解码器原始特征数据
de_length = tf.placeholder(tf.int32, [None], name='de_length')  # 解码器原始特征数据长度 -- 未填充
de_labels = tf.placeholder(tf.int32, [None, seq_length], name='de_labels')  # 解码器原始标签数据长度

# 搭建网络
# 1. 搭建嵌入层 [词 0 - 4999] 补充5000的词，所以需要嵌入维度为5001
# 希望5000所代表的词经过嵌入层后变为0，所以需要拼接一个全0的一行
en_embedding_matrix = tf.get_variable('en_embedding_matrix', [vocab_size, embedding_dim])
de_embedding_matrix = tf.get_variable('de_embedding_matrix', [vocab_size, embedding_dim])
zero_embedding = tf.zeros([1, embedding_dim])

en_embedding_matrix = tf.concat([en_embedding_matrix, zero_embedding], axis=0)
de_embedding_matrix = tf.concat([de_embedding_matrix, zero_embedding], axis=0)

en_embedding = tf.nn.embedding_lookup(en_embedding_matrix, en_seq)
de_embedding = tf.nn.embedding_lookup(de_embedding_matrix, de_seq)


# 2. 搭建LST层
def lstm_cell():
    return tf.nn.rnn_cell.BasicLSTMCell(lstm_size)


with tf.variable_scope('encoder'):
    en_multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
    _, en_final_state = tf.nn.dynamic_rnn(en_multi_cell, en_embedding, sequence_length=en_length, dtype=tf.float32)

with tf.variable_scope('decoder'):
    de_multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
    de_output, de_final_state = tf.nn.dynamic_rnn(de_multi_cell, de_embedding, sequence_length=de_length,
                                                  initial_state=en_final_state)

de_output = tf.reshape(de_output, [-1, lstm_size])
softmax_weights = tf.Variable(tf.random_normal([lstm_size, vocab_size + 1], stddev=.1))
softmax_bias = tf.Variable(tf.zeros([vocab_size + 1]))

logits = tf.matmul(de_output, softmax_weights) + softmax_bias

max_logits_id = tf.argmax(logits, axis=1)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(de_labels, [-1]))

sequence_mask = tf.sequence_mask(de_length, maxlen=seq_length, dtype=tf.float32)
sequence_mask = tf.reshape(sequence_mask, [-1])
cost = tf.reduce_mean(loss * sequence_mask)

optimizer = tf.train.AdamOptimizer().minimize(cost)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        batch_count = 0
        for batch_en_arr, batch_en_length, batch_de_arr, batch_de_length, batch_de_labels in get_batches(train_en_arr,
                                                                                                         train_en_length,
                                                                                                         train_de_arr,
                                                                                                         train_de_length,
                                                                                                         batch_size):
            batch_count += 1
            feed = {
                en_seq: batch_en_arr,
                en_length: batch_en_length,
                de_seq: batch_de_arr,
                de_length: batch_de_length,
                de_labels: batch_de_labels
            }
            _, train_loss = sess.run([optimizer, cost], feed_dict=feed)
            if batch_count % 100 == 0:
                print(f'Epoch: {epoch + 1}\tIteration: {batch_count}\tTrain loss: {train_loss}')
            if batch_count % 500 == 0:
                # 保存模型
                # saver.save(sess, os.path.join(os.getcwd(), 'NLP', 'couplet_model', 'model_' + str(batch_count)))
                saver.save(sess, os.path.join(os.getcwd(), 'couplet_model', 'model_' + str(batch_count)))

# 测试
test_str = '春眠不觉晓'
test_en_arr, test_en_length = get_test_arr(test_str)
test_en_length = test_en_length[None]

de_start = ['<s>']
test_de_arr, test_de_length = get_test_arr(de_start)
test_de_length = test_de_length[None]

sess = tf.Session(config=config)
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'NLP', 'couplet_model'))
test_model = saver.restore(sess, checkpoint)
# 1. 先得到编码器的最终输出状态
feed = {
    en_seq: test_en_arr,
    en_length: test_en_length,
}
en_state = sess.run(en_final_state, feed_dict=feed)

# 2. 再得到解码器的输出
words = []
for i in range(seq_length):
    feed = {
        de_seq: test_de_arr,
        de_length: test_de_length,
        en_final_state: en_state
    }
    max_id, en_state = sess.run([max_logits_id, de_final_state], feed_dict=feed)
    word = id_to_word(max_id[0])
    if word == '</s>':
        break
    test_de_arr, test_de_length = get_test_arr([word])
    test_de_length = test_de_length[None]
    words.append(word)

words = ''.join(words)

print(f'上联：{test_str}')
print(f'下联：{words}')
