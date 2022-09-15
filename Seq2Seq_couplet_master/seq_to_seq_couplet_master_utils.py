import os

import numpy as np

# 词表数据文件地址
vocab_path = os.path.join(os.getcwd(), 'data', 'vocabs')
# vocab_path = os.path.join(os.getcwd(), 'NLP', 'data', 'vocabs')

vocab_size = 5000

# 读取词
vocab_lst = []
with open(vocab_path, encoding='utf-8') as f:
    for vocab in f:
        vocab_lst.append(vocab.strip())
vocab_lst = vocab_lst[:vocab_size]

# 字和数字列表之间的转化
word_to_int = {word: c for c, word in enumerate(vocab_lst)}
int_to_word = dict(enumerate(vocab_lst))

# 对联数据文件地址
# train_in_txt_path = os.path.join(os.getcwd(), 'NLP', 'data', 'train', 'in.txt')
train_out_txt_path = os.path.join(os.getcwd(), 'data', 'train', 'out.txt')

seq_en_length = 30


def train_en_arr(train_en_dir, seq_length):
    with open(train_en_dir, encoding='utf-8') as f:
        train_en_arr = []
        train_en_length = []
        for line in f:
            words = line.split()
            arr = []
            num_words = len(words)
            # 将每个字符转化为字典里的序号
            for word in words:
                if word in word_to_int:
                    arr.append(word_to_int[word])
                else:
                    arr.append(vocab_size)
            if num_words < seq_length:
                arr += [vocab_size] * (seq_length - num_words)
                train_en_length.append(num_words)
            else:
                arr = arr[:seq_length]
                train_en_length.append(seq_length)
            train_en_arr.append(arr)
    return np.array(train_en_arr), np.array(train_en_length)


def train_de_arr(train_de_dir, seq_length):
    with open(train_de_dir, encoding='utf-8') as f:
        train_de_arr = []
        train_de_length = []
        for line in f:
            words = line.split()
            words = ['<s>'] + words + ['</s>']
            arr = []
            num_words = len(words)
            # 将每个字符转化为字典里的序号
            for word in words:
                if word in word_to_int:
                    arr.append(word_to_int[word])
                else:
                    arr.append(vocab_size)
            if num_words < seq_length + 1:
                # 不足序列长度，需要补齐
                arr += [vocab_size] * (seq_length + 1 - num_words)
                train_de_length.append(num_words)
            else:
                # 超过序列长度，需要截断
                arr = arr[:seq_length + 1]
                train_de_length.append(seq_length)
            train_de_arr.append(arr)
    return np.array(train_de_arr), np.array(train_de_length)


batch_size = 128


def get_batches(train_en_arr, train_en_length, train_de_arr, train_de_length, batch_size):
    num_batches = train_en_arr.shape[0] // batch_size
    end_idx = batch_size * num_batches
    train_en_arr = train_en_arr[:end_idx]
    train_en_length = train_en_length[:end_idx]
    train_de_arr = train_de_arr[:end_idx]
    train_de_length = train_de_length[:end_idx]
    for i in range(0, end_idx, batch_size):
        batch_en_arr = train_en_arr[i:i + batch_size, :]
        batch_en_length = train_en_length[i:i + batch_size]
        batch_de_arr = train_de_arr[i:i + batch_size, :-1]
        batch_de_length = train_de_length[i:i + batch_size]
        batch_de_labels = train_de_arr[i:i + batch_size, 1:]
        yield batch_en_arr, batch_en_length, batch_de_arr, batch_de_length, batch_de_labels


seq_length = 30


def get_test_arr(test_str):
    test_str = ' '.join(test_str)
    words = test_str.split()
    arr = []
    num_words = len(words)
    for word in words:
        if word in word_to_int:
            arr.append(word_to_int[word])
        else:
            arr.append(vocab_size)
    if num_words < seq_length:
        arr += [vocab_size] * (seq_length - num_words)
    else:
        arr = arr[:seq_length]
        num_words = seq_length
    return np.array(arr)[None, :], np.array(num_words)


def id_to_word(id_):
    return int_to_word[id_]
