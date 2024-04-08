import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences


def data_loader(path, y_name):
    df = pd.read_csv(path)
    label = df[y_name]
    text = df['text'].values
    lastpre = [i for i in range(df.shape[0])]
    return text, np.array(label.values)


def data_process(data, max_len):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)
    input_ids = pad_sequences(
        input_ids,
        maxlen=max_len,
        dtype='long',
        value=0,
        truncating='post',
        padding='post')
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)
    return input_ids, attention_masks


def batch_iter(x, y,  batch_size, lastpre):
    """
    :param x: data
    :param y: label
    :param batch_size: how many samples in one single batch
    :return: a batch of data
    """
    label_pro = [[], []]
    data_len = len(x)
    num_batch = int(
        data_len / batch_size) if data_len % batch_size == 0 else int(data_len / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))

    x = np.array(x)[indices]

    y = np.array(y)[indices]

    lastpre= np.array(lastpre)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = (i + 1) * batch_size
        yield x[start_id:end_id], y[start_id:end_id],lastpre[start_id:end_id]
