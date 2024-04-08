import os
import codecs
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
import numpy as np
import pandas
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# 设置预训练模型的路径
pretrained_path = r'C:\Users\GYHan\Desktop\论文\ALLCODE\chinese_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# 加载预训练模型
model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=128)
tokenizer = Tokenizer(token_dict)

# texts = []
bert_cls=[]
bert_all=[]
bert_mean=[]
df = pandas.read_csv(r'C:\Users\GYHan\Desktop\论文集成代码_自己数据集\data\selfdata\self_test1.csv')
# i=0
# for i in range(p.shape[0]):
#     texts.append(p.iloc[i, 2])
#     texts.append(p.iloc[i, 3])
#     texts.append(p.iloc[i, 4])
#     texts.append(p.iloc[i, 5])
#     texts.append(p.iloc[i, 6])
#     texts.append(p.iloc[i, 7])
# print('size:{}'.format(p.shape))
# print('texts:{}'.format(len(texts)))
texts=df['filename'].values
# s=enumerate(texts)
for text in tqdm(texts):
    tokens = tokenizer.tokenize(text)
    indices, segments = tokenizer.encode(first=text, max_len=128)
    predicts = model.predict([np.array([indices]), np.array([segments])])[0]
    bert_cls.append(predicts[0])
    #     bert_mean.append(np.mean(predicts[1:len(tokens) - 1], axis=0))
    # else:
    #     bert_all.append(np.mean(predicts[0:len(tokens) - 1], axis=0))
np.save(r'C:\Users\GYHan\Desktop\论文集成代码_自己数据集\data\selfdata\self_test.npy', bert_cls)
# np.save('bertdata/bertmean.npy', bert_mean)
# np.save('bertdata/bertall.npy', bert_all)
