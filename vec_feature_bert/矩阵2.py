from tqdm import tqdm
import pandas as pd
from bert_serving.client import BertClient
bc = BertClient()
import numpy
import numpy as np
def get_vec(text):
    return bc.encode(text)
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos

df = pd.read_csv(r'C:\Users\GYHfresh\Desktop\add_vec\lutoushe\gyhuse_train.csv')
dfindex=pd.read_csv(r'C:\Users\GYHfresh\Desktop\ALLCODE\data\index\lutoushe_index.csv')
all_label = dfindex['LABEL'].values

num_in = 0
num_all = 0


pros = np.random.randint(0, 1, 31 * 31)
pros = pros.reshape(31, 31)
pros = pros.astype(np.float32)


for i in tqdm(range(len(all_label))):
    for j in range(len(all_label)):
        label1 = all_label[i]
        label2 = all_label[j]
        a, b = get_vec([label1, label2])
        pros[i][j] = cos_sim(a,b)

numpy.savetxt(r'C:\Users\GYHfresh\Desktop\ALLCODE\data\vec\luvec2.csv', pros, delimiter = ',')

all_list = df.values.tolist()
num_in = 0
num_all = 0

pros = np.random.randint(0, 1, 31 * 31)
pros = pros.reshape(31, 31)
pros = pros.astype(np.float32)

for i in tqdm(range(len(all_label))):
    for j in range(len(all_label)):
        label1 = all_label[i]
        label2 = all_label[j]

        num_in = 0
        num_all = 0
        for num in range(df.shape[0]):
            if label2 in all_list[num]:
                num_all += 1
            if label1 in all_list[num] and label2 in all_list[num]:
                num_in += 1

        if num_all == 0:
            pro = 0
        else:
            pro = num_in / num_all
        pros[i][j] = pro
        if i==j:
            pros[i][j] = 0
        if pros[i][j] ==1:
            print(i,j)
numpy.savetxt(r'C:\Users\GYHfresh\Desktop\ALLCODE\data\vec\luvec3.csv', pros, delimiter = ',')