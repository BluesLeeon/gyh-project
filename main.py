import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, \
    confusion_matrix
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import numpy as np
import torch
import utils
import time
import getdata,Model,ttve,Model2
import warnings,os
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class Config():
    def __init__(self):
        super().__init__()
        self.train_path = "data/selfdata/self_train1.csv"
        self.test_path = "data/selfdata/self_test1.csv"
        self.valid_path = "data/selfdata/self_test1.csv"
        self.model_save_path = "model/model.pt"
        self.train_loss_path = "loss_record/train_loss_path.txt"
        self.test_loss_path = "loss_record/test_loss_path.txt"
        self.max_sen_len = 256
        self.epoch = 10000
        self.batch_size = 128
        self.hidden_size=  768



if __name__ == '__main__':

    config = Config()
    seed_val = 123
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    df_in = pd.read_csv(r'data/index/self_index.csv',
                        encoding='utf8')
    # df1 = pd.read_csv(r'data/vec/luvec3.csv', names=[i for i in range(31)])
    # df2 = pd.read_csv(r'data/vec/luvec1.csv', names=[i for i in range(31)])
    dfall = pd.read_csv(r'data/selfdata/self19962.csv')
    label1=[]
    print('begin')
    for i in [1]:
        text1 = np.load("data/selfdata/self_train.npy")
        vec_1 = []

        df1 = pd.read_csv(r'data/vec/selfvec1.csv', header=None)
        for ii in df1.values:
            for j in ii:
                vec_1.append(j)
        df2 = pd.read_csv(r'data/vec/selfvec2.csv', header=None)
        for ii in df2.values:
            for j in ii:
                vec_1.append(j)
        df3 = pd.read_csv(r'data/vec/selfvec3.csv', header=None)
        for ii in df3.values:
            for j in ii:
                vec_1.append(j)



        all_text1=[]
        for ii in text1:
            s=[]
            for j in ii:
                s.append(j)
            s.extend(vec_1)
            all_text1.append(s)
        print(len(all_text1[0]))
        text2 = np.load("data/selfdata/self_test.npy")
        all_text2=[]
        for ii in text2:
            s=[]
            for j in ii:
                s.append(j)
            s.extend(vec_1)
            all_text2.append(s)
        print('ok')
        # text2 = np.load("data/lutoushe/gyhuse_test_bertcls.npy")
        y_name = 'y{}'.format(i)
        label_name = 'label{}'.format(i)


        print('加载数据')
        print(len(all_text1),len(all_text1[0]))
        train_data, train_label = getdata.train_data_loader(config.train_path,all_text1, y_name, i)

        valid_data, valid_label = getdata.test_data_loader(config.valid_path, all_text2,y_name, i)
        test_data, test_label= getdata.test_data_loader(config.test_path,all_text2, y_name, i)

        print('加载完成')
        print(len(train_data),len(train_data[0]))
        total_steps = (len(train_data) / config.batch_size) * config.epoch
        df = pd.read_csv(config.train_path)
        num_labels = len(df[y_name].unique())
        print(num_labels)
        # config=Config()
        model = Model.BertForSequenceClassification(Config(),num_labels)
        optimizer = AdamW(model.parameters(),lr=2e-5,eps=1e-8,weight_decay=1e-3)
        schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        print('开始训练')
        label_ju = [i[768::] for i in train_data]
        train_data=[i[0:768] for i in train_data]
        print(len(label_ju),len(label_ju[0]))

        label_jutest = [i[768::] for i in test_data]
        test_data=[i[0:768] for i in test_data]

        path_for_best=ttve.train(optimizer,schedule,model,i,train_data,train_label,config.batch_size,label_ju,
                                 test_data,test_label,label_jutest)

        # print('开始预测')
        # label, y_pre_label = ttve.predict(path_for_best,model,i,all_text2,test_label,1,lastpre_test)
        # dict={'predict':label}
        # df_tem=pd.DataFrame(dict)
        # df_tem.to_csv(r'pre.csv')
