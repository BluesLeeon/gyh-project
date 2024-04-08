import pandas as pd
import ttve
import numpy as np

df_in = pd.read_csv(r'data/index/lutoushe_index.csv',
                    encoding='gbk')
df1 = pd.read_csv(r'data/vec/luvec3.csv', names=[i for i in range(31)])
df2 = pd.read_csv(r'data/vec/luvec1.csv', names=[i for i in range(31)])
dfall = pd.read_csv(r'data/lutoushe/gyhuse.csv')


def train_data_loader(path, text,y_name, i):
    df = pd.read_csv(path)
    label = df[y_name]
    # text = df['text'].values
    vec_level0 = []
    for label in df_in['num'].values[0:11]: vec_level0.append(1)

    # if i == 1:
    label = df['y1']
    vec_levels = []
    for name in label:
        vec_levels.append(vec_level0)
    lastpre = vec_levels
    label = df[y_name]-11
    # elif i == 2:
    #     label = df['y1']
    #     vec_levels = []
    #     for name in label:
    #
    #         vec_level1 = df1.iloc[name, 11:31].values
    #         vec_level2 = df2.iloc[name, 11:31].values
    #         vec_level = []
    #         for i in vec_level1:
    #             vec_level.append(i)
    #         for i in vec_level2:
    #             vec_level.append(i)
    #         vec_levels.append(vec_level)

    # lastpre = vec_levels
    # label = df[y_name] - 11

    return np.array(text), np.array(label.values)


def test_data_loader(path, text,y_name, i):
    df = pd.read_csv(path)
    # if i == 1:

    label = df[y_name]
    vec_level0 = []
    for label in df_in['num'].values[0:11]: vec_level0.append(0)
    label = df['y1']
    vec_levels = []
    for name in label: vec_levels.append(vec_level0)
    lastpre = vec_levels
    label = df[y_name]-11
    # elif i == 2:
    #     vec_levels = []
    #     import main
    #     df_tem=pd.read_csv(r'pre.csv')
    #     label1=df_tem['predict'].values.tolist()
    #     # label, y_pre_label = ttve.predict(path_for_best,model,i,text2,test_label,1,lastpre_test)
    #     for name in label1:
    #         name=int(name)
    #         vec_level1 = df1.iloc[name, 11:31].values
    #         vec_level2 = df2.iloc[name, 11:31].values
    #         vec_level = []
    #         for i in vec_level1:
    #             vec_level.append(i)
    #         for i in vec_level2:
    #             vec_level.append(i)
    #         vec_levels.append(vec_level)
    #
    #     lastpre = vec_levels
    #     label = df[y_name] - 11

    return np.array(text), np.array(label.values)