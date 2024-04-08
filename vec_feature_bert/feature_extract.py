import pandas as pd
import json,os
from tqdm import tqdm
from utils.unified_file_format import file_to_feature
#
# path = 'E:\\work\\中国移动\\data904\\all_file904\\'
# df = pd.read_csv(r'E:\work\中国移动\data904\904_修改后.csv')
#
# with open(r'E:\work\中国移动\data904\feature.json', 'w', encoding='utf-8') as f:
#     for i in range(len(df)):
#         filepath = path + df.iloc['filename']
#         feature_dic = file_to_feature(filepath)
#         json.dump(feature_dic, f, ensure_ascii=False)
# path = 'C:\\Users\\GYHfresh\\Desktop\\all-data-1203-85305\\'
path = 'D:\\!gyh\\all-data-1203-84750\\'
name = os.listdir(path)
dict={'newname':name}

df = pd.DataFrame(dict)
# df=df.sort_values('newname')

for i in tqdm(range(len(df))):
    newname = []
    title = []
    content = []
    filepath = path + df.iloc[i,0]
    feature_dic = file_to_feature(filepath)
    if 'heading' in feature_dic.keys():
        title.append(feature_dic['name_chineseall']+feature_dic['heading'])

    else:
        title.append(feature_dic['name_chineseall'])

    # if 'allcsv_chinese' in feature_dic.keys():
    #     content.append(feature_dic['allcsv_chinese'])
    # else:
    #     content.append(' ')
    newname.append(df.iloc[i,0])

    dict={
        'newname':newname,
        'title':title,
        # 'content':content
    }

    df1=pd.DataFrame(dict)
    df1.to_csv(r'gyh_feature_paper.csv',mode='a',index=None,header=None)
    del  newname
    del title