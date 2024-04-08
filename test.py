import numpy as np
import pandas as pd
text1 = np.load("data/lutoushe/gyhuse_train_bertcls.npy")
vec_1 = []
df = pd.read_csv(r'data/vec/luvec1.csv', header=None)
for i in df.values:
    for j in i:
        vec_1.append(j)
all_text1=[]

for i in text1:
    s=[]
    for j in i:
        s.append(j)
    for k in vec_1:
        s.append(k)
    all_text1.append(s)
print(all_text1[0])





