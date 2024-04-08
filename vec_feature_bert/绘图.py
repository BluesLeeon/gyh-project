import seaborn as sns
import pandas as pd
def ShowGRAHeatMap(DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    colormap = plt.cm.Greys

    plt.figure(figsize=(14,14))
    plt.title('Correlation of Features', y=1.05, size=15)
    sns.heatmap(DataFrame.astype(float),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)


    sns.set_style('whitegrid')

    plt.show()
df = pd.read_csv(r'C:\Users\GYHfresh\Desktop\add_vec\矩阵1.csv', names=[i for i in range(152)])

def get(a1,b1,a2,b2):
    s1=[i for i in range(a1,b1)]
    s2 = [i for i in range(a2, b2)]
    s=s1+s2
    return s


# ShowGRAHeatMap(df.iloc[0:76,0:76])#1 2
# ShowGRAHeatMap(df.iloc[get(0,34,76,82),get(0,34,76,82)])#1 3
# ShowGRAHeatMap(df.iloc[get(0,34,82,152),get(0,34,82,152)])#1 4
ShowGRAHeatMap(df.iloc[34:82,34:82])#23
# ShowGRAHeatMap(df.iloc[0:77,0:77])