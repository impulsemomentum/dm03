import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth
from sklearn import datasets  
from itertools import cycle
#X = iris.data[:, 2:4] ##表示我们只取特征空间中的后两个维度
#print(X.shape)


os.chdir("./west_nile/working/")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')
weather = pd.read_csv('../input/weather.csv')

# Get labels
labels = train.WnvPresent.values

# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

# replace some missing values and T with -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
weather = weather.replace(' T', -1)
weather = weather.replace('  T', -1)

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]

train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
#test['month'] = test.Date.apply(create_month)
#test['day'] = test.Date.apply(create_day)

# Add integer latitude/longitude columns
train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
#test['Lat_int'] = test.Latitude.apply(int)
#test['Long_int'] = test.Longitude.apply(int)

# Merge with weather data
train = train.merge(weather, on='Date')
#test = test.merge(weather, on='Date')
train = train.drop(['Date'], axis = 1)
#test = test.drop(['Date'], axis = 1)

# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
#test['Species'] = lbl.transform(test['Species'].values)

lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
#test['Street'] = lbl.transform(test['Street'].values)

lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
#test['Trap'] = lbl.transform(test['Trap'].values)

# drop columns with -1s
train = train.ix[:,(train != -1).any(axis=0)]
#test = test.ix[:,(test != -1).any(axis=0)]
train1 = train
t =0
for item in train1.columns:
    a = train1[item]
    if type(a[1]) == str:
        train1 = train1.drop(item,axis = 1)
        t=t+1

#for item_x  in train1.columns:
i = 1
for item_y in train1.columns:
    ax = plt.subplot(len(train1.columns), len(train1.columns), i)
    ax.set_title(item_x+'--'+item_y)
    ax.scatter(train1[item_x], train1[item_y])
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    plt.show()


i = 1
for itema in train1.columns:
    if itema != 'WnvPresent':
        X = train1.loc[:,[itema,'WnvPresent']]
        X = X.dropna(axis = 0)
        estimator = KMeans(n_clusters=2)
        estimator.fit(X)
        label_pred = estimator.labels_
        x0 = X[label_pred == 0]
        x1 = X[label_pred == 1]
        ax = plt.subplot(5, 4, i)
        ax.scatter(x0[x0.columns[0]], x0[x0.columns[1]], c = "red", marker='o', label='label0')  
        ax.scatter(x1[x1.columns[0]], x1[x1.columns[1]], c = "green", marker='*', label='label1')  
        ax.set_title('WnvPreset'+'--'+itema)
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
plt.show()


i = 1
for itema in train1.columns:
    if itema != 'WnvPresent':
        X = train1.loc[:,[itema,'WnvPresent']]
        X = X.dropna(axis = 0)
        bandwidth = estimate_bandwidth(X)
        estimator = MeanShift(bandwidth=bandwidth,bin_seeding=True)
        estimator.fit(X)
        label_pred = estimator.labels_
        cluster_centers = estimator.cluster_centers_
        labels_unique = np.unique(label_pred)
        n_clusters_=len(labels_unique)
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        ax = plt.subplot(5, 4, i)
        ax.set_title('WnvPreset'+'--'+itema)
        for k, col in zip(range(n_clusters_), colors):
            #my_members = label_pred == k
            cluster_center = cluster_centers[k]
            x = X[label_pred == k]
            ax.plot(x[x.columns[0]], x[x.columns[1]], col + '.')
            ax.plot(cluster_center[0], cluster_center[1], '^', markerfacecolor=col, markeredgecolor='k')
        i += 1
plt.show()

for i in range [0,len(train.columns)]:
    #绘制数据分布图
    plt.scatter(X[:, train.columns[i]], X[:, train.columns[i+1]], c = "red", marker='o', label='see')  
    plt.xlabel(train.columns[i])  
    plt.ylabel(train.columns[i+1])  
    plt.legend(loc=2)  
    plt.show()  
    i = i+ 2


X = train.iloc[2:4]
estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')  
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')  
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')  
plt.xlabel('petal length')  
plt.ylabel('petal width')  
plt.legend(loc=2)  
plt.show()  