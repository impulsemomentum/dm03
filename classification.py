import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, preprocessing
import os
import pandas as pd
#os.chdir("./west_nile/working/")
# Load dataset 
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

train_orig = train
#train_orig = train_orig.merge(weather, on='Date')


i = 1
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
for item_x  in train_orig.columns:
    ax = plt.subplot(len(train_orig.columns), len(train_orig.columns), i)
    ax.set_title(item_x+'--'+item_y)
    items = train_orig[item_x].unique()
    area = train_orig[item_x].value_count()
    ax.scatter(train_orig[item_x], train_orig['WnvPresent'], s = area,cmap=cm_bright, alpha=0.6,  edgecolors='k')
    ax.set_xlim(train_orig[item_x].min(), train_orig[item_x].max())
    ax.set_ylim(train_orig[item_y].min(), train_orig[item_y].max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
plt.show()
# i = 1
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
# for item_x  in train_orig.columns:
#     for item_y in train_orig.columns:
#         ax = plt.subplot(len(train_orig.columns), len(train_orig.columns), i)
#         ax.set_title(item_x+'--'+item_y)
#         ax.scatter(train_orig[item_x], train_orig[item_y], cmap=cm_bright, alpha=0.6,  edgecolors='k')
#         ax.set_xlim(train_orig[item_x].min(), train_orig[item_x].max())
#         ax.set_ylim(train_orig[item_y].min(), train_orig[item_y].max())
#         ax.set_xticks(())
#         ax.set_yticks(())
#         i += 1
# plt.show()
# drop address columns
train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis = 1)
#test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)


test_data = train[0:10000:2]
train_data = train[1:10001:2]
test_labels = labels[0:10000:2]
train_labels = labels[1:10001:2]
# step size in the mesh
h = 0.02 

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
    "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
    "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

clf0 = classifiers[0]
clf0.fit(train_data, train_labels)
score = clf0.score(test_data, test_labels)
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#     random_state=1, n_clusters_per_class=1)

# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)

# datasets = [make_moons(noise=0.3, random_state=0),
#     make_circles(noise=0.2, factor=0.5, random_state=1),
#     linearly_separable
#     ]

#figure = plt.figure(figsize=(27, 9))
#i = 1


#iterate over datasets
#for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    #X, y = ds
    #X = StandardScaler().fit_transform(X)
    #X_train, X_test, y_train, y_test = \
    #    train_test_split(X, y, test_size=.4, random_state=42)
    #x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    #y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #np.arange(y_min, y_max, h))
    # just plot the dataset first
    #cm = plt.cm.RdBu
    # cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # #if ds_cnt == 0:
    # ax.set_title("Input data")
    # # Plot the training points
    # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
    #     edgecolors='k')
    # # and testing points
    # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
    #     edgecolors='k')
    # ax.set_xlim(xx.min(), xx.max())
    # ax.set_ylim(yy.min(), yy.max())
    # ax.set_xticks(())
    # ax.set_yticks(())
    #i += 1
    # iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(train, train_labels)
    score = clf.score(test, test_labels)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        #if hasattr(clf, "decision_function"):
        #    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        #else:
        #    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        # Put the result into a color plot
        #Z = Z.reshape(xx.shape)
        #ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # Plot also the training points
        # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
        #     edgecolors='k')
        # # and testing points
        # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
        #     edgecolors='k', alpha=0.6)
        # ax.set_xlim(xx.min(), xx.max())
        # ax.set_ylim(yy.min(), yy.max())
        # ax.set_xticks(())
        # ax.set_yticks(())
        # if ds_cnt == 0:
        #     ax.set_title(name)
        # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #     size=15, horizontalalignment='right')
        # i += 1
        predictions = clf.predict_proba(test_data)[:,1]
        x = np.arange(0,5000,1)
        plt.plot(x,test_labels[0:5000],'rs',x,predictions[0:5000],'b^')
plt.tight_layout()
plt.show()