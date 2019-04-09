
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#支持向量机
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#KNN近邻算法
from sklearn.neighbors import KNeighborsClassifier
#线性回归与逻辑斯蒂回归（其实他是分类）
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
#决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
#贝叶斯
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#数据分组
from sklearn.model_selection import train_test_split

data=pd.read_csv(r'.\data\student-data.csv')
#将字符串转化为数字
for i in data[['school', 'sex','address', 'famsize', 'Pstatus',
       'Mjob', 'Fjob', 'reason', 'guardian',
        'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'passed']]:
    stu=data[i].unique() #不重复
    def transform(i): #每个字符串匹配一个数字
        return np.argwhere(stu==i)[0,0]
    data[i]=data[i].map(transform) #所有特征，数字化
print(data.shape)
#将字符串转化为数字
for i in data[['school', 'sex','address', 'famsize', 'Pstatus',
       'Mjob', 'Fjob', 'reason', 'guardian',
        'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'passed']]:
    stu=data[i].unique()
    def transform(i):
        return np.argwhere(stu==i)[0,0]
    data[i]=data[i].map(transform)
print(data.shape)
#归一操作
for i in data.columns:
    sum=data[i].sum()
    data[i]=data[i]/sum
print(data.shape)


#将data中的部分列复制提取为X
X=data[[ 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health','passed']].copy()
#去掉偏离波动大的值
X=X.drop(X[(np.abs(X-X.mean()) > (2*X.std())).any(axis=1)].index)
print(X.shape)



X=X[[ 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health']].copy()
y=data['passed'].copy()

y=y.ravel()
#生成训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=63)
print(X.shape,y.shape)
score2=KNeighborsClassifier(30).fit(X_train,y_train).score(X_test,y_test)
print(score2)