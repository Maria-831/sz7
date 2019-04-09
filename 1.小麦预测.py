import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_table('./data/seeds.csv',header = None)
data.head()

#处理数据获取训练数据train以及测试数据test

X = data.iloc[:,:-1]
y = data[7]

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.9,random_state = 8)

#创建KNN近邻算法
knc = KNeighborsClassifier()

knc.fit(X_train,y_train)

score = knc.score(X_test,y_test)
print(score)
#准确率很高，此时只要输入小麦的数据就可以预测小麦的种类了
knc.predict([X_test.iloc[0]])