import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LinearRegression,Ridge,Lasso

data = pd.read_table('./data/abalone.txt',header=None)
data.head()

X = data.iloc[:,:-1]
y = data[8]
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.95,random_state = random.randint(1,100))

lr = LinearRegression()
# 训练
score1 = lr.fit(X_train,y_train).score(X_test,y_test)

ridge = Ridge()
score2 = ridge.fit(X_train,y_train).score(X_test,y_test)

lasso = Lasso()

score3 = lasso.fit(X_train,y_train).score(X_test,y_test)
print(score1,score2,score3)

# 提取第4列作为唯一的特征
X_train5 = X[5]
# 把X_train5由1维变成2维
# 训练时，X必须是二维矩阵
X_train5 = X_train5.values.reshape(-1,1)

# 得到画图中x轴和y轴的范围
x_min,x_max = X[5].min(),X[5].max()
y_min,y_max = y.min()-1,y.max()+1


#步长
h = 0.001
#生成曲线的X坐标
X = np.arange(x_min,x_max,h).reshape(-1,1)

reg2 = LinearRegression()
# 针对第6列特征再训练一次
reg2.fit(X_train5,y)

reg2.score(X_train5,y)

#对X轴坐标进行预测
y_ = reg2.predict(X_train5)


#进行绘制图形
# 设置图片大小
plt.figure(figsize=(12,8))

# 画出真实点
plt.scatter(X_train5,y)
# 预测结果，年龄线
plt.plot(X_train5,y_)

