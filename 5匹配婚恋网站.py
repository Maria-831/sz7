import sklearn
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import SGDClassifier
# ----------fb.model.x.xxx
from sklearn.ensemble  import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from  sklearn import  isotonic


#梯度树
def mx_GradientBoostingClassifier(train_x, train_y):
    mx = GradientBoostingClassifier()
    mx.fit(train_x, train_y)
    return mx

#提升算法 AdaBoost
def mx_AdaBoostClassifier(train_x, train_y):
    mx = AdaBoostClassifier()
    mx.fit(train_x, train_y)
    return mx

#极限组合森林
def mx_ExtraTreesClassifier(train_x, train_y):
    mx = ExtraTreesClassifier()
    mx.fit(train_x, train_y)
    return mx

#随机数随机森林
def mx_RandomForestClassifier(train_x, train_y):
    mx = RandomForestClassifier()
    mx.fit(train_x, train_y)
    return mx

#元估计器
def mx_BaggingClassifier(train_x, train_y):
    mx = BaggingClassifier()
    mx.fit(train_x, train_y)
    return mx

from  sklearn import  cross_decomposition
#交叉分解
def mx_PLSRegression(train_x, train_y):
    mx = cross_decomposition.PLSRegression()
    mx.fit(train_x, train_y)
    return mx
#交叉分解
def mx_PLSCanonical(train_x, train_y):
    mx = sklearn.cross_decomposition.PLSCanonical()
    mx.fit(train_x, train_y)
    return mx


#BernoulliNB
def mx_BernoulliNB(train_x, train_y):
    mx = sklearn.naive_bayes.BernoulliNB()
    mx.fit(train_x, train_y)
    return mx

#高斯贝叶斯
def mx_GaussianNB(train_x, train_y):
    mx = sklearn.naive_bayes.GaussianNB()
    mx.fit(train_x, train_y)
    return mx



#随机梯度下降
def mx_SGDClassifier(train_x, train_y):
    mx = SGDClassifier()
    mx.fit(train_x, train_y)
    return mx

#高斯过程
def mx_GaussianProcessRegressor(train_x, train_y):
    mx = sklearn.gaussian_process.GaussianProcessRegressor()
    mx.fit(train_x, train_y)
    return mx


# 线性回归算法，最小二乘法，函数名，LinearRegression
def mx_line(train_x, train_y):
    mx = LinearRegression()
    mx.fit(train_x, train_y)
    # print('\nlinreg.intercept_')
    # print (mx.intercept_);print (mx.coef_)
    # linreg::model
    #
    return mx

#岭回归
def mx_Ridge(train_x, train_y):
    mx = linear_model.Ridge ()
    mx.fit(train_x, train_y)
    return mx

#拉格朗日回归
def mx_Lasso(train_x, train_y):
    mx = linear_model.Lasso()
    mx.fit(train_x, train_y)
    return mx




#多任务拉格朗日回归
def mx_LassoLars(train_x, train_y):
    mx = linear_model.LassoLars()
    mx.fit(train_x, train_y)
    return mx

#主动决策
def mx_ARDRegression(train_x, train_y):
    mx = linear_model.ARDRegression()
    mx.fit(train_x, train_y)
    return mx

#稳健回归（Robustness regression） #随机抽样一致性算法
def mx_RANSACRegressor(train_x, train_y):
    mx = linear_model.RANSACRegressor()
    mx.fit(train_x, train_y)
    return mx
#稳健回归（Robustness regression） #广义中值估计
def mx_TheilSenRegressor(train_x, train_y):
    mx = linear_model.TheilSenRegressor()
    mx.fit(train_x, train_y)
    return mx


#贝叶斯回归
def mx_bayeslinear(train_x, train_y):
    mx = linear_model.BayesianRidge()
    mx.fit(train_x, train_y)
    return mx


# 逻辑回归算法，函数名，LogisticRegression
def mx_log(train_x, train_y):
    mx = LogisticRegression(penalty='l2')
    mx.fit(train_x, train_y)
    return mx



# SGDR回归算法，
def mx_SGDRegressor(train_x, train_y):
    mx =  linear_model.SGDRegressor()
    mx.fit(train_x, train_y)
    return mx

#PassiveAggressiveRegressor被动攻击算法
def mx_PassiveAggressiveRegressor(train_x, train_y):
    mx =  linear_model.PassiveAggressiveRegressor()
    mx.fit(train_x, train_y)
    return mx

#感知器Perceptron
def mx_Perceptron(train_x, train_y):
    mx =  linear_model.Perceptron()
    mx.fit(train_x, train_y)
    return mx
#最小角回归¶
def mx_Lars(train_x, train_y):
    mx =  linear_model.Lars()
    mx.fit(train_x, train_y)
    return mx
#ElasticNetCV 弹性网络¶
def mx_ElasticNetCV(train_x, train_y):
    mx =  linear_model.ElasticNetCV()
    mx.fit(train_x, train_y)
    return mx




# 多项式朴素贝叶斯算法，Multinomial Naive Bayes，函数名，multinomialnb
def mx_bayes(train_x, train_y):
    mx = MultinomialNB(alpha=0.01)
    mx.fit(train_x, train_y)
    return mx


# KNN近邻算法，函数名，KNeighborsClassifier
def mx_knn(train_x, train_y):
    mx = KNeighborsClassifier()
    mx.fit(train_x, train_y)
    return mx


# 随机森林算法， Random Forest Classifier, 函数名，RandomForestClassifier
def mx_forest(train_x, train_y):
    mx = RandomForestClassifier(n_estimators=8)
    mx.fit(train_x, train_y)
    return mx


# 决策树算法，函数名，tree.DecisionTreeClassifier()
def mx_dtree(train_x, train_y):
    mx = tree.DecisionTreeClassifier()
    mx.fit(train_x, train_y)
    return mx


# GBDT迭代决策树算法，Gradient Boosting Decision Tree，
# 又叫 MART(Multiple Additive Regression Tree)，函数名，GradientBoostingClassifier
def mx_GBDT(train_x, train_y):
    mx = GradientBoostingClassifier(n_estimators=200)
    mx.fit(train_x, train_y)
    return mx


# SVM向量机算法，函数名，SVC
def mx_svm(train_x, train_y):
    mx = SVC(kernel='rbf', probability=True)
    mx.fit(train_x, train_y)
    return mx


# SVM- cross向量机交叉算法，函数名，SVC
def mx_svm_cross(train_x, train_y):
    mx = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(mx, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in best_parameters.items():
    #    print( para, val)
    mx = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    mx.fit(train_x, train_y)
    return mx


# ----神经网络算法


# MLP神经网络算法
def mx_MLP(train_x, train_y):
    # mx = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    mx = MLPClassifier()
    mx.fit(train_x, train_y)
    return mx


# MLP神经网络回归算法
def mx_MLP_reg(train_x, train_y):
    # mx = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    mx = MLPRegressor()
    mx.fit(train_x, train_y)
    return mx

mxfunSgn = {

              'GradientBoostingClassifier': mx_GradientBoostingClassifier,
              'AdaBoostClassifier': mx_AdaBoostClassifier,
              'ExtraTreesClassifier':mx_ExtraTreesClassifier ,
              'RandomForestClassifier': mx_RandomForestClassifier,
              'BaggingClassifier':mx_BaggingClassifier ,
              'PLSRegression': mx_PLSRegression,
              'PLSCanonical':mx_PLSCanonical ,
              'BernoulliNB':  mx_BernoulliNB,
              'GaussianNB': mx_GaussianNB,
              'SGDClassifier':mx_SGDClassifier ,
              'GaussianProcessRegressor': mx_GaussianProcessRegressor,
              'Ridge': mx_Ridge,
              'Lasso':mx_Lasso ,

              'LassoLars': mx_LassoLars,
              'ARDRegression': mx_ARDRegression,
              'RANSACRegressor': mx_RANSACRegressor,
              'TheilSenRegressor': mx_TheilSenRegressor,
              'bayeslinear':mx_bayeslinear,
              'PassiveAggressiveRegressor': mx_PassiveAggressiveRegressor,
              'SGDRegressor': mx_SGDRegressor,
              'Perceptron': mx_Perceptron,
              'Lars': mx_Lars,
              'ElasticNetCV': mx_ElasticNetCV,

            'line': mx_line,
            'log': mx_log,
            'bayes': mx_bayes,
            'knn': mx_knn,
            'forest': mx_forest,
            'dtree': mx_dtree,
            'gbdt': mx_GBDT,
            'svm': mx_svm,
            'svmcr': mx_svm_cross,
            'mlp': mx_MLP,
            'mlpreg': mx_MLP_reg
            }



import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

path=r".\data\datingTestSet.txt"
data = pd.read_table(path,header=None)
#得到训练数据和测试数据
X = data.iloc[:,:-1]
y = data[3]
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.88)

scoredict={}
for  key  in mxfunSgn :
    try:
        knc = mxfunSgn[key](X_train,y_train)
        score = knc.score(X_test, y_test)
        scoredict[key]=score
    except:
        score=0
    print(key ,  score )

print("max  :",sorted(scoredict,key=lambda x:scoredict[x])[-1],
      scoredict[sorted(scoredict,key=lambda x:scoredict[x])[-1]])
print("next max   :",sorted(scoredict,key=lambda x:scoredict[x])[-2],
      scoredict[sorted(scoredict, key=lambda x: scoredict[x])[-2]])
print("thirt max   :",sorted(scoredict,key=lambda x:scoredict[x])[-3],
      scoredict[sorted(scoredict, key=lambda x: scoredict[x])[-3]])