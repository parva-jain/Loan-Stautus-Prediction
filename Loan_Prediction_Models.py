# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:27:23 2020

@author: parva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_data = pd.read_csv('Processed Data/training_data.csv')
test_data = pd.read_csv('Processed Data/testing_data.csv')
train_label = pd.read_csv('Processed Data/training_targets.csv')
raw_test_data = pd.read_csv('Data/test_lAUu6dG.csv')

seed = 20
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
# from sklearn.ensemble.voting_classifier import VotingClassifier
# from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
# from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
# from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture


et = ExtraTreeClassifier(random_state = seed)
dt = DecisionTreeClassifier(random_state = seed)
nn = MLPClassifier(activation='logistic', early_stopping=True)
rnc = RadiusNeighborsClassifier()
knc = KNeighborsClassifier()
sgd = SGDClassifier(n_jobs = -1,random_state = seed, early_stopping=True)
ridge = RidgeClassifier(random_state = seed)
ridgecv = RidgeClassifierCV()
pa = PassiveAggressiveClassifier(n_jobs = -1,random_state = seed, early_stopping=True)
gboost = GradientBoostingClassifier(random_state = seed)
etc = ExtraTreesClassifier(n_jobs = -1, random_state = seed)
rf = RandomForestClassifier(n_jobs = -1, random_state = seed)
bnb = BernoulliNB()
gnb = GaussianNB()
linsvc = LinearSVC(random_state = seed)
lr = LogisticRegression()
lrcv = LogisticRegressionCV()
nusvc = NuSVC(random_state = seed)
per = Perceptron(n_jobs = -1, random_state = seed)
svc = SVC(random_state = seed)
gm = GaussianMixture()


def train_r2(model):
    model.fit(train_data, train_label)
    return model.score(train_data, train_label)


models = [et, dt, nn, rnc, knc, sgd, ridge, ridgecv, pa, gboost, etc, rf,
          bnb, gnb, linsvc, lr, lrcv, nusvc, per, svc, gm ]
training_score = []
for model in models:
    training_score.append(train_r2(model))
    
train_score = pd.DataFrame(data = training_score, columns = ['Training_R2'])
train_score.index = ['ET', 'DT', 'NN', 'RNC', 'KNC', 'SGD', 'RIDGE', 'RidgeCV',
                     'PA', 'GBOOST', 'ETC', 'RF', 'BNB', 'GNB', 'LinSVC', 'LR',
                     'LRCV', 'NuSVC', 'PER', 'SVC', 'GM']
train_score = (train_score*100).round(4)

def train_test_split_score(model):
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size = 0.3, random_state = seed)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    mse = mean_squared_error(prediction, Y_test)
    rmse = np.sqrt(mse)
    error = [rmse]
    return error

models = [et, dt, ridgecv, gboost, etc, rf, linsvc, lr, lrcv, nusvc]
train_test_split_rmse = []
for model in models:
    train_test_split_rmse.append(train_test_split_score(model)[0])
    
train_test_score = pd.DataFrame(data = train_test_split_rmse, columns = ['Train_Test_RMSE'])
train_test_score.index = ['ET', 'DT', 'RidgeCV', 'Gboost', 'ETC', 'RF',
                          'LinSVC', 'LR', 'LRCV', 'NuSVC']
train_test_score = train_test_score.round(8)

train_test_score_sorted = train_test_score.sort_values(by = ['Train_Test_RMSE'])

def cross_validate(model):
    from sklearn.model_selection import cross_val_score
    neg_x_val_score = cross_val_score(model, train_data, train_label, cv = 10, n_jobs = -1, scoring = 'neg_mean_squared_error')
    x_val_score = np.round(np.sqrt(-1*neg_x_val_score), 5)
    return x_val_score.mean()

models = [et, dt, ridgecv, gboost, etc, rf, linsvc, lr, lrcv, nusvc]
cross_val_scores = []
for model in models:
    cross_val_scores.append(cross_validate(model))

x_val_score = pd.DataFrame(data = cross_val_scores, columns = ['Cross Validation Scores (RMSE)'])
x_val_score.index = ['ET', 'DT', 'RidgeCV', 'Gboost', 'ETC', 'RF',
                          'LinSVC', 'LR', 'LRCV', 'NuSVC']
x_val_score = x_val_score.round(5)
x = x_val_score.index
y = x_val_score['Cross Validation Scores (RMSE)']
sns.scatterplot(x,y)

def predict_with_models(model):
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    my_map = {1:'Y',0:'N'}
    y_pred = np.vectorize(my_map.get)(y_pred)
    submission = pd.DataFrame()
    submission['Loan_ID']= raw_test_data.Loan_ID
    submission['Loan_Status'] = y_pred
    return submission

predict_with_models(lrcv).to_csv('Submissions\LogisticRegressionCV.csv', index = False)
predict_with_models(rf).to_csv('Submissions\RandomForest.csv', index = False)
predict_with_models(ridgecv).to_csv('Submissions\RidgeCV.csv', index = False)


