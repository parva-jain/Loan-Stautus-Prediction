# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:36:18 2020

@author: parva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_train_data = pd.read_csv('Data/train_ctrUa4K.csv')
raw_test_data = pd.read_csv('Data/test_lAUu6dG.csv')

des_train = raw_train_data.describe()
des_test = raw_test_data.describe()

raw_train_data.info()
raw_test_data.info()

raw_train_data.nunique()

merged = pd.concat([raw_train_data,raw_test_data],axis = 0, sort = True)
merged.dtypes.value_counts()


num_merged = merged.select_dtypes(include = ['int64', 'float64'])
num_merged.hist(bins=30, figsize=(15, 10))

merged.loc[:,['Credit_History','Loan_Amount_Term']] = merged.loc[:,['Credit_History','Loan_Amount_Term']].astype('object')
merged.dtypes.value_counts()

train_data = merged.iloc[:614, :].drop(columns = ['Loan_ID'], axis = 1)
test_data = merged.iloc[614:, :].drop(columns = ['Loan_ID', 'Loan_Status'], axis = 1)

sns.distplot(train_data['ApplicantIncome'])
sns.distplot(train_data['CoapplicantIncome'])
sns.distplot(train_data['LoanAmount'])

train_data.drop(train_data[train_data.ApplicantIncome>30000].index, inplace = True)
train_data.drop(train_data[train_data.CoapplicantIncome>15000].index, inplace = True)

y_train = train_data.Loan_Status
train_data.drop('Loan_Status', axis = 1, inplace = True)
df_merged = pd.concat([train_data, test_data], axis = 0)

missing_columns = df_merged.columns[df_merged.isnull().any()].values

df_merged.info()

to_impute_by_none = df_merged.loc[:, ['Gender','Married','Self_Employed']]
for i in to_impute_by_none.columns:
    df_merged[i].fillna('None', inplace = True)
    
to_impute_by_rand = df_merged.loc[:, ['Dependents','Loan_Amount_Term','Credit_History']]
for i in to_impute_by_rand.columns:
    df_merged[i].fillna(-1, inplace = True)
    
to_impute_by_median = df_merged.loc[:, ['LoanAmount']]
for i in to_impute_by_median.columns:
    df_merged[i].fillna(df_merged[i].median(), inplace = True)


df_merged.columns[df_merged.isna().any()].values

skew_merged = pd.DataFrame(data = df_merged.select_dtypes(include = ['int64', 'float64']).skew(), columns = ['Skewness'])
skew_merged_sorted = skew_merged.sort_values(ascending = False, by = 'Skewness')

sns.barplot(skew_merged_sorted.index, skew_merged_sorted.Skewness)

df_merged_num = df_merged[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
df_merged_num.skew()[df_merged_num.skew()>0.75].index


df_merged_skewed = np.log1p(df_merged_num[df_merged_num.skew()[df_merged_num.skew()>0.75].index])
df_merged_normal = df_merged_num[df_merged_num.skew()[df_merged_num.skew()< 0.75].index] # Normal variables
df_merged_num_all = pd.concat([df_merged_skewed, df_merged_normal], axis = 1)


df_merged_num.update(df_merged_num_all)
pd.DataFrame(data = df_merged_num.skew(), columns = ['Skewness'])

from sklearn.preprocessing import RobustScaler
robust_scl = RobustScaler()
robust_scl.fit(df_merged_num)
df_merged_num_scaled = robust_scl.transform(df_merged_num)
df_merged_num_scaled = pd.DataFrame(data = df_merged_num_scaled, columns = df_merged_num.columns, index = df_merged_num.index)
pd.DataFrame(data = df_merged_num_scaled.skew(), columns = ['Skewness'])


df_merged.Credit_History.replace(to_replace = [1, 0, -1], value = ['Yes', 'No', 'Missing'], inplace = True)

df_merged_cat = df_merged.select_dtypes(include = ['object']).astype('category')
df_merged_one_hot = pd.get_dummies(df_merged_cat)
df_merged_one_hot = df_merged_one_hot.astype('int64')

df_merged_processed = pd.concat([df_merged_num_scaled, df_merged_one_hot,df_merged['Loan_Amount_Term']], axis = 1)

df_train_final = df_merged_processed.iloc[:603]
df_test_final = df_merged_processed.iloc[603:]

y_train = y_train.map({'Y':1,'N':0})

df_train_final.to_csv(r'C:\Users\parva\Desktop\Analytics Vidhya\Loan Prediction\Processed Data\training_data.csv',index = False)
df_test_final.to_csv(r'C:\Users\parva\Desktop\Analytics Vidhya\Loan Prediction\Processed Data\testing_data.csv',index = False)
y_train.to_csv(r'C:\Users\parva\Desktop\Analytics Vidhya\Loan Prediction\Processed Data\training_targets.csv',index = False)
