# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 22:33:19 2020

@author: depri
"""
##test code
import pandas as pd
import datetime
from datetime import timedelta
import math
import numpy as np
from sklearn.model_selection import  train_test_split
import joblib 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score,cross_validate

file_path_test_data = input('Please enter full path of Test data file: ')
test_data_matrix = pd.read_csv(file_path_test_data)


test_data_matrix.columns = range(0,24)

## FEATURE1: TMAX-TM
tmax_tm = (test_data_matrix.idxmax(axis=1)*5)

##FEATURE2: CGM max - CGM min
cgm_diff = test_data_matrix.max(axis=1) - test_data_matrix.min(axis=1)

##FEATURE 3 AND 4 Velocity max and time at which velocity is max
meal_data_v = test_data_matrix.diff(axis=1)
v_max = meal_data_v.max(axis=1)
t_vmax = meal_data_v.idxmax(axis=1)*5

meal_data_v2=meal_data_v.diff(axis=1)
v2_max=meal_data_v2.max(axis=1)
t_v2max=meal_data_v2.idxmax(axis=1)*5

##FEATURE 4: powers
x_array = test_data_matrix.to_numpy()
f1 = []
f2= []
for each in x_array:
    ps = 2*np.abs(np.fft.fft(each))
    ls=[]
    for p1 in ps:
        ls.append(round(p1,2))
    ls=set(ls)
    ls=list(ls)
    ls.sort()
    w1 = ls[-2]
    w2 = ls[-3]
    f1.append(w1)
    f2.append(w2)

dff1 = pd.DataFrame(f1)
dff2  = pd.DataFrame(f2)    

##FEATURE 5: Windowed mean and standard deviation
df_len = len(test_data_matrix)
m1=[]
m2=[]
m3=[]
d1=[]
d2=[]
d3=[]
for each in range(0,df_len):
    df_test=test_data_matrix.iloc[each]
    m1.append(sum(df_test[10:15])/5)
    m2.append(sum(df_test[15:20])/5)
    m3.append(sum(df_test[20:25])/5)
    d1.append(df_test[10:15].std())
    d2.append(df_test[15:20].std())
    d3.append(df_test[20:25].std())
        
    dfm1=pd.DataFrame(m1)
    dfm2=pd.DataFrame(m2)
    dfm3=pd.DataFrame(m3)

dfd1=pd.DataFrame(d1)
dfd2=pd.DataFrame(d2)
dfd3=pd.DataFrame(d3)    


##concatenating the features:
test_feature_matrix = pd.concat([tmax_tm,cgm_diff,v_max,t_vmax,v2_max,t_v2max],axis=1,ignore_index=True)
test_feature_matrix.reset_index(inplace=True)  
test_feature_matrix = pd.concat([test_feature_matrix,dff1,dff2,dfm1,dfm2,dfm3,dfd1,dfd2,dfd3],axis=1)  
test_feature_matrix.drop(columns='index',inplace=True)

model_from_job = joblib.load('DMpt2.pkl')

Y_pred = model_from_job.predict(test_feature_matrix)

pd.DataFrame(Y_pred).to_csv('Result.csv',index=False)







