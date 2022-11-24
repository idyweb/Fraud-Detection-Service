#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle
import pickle


data = pd.read_csv('feat_data.csv')
output_file = 'fraud_model.bin'

#data preprocessing
X = data.drop('target',axis=1)
y = data.target

# Devide them to training, validation and test parts (60:20:20): 
X_train_full_df, X_test_df, y_train_full_df, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2022)
X_train_df, X_val_df, y_train, y_val = train_test_split(X_train_full_df, y_train_full_df, test_size = 0.25, random_state = 2022)

# Vectorize feature matrices in the form of dictionary (with renewed indexes):
dv = DictVectorizer(sparse=False)
#training
X_train_df = X_train_full_df.reset_index(drop=True)
X_train_dict = X_train_full_df.to_dict(orient='records')
X_train = dv.fit_transform(X_train_dict)

#X_val_df = X_val_df.reset_index(drop=True)
#X_val_dict = X_val_df.to_dict(orient='records')
#X_val = dv.fit_transform(X_val_dict)

X_test_df = X_test_df.reset_index(drop=True)
X_test_dict = X_test_df.to_dict(orient='records')
X_test = dv.fit_transform(X_test_dict)

# Renew the index of target variables
y_train = y_train_full_df.reset_index(drop=True)
#y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# validation
model = RandomForestClassifier(n_estimators=200,
                                    max_depth = 3,min_samples_leaf=3,
                                    random_state =2022)
model.fit(X_train,y_train)

y_pred = model.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred)


# # saving the model
with open(output_file, 'wb') as f_out:
    
    pickle.dump((dv,model), f_out)










    

