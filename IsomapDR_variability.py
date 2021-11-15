# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:51:29 2019

@author: simran
"""
#for calculating variability of components
import numpy as np
import pandas as pd
from sklearn.svm import SVC
NoComp = 11
data= pd.read_excel('Datasets/Gear15.xlsx')
# Dataset is now stored in a Pandas Dataframe
data = data.dropna(axis = 1, how ='all') 

#removing columns havin more than 90% zero
data=data.drop(columns=data.columns[((data==0).mean()>0.90)],axis=1)
#print(data.info)
#saving lables in y
X_data= data.iloc[:,:-1].values
y_data=data.iloc[:,-1].values

#normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_data=sc.fit_transform(X_data)

#X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


from sklearn.manifold import Isomap
iso = Isomap(n_components=NoComp,n_neighbors=20)
X_train = iso.fit_transform(X_train, y_train)
X_test = iso.transform(X_test)

X_t = X_train.transpose()
COV = np.matmul(X_t,X_train)
from scipy import linalg as LA
e_vals, e_vecs = LA.eig(COV)

i = 1
for item in e_vals:
    Variability=item/sum(e_vals)
    print("Component " + str(i) + " : " +Variability)
    i = i + 1
