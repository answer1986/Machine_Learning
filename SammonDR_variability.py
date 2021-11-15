# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 03:18:14 2019

@author: simran
"""
import numpy as np
from sklearn.svm import SVC
from sammon import sammon
import numpy as np
import pandas as pd
NoComp=15
data = pd.read_excel('Datasets/Gear15.xlsx')
# Dataset is now stored in a Pandas Dataframe
data = data.dropna(axis = 1, how ='all')
 
def correlationdatset(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    
#removing columns havin more than 90% zero
data=data.drop(columns=data.columns[((data==0).mean()>0.7)],axis=1)
print(data.info)
data = data.drop_duplicates()

#saving lables in y
X_data= data.iloc[:,:-1].values
y_data=data.iloc[:,-1].values

#Normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_data=sc.fit_transform(X_data)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



   # Run the Sammon projection
[X_train,E] = sammon(X_train, NoComp, maxhalves = 20, maxiter = 10, init='pca')
#print(X_train.shape)
X_t = X_train.transpose()
COV = np.matmul(X_t,X_train)
from scipy import linalg as LA
e_vals, e_vecs = LA.eig(COV)
i = 1
for item in e_vals:
    Variability=item/sum(e_vals)
    print("Componet " + str(i) + " : " +Variability)
    i = i + 1
