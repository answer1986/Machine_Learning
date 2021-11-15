# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#dataset = pd.read_csv(url, names=names)
dff= pd.read_excel('Datasets/Gear10.xlsx')
#df= pd.read_excel('Gear1.xlsx')
#dff = sc.fit_transform(dff)
#dff = pd.DataFrame(data=dff,columns=dataset.columns)
#print(dataset)
dff = dff.dropna(axis = 1, how ='all') 
NCOMP = 15
#removing columns havin more than 90% zero
dff=dff.drop(columns=dff.columns[((dff==0).mean()>0.90)],axis=1)
X = dff.iloc[:, :-1].values
y = dff.iloc[:, -1].values

from scipy.stats import variation

X = sc.fit_transform(X)
#X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)




from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import manifold
#from scipy import mis
#import Sammon




lda = LDA(n_components=NCOMP)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
print(X_train.shape)
X_t = X_train.transpose()
COV = np.matmul(X_t,X_train)


from scipy import linalg as LA
e_vals, e_vecs = LA.eig(COV)
print("Eigen Value ------", len(e_vals))
print("Variability ---", e_vals[NCOMP-1]/sum(e_vals))