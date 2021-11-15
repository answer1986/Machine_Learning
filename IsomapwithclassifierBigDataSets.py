# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:51:29 2019

@author: simran
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
NoComp = 4
data= pd.read_excel('Datasets/Gear16.xlsx')
# Dataset is now stored in a Pandas Dataframe
data = data.dropna(axis = 1, how ='all') 

#removing columns havin more than 90% zero
data=data.drop(columns=data.columns[((data==0).mean()>0.90)],axis=1)

#saving lables in y
X_data= data.iloc[:,:-1].values
y_data=data.iloc[:,-1].values

#normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_data=sc.fit_transform(X_data)

#X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data.astype(int), y_data.astype(int), test_size=0.2, random_state=0)


from sklearn.manifold import Isomap
iso = Isomap(n_components=NoComp)
X_train = iso.fit_transform(X_train, y_train)
X_test = iso.transform(X_test)

X_t = X_train.transpose()
COV = np.matmul(X_t,X_train)
from scipy import linalg as LA
e_vals, e_vecs = LA.eig(COV)

print("Variability ---", e_vals[NoComp-1]/sum(e_vals))

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score

#randomforest

def PrintResults(Accuracy, f1):
    Accuracy = Accuracy * 10
    f1 = f1 * 10 
    j=1
    for i in Accuracy:
      print ("Fold-"+ str(j)+" Accuracy - "+str(i))
      j=j+1
    
    print ("mean accuracy-"+str(np.mean(Accuracy)))
    
    j=1
    for i in f1:
      print ("Fold-"+ str(j)+" F1-score- "+str(i))
      j=j+1
    print ("mean f1-score-"+str(np.mean(f1)))
#tuning paramter of random forest 

optimal_classifier_randomforest=RandomForestClassifier(max_depth=200,n_estimators=100)
optimal_classifier_randomforest.fit(X_train, y_train)
#y_pred = optimal_classifier_randomforest.predict(X_test)
#cm = confusion_matrix(y_test, y_pred)
#print(cn)
#print('Accuracy using random forest ' + str(accuracy_score(y_test, y_pred)))

Accuracy_rf = cross_val_score(optimal_classifier_randomforest, X_train, y_train, cv=10)
#print (Accuracy)

f1_score_rf=cross_val_score(optimal_classifier_randomforest, X_train, y_train, cv=10,scoring='f1_macro')
#print (f1_score)


# K-nn

from sklearn.neighbors import KNeighborsClassifier




optimal_classifier_knn=KNeighborsClassifier(n_neighbors=1)
optimal_classifier_knn.fit(X_train, y_train)
#y_pred1 = optimal_classifier_knn.predict(X_test)
#cm1 = confusion_matrix(y_test, y_pred1)
#print(cn1)
#print('Accuracy using knn ' + str(accuracy_score(y_test, y_pred1)))
Accuracy_knn = cross_val_score(optimal_classifier_knn, X_train, y_train, cv=10)
#print (Accuracy)
f1_score_knn=cross_val_score(optimal_classifier_knn, X_train, y_train, cv=10,scoring='f1_macro')
#print (f1_score)

#Mlp Classifier
from sklearn.neural_network import MLPClassifier

#classifier_mlp=MLPClassifier()
#
#}




optimal_classifier_mlp=MLPClassifier(activation='tanh',alpha=1e-05,solver='lbfgs')
optimal_classifier_mlp.fit(X_train, y_train)

#y_pred2 = optimal_classifier_mlp.predict(X_test)
#cm2 = confusion_matrix(y_test, y_pred2)
#print(cn2)
#print('Accuracy using mlp ' + str(accuracy_score(y_test, y_pred2)))
Accuracy_mlp = cross_val_score(optimal_classifier_mlp, X_train, y_train, cv=10)
#print (Accuracy)
f1_score_mlp=cross_val_score(optimal_classifier_mlp, X_train, y_train, cv=10,scoring='f1_macro')
#print (f1_score)


#radial basis function
#RBFClassifier = SVC(kernel='rbf')


optimal_classifier_rbf=SVC(C=100000,gamma=0.0001,kernel='rbf')
optimal_classifier_rbf.fit(X_train, y_train)
#y_pred3 = optimal_classifier_rbf.predict(X_test)
#cm3 = confusion_matrix(y_test, y_pred3)
#print(cn3)
#print('Accuracy using mlp ' + str(accuracy_score(y_test, y_pred3)))

Accuracy_rbf = cross_val_score(optimal_classifier_rbf, X_train, y_train, cv=10)
#print (Accuracy)
f1_score_rbf=cross_val_score(optimal_classifier_rbf, X_train, y_train, cv=10,scoring='f1_macro')
print("Variability ---", e_vals[NoComp-1]/sum(e_vals))
print('Random Forest -----------')


PrintResults(Accuracy_rf,f1_score_rf)




print('k- Nearest  Neighbor -----------')



PrintResults(Accuracy_knn,f1_score_knn)



print('Multilayer Perceptron -----------')

PrintResults(Accuracy_mlp,f1_score_mlp)




print('Radial Basis Function -----------')

PrintResults(Accuracy_rbf,f1_score_rbf)
