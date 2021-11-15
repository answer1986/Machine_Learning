# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:51:29 2019

@author: simran
"""

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

variability=e_vals[NoComp-1]/sum(e_vals)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score

#randomforest

classifier_randomforest=RandomForestClassifier()
param_grid_randomforest = {
    'max_depth': [100, 150,200],
    'n_estimators': [100,125,150]
}

#tuning paramter of random forest 
grid_search = GridSearchCV(estimator = classifier_randomforest, param_grid=param_grid_randomforest, cv = 10, n_jobs = -1, verbose = 2)
grid_search.fit(X_train,y_train)

optimal_classifier_randomforest=RandomForestClassifier(**grid_search.best_params_)
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

classifier_knn=KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': [1],
    "metric": ["euclidean", "cityblock"],"algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute']
}

#tuning parameter of knn
grid_search1 = GridSearchCV(estimator = classifier_knn, param_grid=param_grid_knn, cv = 10, n_jobs = -1, verbose = 2)
grid_search1.fit(X_train,y_train)

optimal_classifier_knn=KNeighborsClassifier(**grid_search1.best_params_)
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

classifier_mlp=MLPClassifier(max_iter=10)

param_grid_mlp = {'hidden_layer_sizes': [(600,600) ],
    'activation': ['tanh'],
    'solver':['lbfgs'],
     'alpha':[.05],
     'learning_rate': ['constant']
}

grid_search2 = GridSearchCV(estimator = classifier_mlp, param_grid=param_grid_mlp, cv = 2, n_jobs = -1, verbose = 2)
#classifier_randomforest = RandomForestClassifier( max_depth=150, n_estimators=128 )
#classifier_knn= KNeighborsClassifier(n_neighbors=2 ,leaf_size=30)
grid_search2.fit(X_train,y_train)



optimal_classifier_mlp=MLPClassifier(**grid_search2.best_params_)
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
RBFClassifier = SVC(kernel='rbf')
param_grid_rbf = {
      'kernel':['rbf'],

     'gamma': [ .1,1],
                     'C': [100000,1000]
                   
}

#tuning paramter of radial basis
grid_search3 = GridSearchCV(estimator = RBFClassifier, param_grid=param_grid_rbf, cv = 10, n_jobs = -1, verbose = 2)
grid_search3.fit(X_train,y_train)

optimal_classifier_rbf=SVC(**grid_search3.best_params_)
optimal_classifier_rbf.fit(X_train, y_train)
#y_pred3 = optimal_classifier_rbf.predict(X_test)
#cm3 = confusion_matrix(y_test, y_pred3)
#print(cn3)
#print('Accuracy using mlp ' + str(accuracy_score(y_test, y_pred3)))

Accuracy_rbf = cross_val_score(optimal_classifier_rbf, X_train, y_train, cv=10)
#print (Accuracy)
f1_score_rbf=cross_val_score(optimal_classifier_rbf, X_train, y_train, cv=10,scoring='f1_macro')

print("Variability ---", variability)
print('Random Forest -----------')
print('optimal parameters: ',grid_search.best_params_)
best_result = grid_search.best_score_
print('Best result : ', best_result)
j=1
for i in Accuracy_rf:
  print ("Fold-"+ str(j)+" Accuracy - "+str(i))
  j=j+1

print ("mean accuracy-"+str(np.mean(Accuracy_rf)))

j=1
for i in f1_score_rf:
  print ("Fold-"+ str(j)+" F1-score- "+str(i))
  j=j+1
print ("mean f1-score-"+str(np.mean(f1_score_rf)))




print('k- Nearest  Neighbor -----------')
print('optimal parameters:',grid_search1.best_params_)
best_result1 = grid_search1.best_score_
print('Best result : ', best_result1)

j=1
for i in Accuracy_knn:
  print ("Fold-"+ str(j)+" Accuracy - "+str(i))
  j=j+1

print ("mean accuracy-"+str(np.mean(Accuracy_knn)))

j=1
for i in f1_score_knn:
  print ("Fold-"+ str(j)+" F1-score- "+str(i))
  j=j+1
print ("mean f1-score-"+str(np.mean(f1_score_knn)))




print('Multilayer Perceptron -----------')
print('optimal parameters: ', grid_search2.best_params_)
best_result2 = grid_search2.best_score_
print('Best result : ',best_result2)
j=1
for i in Accuracy_mlp:
  print ("Fold-"+ str(j)+" Accuracy - "+str(i))
  j=j+1

print ("mean accuracy-"+str(np.mean(Accuracy_mlp)))

j=1
for i in f1_score_mlp:
  print ("Fold-"+ str(j)+" F1-score- "+str(i))
  j=j+1
print ("mean f1-score-"+str(np.mean(f1_score_mlp)))



print('Radial Basis Function -----------')
print('optimal parameters: ', grid_search3.best_params_)
best_result3 = grid_search3.best_score_
print('Best result : ',best_result3)
j=1
for i in Accuracy_rbf:
  print ("Fold-"+ str(j)+" Accuracy - "+str(i))
  j=j+1

print ("mean accuracy-"+str(np.mean(Accuracy_rbf)))

j=1
for i in f1_score_rbf:
  print ("Fold-"+ str(j)+" F1-score- "+str(i))
  j=j+1
print ("mean f1-score-"+str(np.mean(f1_score_rbf)))