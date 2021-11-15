# -*- coding: utf-8 -*-

"""
Created on Thu Jul 18 20:41:15 2019

@author: simran
"""
#import  RBFN  as rbf
import numpy as np
import pandas as pd
from sklearn.svm import SVC


data = pd.read_excel('Datasets/Gear17.xlsx')
# Dataset is now stored in a Pandas Dataframe
data = data.dropna(axis = 1, how ='all') 

#removing columns havin more than 90% zero
data=data.drop(columns=data.columns[((data==0).mean()>0.90)],axis=1)
print(data.info)

#saving lables in y
X_data= data.iloc[:,:-1].values
y_data=data.iloc[:,-1].values

#Normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_data=sc.fit_transform(X_data)

#splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#radial basis function
RBFClassifier = SVC(kernel='rbf')
param_grid_rbf = {
      'kernel':['rbf'],
     'gamma': [ 1e-4,0.00001],
                     'C': [100,1000,10000]
                   
}

#tuning paramter of radial basis
grid_search = GridSearchCV(estimator = RBFClassifier, param_grid=param_grid_rbf, cv = 2, n_jobs = -1, verbose = 2)
grid_search.fit(X_train,y_train)

optimal_classifier_rbf=SVC(**grid_search.best_params_)
optimal_classifier_rbf.fit(X_train, y_train)

Accuracy_rbf = cross_val_score(optimal_classifier_rbf, X_train, y_train, cv=10)
#print (Accuracy)
f1_score_rbf=cross_val_score(optimal_classifier_rbf, X_train, y_train, cv=10,scoring='f1_macro')

print('Radial Basis Function -----------')
print('optimal parameters: ', grid_search.best_params_)
best_result2 = grid_search.best_score_
print('Best result : ',best_result2)
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








#scores = ['accuracy']
#
#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                       scoring= score)
#    clf.fit(X_train, y_train)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
#    print()
#
#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = y_test, clf.predict(X_test)
#    print(classification_report(y_true, y_pred))
#    print()


