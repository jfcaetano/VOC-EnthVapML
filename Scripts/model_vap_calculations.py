### ΔvapH° ML modelling
### JFCAETANO 2023
### MIT Licence

import csv, math, sys
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


#select algorithms to test
my_model = ["linear_model.LinearRegression()","RandomForestRegressor(random_state=47)", "GradientBoostingRegressor(random_state=47)", "svm.SVR()", "MLPRegressor(random_state=47)"]

o = list()
#munber of trials
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for time in my_list:
    data = pd.read_csv('Model_Database_x.csv')
    y=data.loc[:,"dvap"]
    exclude_cols=['External Database','SMILES','dvap', 'Type'] #select descriptors
    X_names=[x for x in data.columns if x not in exclude_cols]
    X = data.loc[:,X_names]
    X=X.to_numpy()
    X=np.nan_to_num(X, nan=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #select train/test ratio
    for mod in my_model:    
        model = eval(f"{mod}")
        model.fit(X_train, y_train)
        y_train_fitted=model.predict(X_train)
        y_test_fitted=model.predict(X_test)
        rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
        Score_train = model.score(X_train, y_train)
        Score_test = model.score(X_test, y_test)
        MAE=mean_absolute_error(y_test,y_test_fitted)
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=47)
        n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD=std(n_scores)
        AARD=(100/len(X_test))*(sum(abs((y_test_fitted-y_test)/y_test_fitted)))        
    
        nl = dict()
        nl[f"Algorithm"]=f"{mod}"
        nl[f"Score_test"]=Score_test
        nl[f"Score_train"]=Score_train
        nl[f"MAE"]=MAE
        nl[f"AARD"]=AARD
        nl[f"STD"]=STD

        o.append(nl)

#output file
output_fn = 'training_models.csv'
with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)


### Pipeline
# Calcualte best hyperparameters
#Example for Random Forest

steps = [('scaler', StandardScaler()), ('Forest', RandomForestRegressor())]
pipeline = Pipeline(steps)

parameters = {'Forest__n_estimators': [100, 150, 200, 250],
              'Forest__max_features': ['sqrt','log2'],
              'Forest__min_samples_split': [2, 5, 10],
              'Forest__min_samples_leaf': [1, 2, 5, 10],
              'Forest__max_depth': [10, 15, 20, 25, 30, 35, 40, 45, 50],
              'Forest__bootstrap': [True, False],
              'Forest__warm_start': [True, False]}

model = RandomizedSearchCV(pipeline, parameters, n_iter=10, scoring='neg_mean_absolute_error', cv=10)

model.fit(X_train, y_train)
Best_Parameters = model.best_params_


#select optimized descriptors
my_model = ["MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=5000,n_iter_no_change=200, random_state=47)", "RandomForestRegressor(n_estimators= 250, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', max_depth= 35, bootstrap= False, random_state=47)", "GradientBoostingRegressor(n_estimators= 300, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', max_depth= 5, random_state=47)", "svm.SVR(C= 6000, gamma= 0.000001)"]

o = list()
#munber of trials
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for time in my_list:
    data = pd.read_csv('Model_Database_x.csv')
    y=data.loc[:,"dvap"]
    exclude_cols=['External Database','SMILES','dvap', 'Type']
    X_names=[x for x in data.columns if x not in exclude_cols]
    X = data.loc[:,X_names]
    X=X.to_numpy()
    X=np.nan_to_num(X, nan=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for mod in my_model:    
        model = eval(f"{mod}")
        model.fit(X_train, y_train)
        y_train_fitted=model.predict(X_train)
        y_test_fitted=model.predict(X_test)
        rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
        Score_train = model.score(X_train, y_train)
        Score_test = model.score(X_test, y_test)
        MAE=mean_absolute_error(y_test,y_test_fitted)
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=47)
        n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD=std(n_scores)
        AARD=(100/len(X_test))*(sum(abs((y_test_fitted-y_test)/y_test_fitted)))        
    
        nl = dict()
        nl[f"Algorithm"]=f"{mod}"
        nl[f"Score_test"]=Score_test
        nl[f"Score_train"]=Score_train
        nl[f"MAE"]=MAE
        nl[f"AARD"]=AARD
        nl[f"STD"]=STD

        o.append(nl)

output_fn = 'training_models_opt.csv'
with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)


