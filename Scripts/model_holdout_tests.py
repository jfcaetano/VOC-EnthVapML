### ΔvapH° ML modelling
### JFCAETANO 2023
### MIT Licence

import csv, math
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
my_model = ["MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=5000,n_iter_no_change=200, random_state=47)", "RandomForestRegressor(n_estimators= 250, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', max_depth= 35, bootstrap= False, random_state=47)", "GradientBoostingRegressor(n_estimators= 300, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', max_depth= 5, random_state=47)", "svm.SVR(C= 6000, gamma= 0.000001)"]

#select chemical groups to test
my_types = ["Benzenoids", "Hydrocarbons", "Lipids and lipid-like molecules", "Organic acids and derivatives", "Organic nitrogen compounds", "Organic oxygen compounds", "Organohalogen compounds", "Organoheterocyclic compounds"]

o = list()
#munber of trials
my_list = [1]
for time in my_list:
    for group in my_types:
        data_train = pd.read_csv('Model_Database_x.csv')
        data_test=data_train.loc[data_train['Type'] == (f"{group}")]
        data_train.drop(data_train[data_train['Type'] == (f"{group}")].index, inplace = True)

        y_train=data_train.loc[:,"dvap"]
        exclude_cols=['External Database','SMILES','dvap', 'Type']
        X_names=[x for x in data_train.columns if x not in exclude_cols]
        X_train = data_train.loc[:,X_names]
        X_train = X_train.replace(np.nan,0)

        y_test=data_test.loc[:,"dvap"]
        exclude_cols=['External Database','SMILES','dvap', 'Type']
        X_names=[x for x in data_test.columns if x not in exclude_cols]
        X_test = data_test.loc[:,X_names]
        X_test = X_test.replace(np.nan,0)
    
        for mod in my_model:    
            model = eval(f"{mod}")
            model.fit(X_train, y_train)
            y_train_fitted=model.predict(X_train)
            y_test_fitted=model.predict(X_test)
            rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
            Score_train = model.score(X_train, y_train)
            Score_test = model.score(X_test, y_test)
            MAE=mean_absolute_error(y_test,y_test_fitted)
            cv = RepeatedKFold(n_splits=4, n_repeats=10, random_state=47)
            n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
            STD=std(n_scores)
            AARD=(100/len(X_test))*(sum(abs((y_test_fitted-y_test)/y_test_fitted)))        
    
            nl = dict()
            nl[f"Type"]=f"{group}"
            nl[f"Algorithm"]=f"{mod}"
            nl[f"Score_test"]=Score_test
            nl[f"Score_train"]=Score_train
            nl[f"MAE"]=MAE
            nl[f"AARD"]=AARD
            nl[f"STD"]=STD

            o.append(nl)

output_fn = 'training_holdout_models.csv'
with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)
