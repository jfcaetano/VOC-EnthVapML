#dVAP MODEL FAMILY AND GROUP TESTING

import csv, math
import pandas as pd
import numpy as np
from numpy import nan_to_num, std
from sklearn import linear_model, ensemble, svm, neural_network
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV


# Load descriptor data
desc_data = pd.read_csv('Database/desc_VOC-RF.csv')

# Load VOC data
voc_data = pd.read_csv('Database/VOC-Database.csv')

# Extract lists for each descriptor type
vsa = desc_data['VSA'].dropna().tolist()
structural = desc_data['Structural'].dropna().tolist()
electronic = desc_data['Electronic'].dropna().tolist()
combined = vsa + structural + electronic

exclude_cols = ['CAS', 'VOC', 'dvap', 'num', 'External', 'SMILES', 'Key', 'Family']
complete = [col for col in voc_data.columns if col not in exclude_cols]

# Define descriptor sets
descriptor_sets = {'Complete': combined}


# Initialize Random Forest model
model = RandomForestRegressor(n_estimators= 300, min_samples_split= 2, min_samples_leaf= 1, max_depth= 20, random_state=47)

results = []

# Specify the target classes to test individually

target_classes = ['Benzenoids', 'Hydrocarbons', 'Organic oxygen compounds', 'Organic acids and derivatives', 'Organohalogen compounds', 'Organic nitrogen compounds', 'Organoheterocyclic compounds']  # Add your target classes here

# Loop over each target class
for target_class in target_classes:
    # Split data for the current target class
    train_data = voc_data[(voc_data['VOC'] == 'NO') & (voc_data['Family'] != target_class)]
    test_data = voc_data[(voc_data['VOC'] == 'YES') | (voc_data['Family'] == target_class)]

    # Evaluate model for each descriptor set
    for desc_name, descriptors in descriptor_sets.items():
        if not all(item in voc_data.columns for item in descriptors):
            continue  # Skip if any descriptor is not in voc_data

        X_train = train_data[descriptors].to_numpy()
        y_train = train_data['dvap'].to_numpy()
        X_test = test_data[descriptors].to_numpy()
        y_test = test_data['dvap'].to_numpy()

        X_train = nan_to_num(X_train, nan=0)
        X_test = nan_to_num(X_test, nan=0)

        # Fit model and make predictions
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        rsq_train = model.score(X_train, y_train)
        rsq_test = model.score(X_test, y_test)
        Score_train = np.corrcoef(y_train, y_train_pred)[0, 1] ** 2
        Score_test = np.corrcoef(y_test, y_test_pred)[0, 1] ** 2
        MSE = np.square(np.subtract(y_test, y_test_pred)).mean()
        RMSE = math.sqrt(MSE)
        cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=47)
        n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD = std(n_scores)
        MAE = mean_absolute_error(y_test, y_test_pred)
        AARD = (100 / len(X_test)) * sum(abs((y_test_pred - y_test) / y_test_pred))

        # Record results
        result = {
            "Descriptor Set": desc_name,
            "Family": target_class,
            "Score Train": rsq_train,
            "Score Test": rsq_test,
            "R2 Train": Score_train,
            "R2 Test": Score_test,
            "RMSE": RMSE,
            "MAE": MAE,
            "STD": STD,
            "AARD": AARD,
            "N Descriptors": len(descriptors)}
        results.append(result)

# Output file
output_fn = 'VOC_performance_descriptor_sets.csv'
with open(output_fn, 'w', newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

########################






