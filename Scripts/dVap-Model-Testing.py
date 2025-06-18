#dVAP MODEL AND HYPERPARAMETER OPTIMIZATION
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Algorithms to test
models = [
    linear_model.LinearRegression(),
    RandomForestRegressor(random_state=47),
    ensemble.GradientBoostingRegressor(random_state=47),
    svm.SVR(),
    neural_network.MLPRegressor(random_state=47)]

# Load data
data = pd.read_csv('Database/VOC-Database.csv')

# Exclude columns
exclude_cols = ['CAS', 'VOC', 'dvap', 'num', 'External', 'SMILES', 'Key', 'Family']
X_names = [col for col in data.columns if col not in exclude_cols]
X = data.drop(columns=exclude_cols)
y = data['dvap'].to_numpy()

results = []

# Number of iterations per model
n_iterations = 10

for model in models:
    for iteration in range(n_iterations):
        # Split data based on 'VOC' column
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60)
        
        model.fit(X_train, y_train)

        # Calculate feature importance
        p_imp = permutation_importance(model, X_test, y_test, n_repeats=5)
        p_imp_av = p_imp['importances_mean']
        feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': p_imp_av})

        # Export feature importance to CSV
        #feature_importance_fn = f'feature_importance_60{model.__class__.__name__}{iteration}.csv'
        #feature_importance_df.to_csv(feature_importance_fn, index=False)

        # Filter out bad features
        relative_importance = (p_imp_av / sum(p_imp_av))
        bad_features = [name for name, importance in zip(X_train.columns, relative_importance) if importance < 0.00001]
        good_features = [name for name in X_train.columns if name not in bad_features]
        X_train_filtered1 = X_train[good_features]
        X_test_filtered1 = X_test[good_features]

        # Convert to numpy arrays and replace NaNs
        X_train_filtered = nan_to_num(X_train_filtered1.to_numpy(), nan=0)
        X_test_filtered = nan_to_num(X_test_filtered1.to_numpy(), nan=0)

        # Fit model and make predictions
        model.fit(X_train_filtered, y_train)
        y_train_pred = model.predict(X_train_filtered)
        y_test_pred = model.predict(X_test_filtered)
        
        # Calculate metrics
        rsq_train = model.score(X_train_filtered, y_train)
        rsq_test = model.score(X_test_filtered, y_test)
        Score_train = np.corrcoef(y_train, y_train_pred)[0, 1] ** 2
        Score_test = np.corrcoef(y_test, y_test_pred)[0, 1] ** 2
        MSE = np.square(np.subtract(y_test, y_test_pred)).mean()
        RMSE = math.sqrt(MSE)
        cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=iteration)
        n_scores = cross_val_score(model, X_test_filtered, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD = std(n_scores)
        MAE = mean_absolute_error(y_test, y_test_pred)
        AARD = (100 / len(X_test_filtered)) * sum(abs((y_test_pred - y_test) / y_test_pred))
        
        
        # Create a DataFrame for X_test_filtered
        #X_test_filtered_df = pd.DataFrame(X_test_filtered, columns=good_features)

        # Convert y_test and y_test_pred to DataFrames
        #y_test_df = pd.DataFrame(y_test, columns=['y_test'])
        #y_test_pred_df = pd.DataFrame(y_test_pred, columns=['y_test_pred'])

        # Calculate the percentage error
        #percentage_error = (abs(y_test_df['y_test'] - y_test_pred_df['y_test_pred']) / y_test_df['y_test']) * 100
        #percentage_error_df = pd.DataFrame(percentage_error, columns=['Percentage Error'])

        # Combine all the data into one DataFrame
        #export_df = pd.concat([X_test_filtered_df, y_test_df, y_test_pred_df, percentage_error_df], axis=1)

        # Export to CSV
        #export_filename = 'test_predictions_with_error.csv'
        #export_df.to_csv(export_filename, index=False)
        
        # Record results
        result = {
            "Algorithm": model.__class__.__name__,
            "Iteration": iteration + 1,
            "rsq_train": rsq_train,
            "rsq_test": rsq_test,
            "Score_train": Score_train,
            "Score_test": Score_test,
            "RMSE": RMSE,
            "MAE": MAE,
            "STD": STD,
            "AARD": AARD,
            "N Bad Features": len(bad_features)}
        results.append(result)

# Output file
output_fn = 'Database/dVAP-Results.csv'
with open(output_fn, 'w', newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)



### Pipeline
# Calcualte best hyperparameters
#Example for Random Forest

# Load VOC data
voc_data = pd.read_csv('Database/VOC-Database.csv')

# Define a descriptor set (example: using VSA descriptors)
exclude_cols = ['CAS', 'VOC', 'dvap', 'num', 'External', 'SMILES', 'Key', 'Family']
complete = [col for col in voc_data.columns if col not in exclude_cols]
X = voc_data[complete]
y = voc_data['dvap']

# Fill missing values
X = X.fillna(X.mean())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=47)

# Define a pipeline
pipeline = Pipeline([('rf', RandomForestRegressor(random_state=47))])

# Define a parameter grid to search
param_grid = {
    'rf__n_estimators': [100, 150, 200, 300],
    'rf__max_depth': [10, 20, 30, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions and evaluate
y_pred = best_model.predict(X_test)
print("Best Model Parameters:", grid_search.best_params_)
print("R^2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
