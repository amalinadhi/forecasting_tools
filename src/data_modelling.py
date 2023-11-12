import copy
import utils
import numpy as np
import pandas as pd

from itertools import product
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error


# Function here
def time_series_search_cv(estimator=XGBRegressor):
    """Perform hyperparameter search with timeseries CV"""
    # Initialize
    param_list = []
    error_train_list = []
    error_test_list = []
    estimator_ = copy.deepcopy(estimator)
    cols = CONFIG_FILE['features_col']
    n_splits = CONFIG_FILE['n_splits']
    test_size = CONFIG_FILE['test_size']
    param_grid = CONFIG_FILE['param_grid']

    # Load Data
    X = utils.pickle_load(CONFIG_FILE['X_train_clean_path'])
    y = utils.pickle_load(CONFIG_FILE['y_train_path'])
    
    # Create all combination
    combi_list = product(*[val for _, val in param_grid.items()])
    
    # Generate all time series cv
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    # Iterate over all parameter combination
    for k, vals in enumerate(combi_list):
        # Get the combi key
        combi = {key:val for key, val in zip(param_grid.keys(), vals)}
        
        # Iterate over timeseries splitting
        error_train_cv = []
        error_test_cv = []
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Get the train & test data
            X_train_i = X[cols].iloc[train_idx]
            y_train_i = y.iloc[train_idx]
            X_test_i = X[cols].iloc[test_idx]
            y_test_i = y.loc[test_idx]
            
            # Initate the model
            mdl = estimator_(**combi)
            
            # Train model
            mdl.fit(X_train_i, y_train_i)
            
            # Predict for evaluation
            y_train_pred = mdl.predict(X_train_i)
            y_test_pred = mdl.predict(X_test_i)
            
            # Evaluate
            mape_train = mean_absolute_percentage_error(y_train_i, y_train_pred)
            mape_test = mean_absolute_percentage_error(y_test_i, y_test_pred)
            
            # append results
            error_train_cv.append(mape_train)
            error_test_cv.append(mape_test)
            
        # Average the error_train & error_test
        error_train_cv_avg = np.mean(error_train_cv)
        error_test_cv_avg = np.mean(error_test_cv)
        
        # Append
        error_train_list.append(error_train_cv_avg)
        error_test_list.append(error_test_cv_avg)
        param_list.append(combi)
        
        # Print out
        print(f'try: {k:2d}, combi_test: {combi}, train_cv: {error_train_cv_avg:.4e}, test_cv: {error_test_cv_avg:.4e}')
        
    return param_list, error_train_list, error_test_list
        
def choose_best_model(param_list, error_train_list, error_test_list, is_return=False):
    # Load config file data
    X = utils.pickle_load(CONFIG_FILE['X_train_clean_path'])
    y = utils.pickle_load(CONFIG_FILE['y_train_path'])
    cols = CONFIG_FILE['features_col']
    
    # Get model summary
    tuning_df = pd.DataFrame({
        'params': param_list,
        'cv_train': error_train_list,
        'cv_test': error_test_list
    })

    # Find the best parameters
    best_param = tuning_df['params'].iloc[tuning_df['cv_test'].argmin()]
    print('Best parameters:', best_param)

    # Retrain the model
    best_model = XGBRegressor(**best_param)
    best_model.fit(X[cols], y)

    # Dump model
    utils.pickle_dump(best_model, CONFIG_FILE['best_model_path'])

    if is_return:
        return best_model


# Run file
if __name__ == '__main__':
    # Load config
    CONFIG_FILE = utils.config_load()
    
    # Find best model
    param_list, error_train_list, error_test_list = time_series_search_cv()
    choose_best_model(param_list, error_train_list, error_test_list)

