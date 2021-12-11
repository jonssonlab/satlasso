import numpy as np
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

from satlasso import SatLasso, SatLassoCV

aalist = ['A', 'R', 'N', 'D','C','Q','E','G','H', 'I','L','K','M', 'F','P','S', 'T', 'W', 'Y' ,'V']

def get_data():
    # read in metadata to get values
    metadata = pd.read_csv('NeutSeqData_3BNC117.csv', sep=',', header=0)
    
    # retrieve data and target values to use for training regression
    sequences = np.stack(metadata['sequence'].apply(list).values)
    values = metadata['ic50_ngml'].values
    
    # one-hot encode amino acid sequences and save as sparse matrix
    encoder = OneHotEncoder(categories = np.tile(aalist, (sequences.shape[1],1)).tolist(), sparse = False, handle_unknown = "ignore")
    data = encoder.fit_transform(sequences.tolist())
    
    return data, values
    
def calculate_sat_error(predicted, values, saturation_val):
    # get indices of saturated values
    idx_saturation = np.where(values == saturation_val)
    
    # create mask for non-saturated data
    mask = np.ones(values.shape, dtype=bool)
    mask[idx_saturation] = False
    
    # calculate non-saturated squared error and saturated squared error separately (not reduced)
    non_sat_mse = predicted[mask] - values[mask] ** 2
    sat_mse = np.minimum(predicted[np.logical_not(mask)] - values[np.logical_not(mask)], 0) ** 2
    
    # combine and take mean
    mse = np.concatenate((non_sat_mse, sat_mse)).mean()
    
    return mse
    
# satlasso code
def run_satlasso(data, values):
    # set up satlasso object
    regressor = SatLasso(lambda_1 = 1., lambda_2 = 7.75, lambda_3 = 10., saturation='max', fit_intercept=True, normalize=False, copy_X=True)
    
    # fit satlasso regressor with data and values
    regressor.fit(data, values)
    
    # get coefficients, intercept
    coef, intercept = regressor.coef_, regressor.intercept_
    
    # get predicted values
    predicted = regressor.predict(data)
    
    # get saturated error of prediction
    saturation_val = regressor.saturation_val_
    error = calculate_sat_error(predicted, values, saturation_val)

    return coef, intercept, predicted, error

# satlassocv code
def run_satlassocv(data, values):
    lambda1s = np.linspace(1,10,3)
    lambda2s = np.linspace(1,10,3)
    lambda3s = np.linspace(1,10,3)
    
    # set up satlasso cross-validation object
    regressorcv = SatLassoCV(lambda_1s = lambda1s, lambda_2s = lambda2s, lambda_3s = lambda3s, saturation='max', fit_intercept=True, normalize=False, copy_X=True, cv=3)
    
    # fit satlasso regressor with data and values
    regressorcv.fit(data, values)
    
    # get coefficients, intercept, optimal lambdas and cross-validation error for lambda combinations tested
    coef, intercept, lambda1, lambda2, lambda3, mse_dict = regressorcv.coef_, regressorcv.intercept_, regressorcv.lambda_1_, regressorcv.lambda_2_, regressorcv.lambda_3_, regressorcv.mse_dict_
    
    # get predicted values
    predicted = regressorcv.predict(data)
    
    # get saturated error of prediction
    saturation_val = regressorcv.saturation_val_
    error = calculate_sat_error(predicted, values, saturation_val)

    return coef, intercept, (lambda1, lambda2, lambda3), mse_dict, predicted, error

def main():
    # load data
    data, values = get_data()
    
    # run satlasso, get metrics and output of satlasso
    coef, intercept, predicted, error = run_satlasso(data, values)
    
    # run satlassocv, get metrics and output of satlassocv
    cv_coef, cv_intercept, optimal_lambdas, mse_dict, cv_predicted, cv_error = run_satlassocv(data, values)

if __name__ == "__main__":
    main()
