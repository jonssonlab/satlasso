"""
Test code for satlasso module

Can be run with py.test
"""

import os
import pytest

import numpy as np
from satlasso import SatLasso

def create_regression_data():
    """ helper function to create regression data for testing """
    from sklearn.datasets import make_regression
    
    # make regression dataset with sklearn
    X, y, coef = make_regression(n_samples = 500, coef = True)
    
    ## make saturation data
    # get maximum value of y
    max_y = max(y)
    idxmax_y = np.argmax(y)
    # add minimal noise to corresponding features in X
    max_feat = X[idxmax_y, :]
    append_X = (max_feat + np.random.randn(50, X.shape[1]))
    # append saturation data to X and y
    X = np.vstack((X, append_X))
    y = np.concatenate((y, np.repeat(max_y, 50)))
    
    return X, y, coef

def test_satlasso_sanity():
    """ sanity check on satlasso fitted data """
    X, y, _ = create_regression_data()
    
    regressor = SatLasso()
    
    regressor.fit(X, y)
    y_hat = regressor.predict(X)
    
    assert y_hat.dtype == y.dtype
    assert len(y_hat) == len(y)

def test_satlasso_score():
    """ test R^2 score of satlasso regressor """
    regressor = SatLasso()
    
    X, y, coef = create_regression_data()
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
    regressor.fit(X_train, y_train)
    
    
    score_train = regressor.score(X_train, y_train)
    score_test = regressor.score(X_test, y_test)
    
    assert score_train > 0.7
    assert score_test > 0.5
    
def test_satlasso_coef():
    """ test similar coefficients found by satlasso regressor to coefficients used to generate data """
    regressor = SatLasso()

    X, y, coef = create_regression_data()
    
    regressor.fit(X, y)

    true_nonzero_coefs = np.where(coef != 0)[0].tolist()
    regressor_nonzero_coefs = np.where(regressor.coef_ != 0)[0].tolist()
    
    assert all(item in regressor_nonzero_coefs for item in true_nonzero_coefs) or all(item in true_nonzero_coefs for item in regressor_nonzero_coefs)
